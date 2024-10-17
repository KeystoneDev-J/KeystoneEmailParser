# src/parsers/enhanced_parser.py

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import phonenumbers
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser
from email import message_from_string
from phonenumbers import PhoneNumberFormat
from thefuzz import fuzz
from transformers import pipeline
from huggingface_hub import login

from src.utils.config_loader import ConfigLoader
from src.utils.quickbase_schema import QUICKBASE_SCHEMA
from src.utils.security import sanitize_content
from src.utils.validation import validate_json

# Constants
LLM_TIMEOUT_SECONDS = 500
RATE_LIMIT = 5
CACHE_EXPIRATION = 3600

# Custom Exceptions
class ModelLoadingException(Exception):
    pass

class ParsingException(Exception):
    pass

class EnhancedParser:
    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        model_manager: Optional['ModelManager'] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing EnhancedParser.")
        try:
            self.config_loader = config_loader or ConfigLoader.load_config()
            self.logger.debug(f"Loaded configuration: {self.config_loader}")
            self._validate_environment()
            self.model_manager = model_manager or ModelManager(self.logger)
            self.rate_limit_semaphore = asyncio.Semaphore(RATE_LIMIT)
            self.executor = ThreadPoolExecutor(max_workers=RATE_LIMIT)
            self.logger.info("EnhancedParser initialized successfully.")
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            raise

    def _validate_environment(self):
        required_vars = ["HF_TOKEN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            self.logger.error(error_msg)
            raise EnvironmentError(error_msg)

    @lru_cache(maxsize=1000)
    async def parse(self, email_content: str) -> Dict[str, Any]:
        self.logger.info("Starting parsing process.")
        try:
            sanitized_content = sanitize_content(email_content)
            parsed_content = self._extract_email_body(sanitized_content)

            # Extract data using various methods concurrently
            extraction_results = await self._extract_all(parsed_content)

            # Merge all extracted data
            all_extracted_data = self._merge_extraction_results(extraction_results)

            # Organize data according to QUICKBASE_SCHEMA
            quickbase_data, additional_info = self._organize_data(all_extracted_data)

            # Merge quickbase_data and additional_info for validation
            merged_data = self._merge_quickbase_and_additional(quickbase_data, additional_info)

            # Post-processing
            merged_data = self._post_process(merged_data)

            # Validation
            self._validate_data(parsed_content, merged_data)

            self.logger.info("Parsing process completed successfully.")
            return merged_data
        except ParsingException as pe:
            self.logger.error(f"Parsing failed: {pe}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
            raise ParsingException(f"Unexpected error during parsing: {e}") from e

    async def _extract_all(self, email_content: str) -> Dict[str, Any]:
        tasks = [
            asyncio.create_task(self._run_stage(self._extract_with_regex, email_content)),
            asyncio.create_task(self._run_stage(self._extract_with_ner, email_content)),
            asyncio.create_task(self._run_stage(self._extract_with_sequence_model, email_content)),
            asyncio.create_task(self._run_stage(self._extract_with_qa_model, email_content)),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        extraction_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Extraction stage failed: {result}", exc_info=True)
                continue
            extraction_results.update(result)
        return extraction_results

    async def _run_stage(self, stage_func, email_content: str) -> Dict[str, Any]:
        async with self.rate_limit_semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, stage_func, email_content)

    def _extract_with_regex(self, email_content: str) -> Dict[str, Any]:
        try:
            self.logger.debug("Starting regex extraction.")
            regex_extractor = RegexExtractor(self.config_loader, self.logger)
            return regex_extractor.extract(email_content)
        except Exception as e:
            self.logger.error(f"Regex extraction error: {e}", exc_info=True)
            return {}

    def _extract_with_ner(self, email_content: str) -> Dict[str, Any]:
        try:
            self.logger.debug("Starting NER extraction.")
            ner_extractor = NERExtractor(self.model_manager, self.config_loader, self.logger)
            return ner_extractor.extract(email_content)
        except Exception as e:
            self.logger.error(f"NER extraction error: {e}", exc_info=True)
            return {}

    def _extract_with_sequence_model(self, email_content: str) -> Dict[str, Any]:
        try:
            self.logger.debug("Starting Sequence Model extraction.")
            sequence_extractor = SequenceModelExtractor(self.model_manager, self.config_loader, self.logger)
            return sequence_extractor.extract(email_content)
        except Exception as e:
            self.logger.error(f"Sequence Model extraction error: {e}", exc_info=True)
            return {}

    def _extract_with_qa_model(self, email_content: str) -> Dict[str, Any]:
        try:
            self.logger.debug("Starting QA Model extraction.")
            qa_extractor = QAExtractor(self.model_manager, self.config_loader, self.logger)
            return qa_extractor.extract(email_content)
        except Exception as e:
            self.logger.error(f"QA extraction error: {e}", exc_info=True)
            return {}

    def _merge_extraction_results(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug("Merging extraction results.")
        merged_data = {}
        for extractor, data in extraction_results.items():
            self._deep_merge(merged_data, data)
        return merged_data

    def _deep_merge(self, original: Dict[str, Any], new_data: Dict[str, Any]):
        for section, fields in new_data.items():
            if section not in original:
                original[section] = {}
            for field, value in fields.items():
                if isinstance(value, list):
                    original[section].setdefault(field, []).extend([v for v in value if v not in original[section][field]])
                else:
                    if original[section].get(field) in [None, "N/A"]:
                        original[section][field] = value
                    elif original[section].get(field) != value:
                        if isinstance(original[section].get(field), list):
                            if value not in original[section][field]:
                                original[section][field].append(value)
                        else:
                            original[section][field] = [original[section][field], value]
                self.logger.debug(f"Merged {section} -> {field}: {original[section][field]}")

    def _organize_data(self, extracted_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.logger.debug("Organizing data according to QuickBase schema.")
        quickbase_data = {}
        additional_info = {}
        for section, fields in extracted_data.items():
            if section in QUICKBASE_SCHEMA:
                for field, value in fields.items():
                    if field in QUICKBASE_SCHEMA[section]:
                        quickbase_data.setdefault(section, {})[field] = value
                    else:
                        additional_info.setdefault(section, {})[field] = value
            else:
                additional_info.setdefault(section, fields)
        return quickbase_data, additional_info

    def _merge_quickbase_and_additional(self, quickbase: Dict[str, Any], additional: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug("Merging QuickBase data with additional information.")
        merged = quickbase.copy()
        for section, fields in additional.items():
            if section in merged:
                for field, value in fields.items():
                    if field in merged[section]:
                        if isinstance(merged[section][field], list):
                            if isinstance(value, list):
                                merged[section][field].extend([v for v in value if v not in merged[section][field]])
                            else:
                                if value not in merged[section][field]:
                                    merged[section][field].append(value)
                        else:
                            if merged[section][field] != value:
                                merged[section][field] = [
                                    merged[section][field],
                                    value
                                ] if not isinstance(merged[section][field], list) else merged[section][field] + [value]
                    else:
                        merged.setdefault(section, {})[field] = value
            else:
                merged[section] = fields
        return merged

    def _post_process(self, merged_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.debug("Starting post-processing of parsed data.")
        try:
            post_processor = PostProcessor(self.config_loader, self.logger)
            merged_data = post_processor.process(merged_data)
            self._ensure_all_fields_present(merged_data)
            return merged_data
        except Exception as e:
            self.logger.error(f"Post-processing error: {e}", exc_info=True)
            return merged_data

    def _validate_data(self, parsed_content: str, merged_data: Dict[str, Any]):
        self.logger.debug("Validating parsed data.")
        try:
            validator = DataValidator(self.model_manager, self.config_loader, self.logger)
            validator.validate(parsed_content, merged_data)
        except Exception as e:
            self.logger.error(f"Validation error: {e}", exc_info=True)

    def _ensure_all_fields_present(self, merged_data: Dict[str, Any]):
        self.logger.debug("Ensuring all QuickBase schema fields are present.")
        for section, fields in QUICKBASE_SCHEMA.items():
            for field in fields:
                if field not in merged_data.get(section, {}):
                    merged_data.setdefault(section, {})[field] = "N/A"
                    self.logger.debug(f"Set default 'N/A' for missing field: {section} -> {field}")

    async def parse_email(self, email_content: str, parser_option: Any = None) -> Dict[str, Any]:
        self.logger.info("parse_email called.")
        if not self._security_checks(email_content):
            self.logger.error("Security check failed. Parsing aborted.")
            raise ValueError("Security check failed.")
        return await self.parse(email_content)

    def _extract_email_body(self, email_content: str) -> str:
        self.logger.debug("Extracting email body.")
        try:
            msg = message_from_string(email_content)
            if msg.is_multipart():
                for part in msg.get_payload():
                    content_type = part.get_content_type()
                    payload = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                    if content_type == "text/plain":
                        return payload
                    elif content_type == "text/html":
                        return BeautifulSoup(payload, "html.parser").get_text()
            else:
                content_type = msg.get_content_type()
                payload = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
                if content_type == "text/html":
                    return BeautifulSoup(payload, "html.parser").get_text()
                return payload
        except Exception as e:
            self.logger.error(f"Error extracting email body: {e}", exc_info=True)
            return ""

    def _security_checks(self, email_content: str) -> bool:
        self.logger.debug("Performing security checks.")
        try:
            if re.search(r'<script.*?>.*?</script>', email_content, re.IGNORECASE | re.DOTALL):
                self.logger.warning("Malicious script detected.")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Security checks failed: {e}", exc_info=True)
            return False

    async def shutdown(self):
        """Gracefully shutdown the parser, releasing resources."""
        self.logger.info("Shutting down EnhancedParser.")
        self.executor.shutdown(wait=True)
        for model in self.model_manager.models.values():
            if hasattr(model, 'close'):
                model.close()
        self.logger.info("EnhancedParser shut down successfully.")

# ----------------------------
# Model Manager
# ----------------------------

class ModelManager:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        self.logger.info("Initializing ModelManager.")
        # Models are loaded on-demand; initial placeholders
        self.models = {
            "ner": None,
            "sequence": None,
            "qa": None,
            "validation": None
        }

    def get_model(self, model_type: str, pipeline_type: str, model_name: str, tokenizer_name: str, **kwargs):
        if self.models.get(model_type) is None:
            self.logger.info(f"Loading {model_type} model: {model_name}")
            try:
                self.models[model_type] = pipeline(
                    pipeline_type,
                    model=model_name,
                    tokenizer=tokenizer_name,
                    **kwargs
                )
                self.logger.info(f"{model_type.capitalize()} model loaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to load {model_type} model: {e}", exc_info=True)
                raise ModelLoadingException(f"Failed to load {model_type} model: {e}") from e
        return self.models[model_type]

    def unload_model(self, model_type: str):
        """Unload a model to free up memory."""
        if model_type in self.models and self.models[model_type]:
            self.logger.info(f"Unloading {model_type} model.")
            try:
                # Assuming the pipeline has a close method or similar
                if hasattr(self.models[model_type], 'close'):
                    self.models[model_type].close()
                del self.models[model_type]
                self.models[model_type] = None
                self.logger.info(f"{model_type.capitalize()} model unloaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to unload {model_type} model: {e}", exc_info=True)

# ----------------------------
# Extractors
# ----------------------------

class RegexExtractor:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        self.logger.debug("Compiling regex patterns.")
        field_patterns = {
            # Requesting Party
            "Insurance Company": r"Insurance Company:\s*(.+)",
            "Handler": r"Handler:\s*(.+)",
            "Carrier Claim Number": r"Carrier Claim Number:\s*(\S+)",

            # Insured Information
            "Name": r"Name:\s*(.+)",
            "Contact #": r"Contact #:\s*([\d\-()+\s]+)",
            "Loss Address": r"Loss Address:\s*(.+)",
            "Public Adjuster": r"Public Adjuster:\s*(.+)",
            "Owner or Tenant": r"Owner or Tenant:\s*(.+)",

            # Adjuster Information
            "Adjuster Name": r"Adjuster Name:\s*(.+)",
            "Adjuster Phone Number": r"Adjuster Phone Number:\s*([\d\-()+\s]+)",
            "Adjuster Email": r"Adjuster Email:\s*([^\s]+)",
            "Job Title": r"Job Title:\s*(.+)",
            "Address": r"Address:\s*(.+)",
            "Policy #": r"Policy #:\s*(\S+)",

            # Assignment Information
            "Date of Loss/Occurrence": r"Date of Loss/Occurrence:\s*(.+)",
            "Cause of loss": r"Cause of loss:\s*(.+)",
            "Facts of Loss": r"Facts of Loss:\s*(.+)",
            "Loss Description": r"Loss Description:\s*(.+)",
            "Residence Occupied During Loss": r"Residence Occupied During Loss:\s*(.+)",
            "Was Someone Home at Time of Damage": r"Was Someone Home at Time of Damage:\s*(.+)",
            "Repair or Mitigation Progress": r"Repair or Mitigation Progress:\s*(.+)",
            "Type": r"Type:\s*(.+)",
            "Inspection type": r"Inspection type:\s*(.+)",
            "Assignment Type": r"Assignment Type:\s*(.+)",
            "Additional details/Special Instructions": r"Additional details/Special Instructions:\s*(.+)",
            "Attachment(s)": r"Attachment\(s\):\s*(.+)",
        }
        compiled = {field: re.compile(pattern, re.DOTALL | re.IGNORECASE) for field, pattern in field_patterns.items()}
        return compiled

    def extract(self, email_content: str) -> Dict[str, Any]:
        extracted_data = {}
        field_section_mapping = self._get_field_section_mapping()
        for field, pattern in self.patterns.items():
            matches = pattern.findall(email_content)
            if matches:
                value = matches[0].strip()
                section = field_section_mapping.get(field)
                if section:
                    if isinstance(value, str):
                        value = value.replace('\n', ' ').strip()
                    extracted_data.setdefault(section, {}).setdefault(field, value)
                    self.logger.debug(f"Regex extracted {section} -> {field}: {value}")
        return extracted_data

    def _get_field_section_mapping(self) -> Dict[str, str]:
        return {
            "Insurance Company": "Requesting Party",
            "Handler": "Requesting Party",
            "Carrier Claim Number": "Requesting Party",

            "Name": "Insured Information",
            "Contact #": "Insured Information",
            "Loss Address": "Insured Information",
            "Public Adjuster": "Insured Information",
            "Owner or Tenant": "Insured Information",

            "Adjuster Name": "Adjuster Information",
            "Adjuster Phone Number": "Adjuster Information",
            "Adjuster Email": "Adjuster Information",
            "Job Title": "Adjuster Information",
            "Address": "Adjuster Information",
            "Policy #": "Adjuster Information",

            "Date of Loss/Occurrence": "Assignment Information",
            "Cause of loss": "Assignment Information",
            "Facts of Loss": "Assignment Information",
            "Loss Description": "Assignment Information",
            "Residence Occupied During Loss": "Assignment Information",
            "Was Someone Home at Time of Damage": "Assignment Information",
            "Repair or Mitigation Progress": "Assignment Information",
            "Type": "Assignment Information",
            "Inspection type": "Assignment Information",
            "Assignment Type": "Assignment Information",
            "Additional details/Special Instructions": "Assignment Information",
            "Attachment(s)": "Assignment Information",
        }

class NERExtractor:
    def __init__(self, model_manager: ModelManager, config: dict, logger: logging.Logger):
        self.model_manager = model_manager
        self.config = config
        self.logger = logger
        self.ner_pipeline = self.model_manager.get_model(
            model_type="ner",
            pipeline_type="ner",
            model_name="dslim/bert-base-NER",
            tokenizer_name="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

    def extract(self, email_content: str) -> Dict[str, Any]:
        try:
            self.logger.debug("Running NER pipeline.")
            entities = self.ner_pipeline(email_content)
            extracted_entities = self._map_entities(entities)
            return extracted_entities
        except Exception as e:
            self.logger.error(f"NER extraction failed: {e}", exc_info=True)
            return {}

    def _map_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        label_field_mapping = {
            "PER": ("Insured Information", "Name"),
            "ORG": ("Requesting Party", "Insurance Company"),
            "LOC": ("Insured Information", "Loss Address"),
            "MISC": ("Assignment Information", "Cause of loss"),
            "DATE": ("Assignment Information", "Date of Loss/Occurrence"),
        }
        mapped = {}
        for entity in entities:
            label = entity.get("entity_group")
            text = entity.get("word")
            if label and text:
                mapping = label_field_mapping.get(label)
                if mapping:
                    section, field = mapping
                    if field in QUICKBASE_SCHEMA.get(section, {}):
                        mapped.setdefault(section, {}).setdefault(field, []).append(text.strip())
                        self.logger.debug(f"NER extracted {section} -> {field}: {text.strip()}")
        return mapped

class SequenceModelExtractor:
    def __init__(self, model_manager: ModelManager, config: dict, logger: logging.Logger):
        self.model_manager = model_manager
        self.config = config
        self.logger = logger
        self.sequence_pipeline = self.model_manager.get_model(
            model_type="sequence",
            pipeline_type="summarization",
            model_name="facebook/bart-large",
            tokenizer_name="facebook/bart-large"
        )

    def extract(self, email_content: str) -> Dict[str, Any]:
        try:
            self.logger.debug("Generating summary with Sequence Model.")
            summary = self.sequence_pipeline(email_content, max_length=150, min_length=40, do_sample=False)
            summary_text = summary[0].get("summary_text", "")
            self.logger.debug(f"Sequence Model summary: {summary_text}")
            return self._parse_summary(summary_text)
        except Exception as e:
            self.logger.error(f"Sequence Model extraction failed: {e}", exc_info=True)
            return {}

    def _parse_summary(self, summary: str) -> Dict[str, Any]:
        pattern = r"(?P<field>[\w\s]+):\s*(?P<value>[^,]+)"
        matches = re.finditer(pattern, summary)
        extracted = {}
        for match in matches:
            field = match.group("field").strip()
            value = match.group("value").strip()
            for section, fields in QUICKBASE_SCHEMA.items():
                for schema_field in fields:
                    if field.lower() == schema_field.lower():
                        extracted.setdefault(section, {})[schema_field] = value
                        self.logger.debug(f"Sequence Model extracted {section} -> {schema_field}: {value}")
        return extracted

class QAExtractor:
    def __init__(self, model_manager: ModelManager, config: dict, logger: logging.Logger):
        self.model_manager = model_manager
        self.config = config
        self.logger = logger
        self.qa_pipeline = self.model_manager.get_model(
            model_type="qa",
            pipeline_type="question-answering",
            model_name="bert-large-uncased-whole-word-masking-finetuned-squad",
            tokenizer_name="bert-large-uncased-whole-word-masking-finetuned-squad"
        )

    def extract(self, email_content: str) -> Dict[str, Any]:
        try:
            self.logger.debug("Running QA pipeline.")
            questions = self._generate_questions()
            extracted_qa = {}
            for section, fields in QUICKBASE_SCHEMA.items():
                for field in fields:
                    question = f"What is the {field}?"
                    result = self.qa_pipeline(question=question, context=email_content)
                    answer = result.get("answer", "").strip()
                    if answer and answer.lower() not in ["n/a", ""]:
                        extracted_qa.setdefault(section, {})[field] = answer
                        self.logger.debug(f"QA extracted {section} -> {field}: {answer}")
            return extracted_qa
        except Exception as e:
            self.logger.error(f"QA extraction failed: {e}", exc_info=True)
            return {}

    def _generate_questions(self) -> List[str]:
        questions = []
        for section, fields in QUICKBASE_SCHEMA.items():
            for field in fields:
                questions.append(f"What is the {field}?")
        return questions

# ----------------------------
# Post-Processing
# ----------------------------

class PostProcessor:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for section, fields in data.items():
            if section not in QUICKBASE_SCHEMA:
                continue
            for field, value in fields.items():
                if field not in QUICKBASE_SCHEMA[section]:
                    continue
                if isinstance(value, list):
                    value = value[0] if value else "N/A"
                if "Date" in field:
                    value = self._format_date(value)
                if "Phone Number" in field or "Contact #" in field:
                    value = self._format_phone_number(value)
                value = self._clean_data(value)
                data[section][field] = value
        # Verify attachments
        attachments = data.get("Assignment Information", {}).get("Attachment(s)", "N/A")
        if attachments != "N/A" and not self._verify_attachments([attachments]):
            data["user_notifications"] = "Attachments mentioned but not found in the email."
            self.logger.warning("Attachments mentioned but not found in the email.")
        return data

    def _clean_data(self, value: str) -> str:
        value = re.sub(r'[^\w\s\-@.,]', '', value)
        return value.strip() if value else "N/A"

    def _format_date(self, date_str: str) -> str:
        if date_str == "N/A":
            return date_str
        try:
            date_obj = dateutil_parser.parse(date_str, fuzzy=True)
            formatted_date = date_obj.strftime("%Y-%m-%d")
            self.logger.debug(f"Formatted date: {formatted_date}")
            return formatted_date
        except (ValueError, TypeError):
            self.logger.warning(f"Failed to parse date: {date_str}")
            return "N/A"

    def _format_phone_number(self, phone: str) -> str:
        if phone == "N/A":
            return phone
        try:
            parsed_number = phonenumbers.parse(phone, "US")
            if phonenumbers.is_valid_number(parsed_number):
                formatted_phone = phonenumbers.format_number(parsed_number, PhoneNumberFormat.E164)
                self.logger.debug(f"Formatted phone number: {formatted_phone}")
                return formatted_phone
            self.logger.warning(f"Invalid phone number: {phone}")
            return "N/A"
        except phonenumbers.NumberParseException:
            self.logger.warning(f"Failed to parse phone number: {phone}")
            return "N/A"

    def _verify_attachments(self, attachments: List[str]) -> bool:
        self.logger.debug(f"Verifying attachments: {attachments}")
        valid_extensions = self.config.get("valid_extensions", [])
        for attachment in attachments:
            if not any(attachment.lower().endswith(ext) for ext in valid_extensions):
                self.logger.warning(f"Invalid attachment extension detected: {attachment}")
                return False
        return True

# ----------------------------
# Data Validator
# ----------------------------

class DataValidator:
    def __init__(self, model_manager: ModelManager, config: dict, logger: logging.Logger):
        self.model_manager = model_manager
        self.config = config
        self.logger = logger
        self.validation_pipeline = self.model_manager.get_model(
            model_type="validation",
            pipeline_type="text-generation",
            model_name="gpt2",
            tokenizer_name="gpt2"
        )

    def validate(self, email_content: str, merged_data: Dict[str, Any]):
        try:
            self.logger.debug("Running Validation pipeline.")
            self._validate_fields(merged_data)
            self._validate_schema(merged_data)
            self._validate_json(merged_data)
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}", exc_info=True)

    def _validate_fields(self, merged_data: Dict[str, Any]):
        self.logger.debug("Validating individual fields.")
        for section, fields in QUICKBASE_SCHEMA.items():
            for field in fields:
                value = merged_data.get(section, {}).get(field)
                if not value or value == "N/A":
                    merged_data.setdefault("missing_fields", []).append(f"{section} -> {field}")
                    self.logger.warning(f"Missing field: {section} -> {field}")
                    continue
                known_values = self.config.get("known_values", {}).get(field, [])
                if known_values:
                    best_match = max(
                        known_values,
                        key=lambda x: fuzz.partial_ratio(x.lower(), value.lower()),
                        default=None
                    )
                    if best_match and fuzz.partial_ratio(best_match.lower(), value.lower()) >= self.config.get("fuzzy_threshold", 90):
                        merged_data[section][field] = best_match
                        self.logger.debug(f"Field {section} -> {field} matched to known value: {best_match}")
                    else:
                        merged_data.setdefault("inconsistent_fields", []).append(f"{section} -> {field}")
                        self.logger.warning(f"Inconsistent field: {section} -> {field} with value: {value}")

    def _validate_schema(self, merged_data: Dict[str, Any]):
        self.logger.debug("Validating schema alignment.")
        try:
            validation_text = " ".join([
                str(value) for section in merged_data.values() if isinstance(section, dict)
                for value in section.values() if isinstance(value, str)
            ])
            validation_result = self.validation_pipeline(validation_text, max_new_tokens=50)
            merged_data["validation"] = validation_result[0].get("generated_text", "")
            self.logger.debug(f"Validation result: {merged_data['validation']}")
        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}", exc_info=True)

    def _validate_json(self, merged_data: Dict[str, Any]):
        self.logger.debug("Performing JSON validation.")
        try:
            is_valid, error_message = validate_json(merged_data)
            if not is_valid:
                merged_data.setdefault("validation_issues", []).append(error_message)
                self.logger.error(f"JSON validation issues: {error_message}")
            else:
                self.logger.debug("JSON validation successful.")
        except Exception as e:
            self.logger.error(f"JSON validation failed: {e}", exc_info=True)
            merged_data.setdefault("validation_issues", []).append(str(e))
