import logging
from typing import Optional, Dict, Any, List
import signal
import os
from transformers import pipeline
from huggingface_hub import login
from thefuzz import fuzz
from dateutil import parser as dateutil_parser
import phonenumbers
from phonenumbers import PhoneNumberFormat

from src.parsers.base_parser import BaseParser
from src.utils.validation import validate_json
from src.utils.config_loader import ConfigLoader
from src.utils.quickbase_schema import QUICKBASE_SCHEMA

LLM_TIMEOUT_SECONDS = 5

class TimeoutException(Exception):
    pass

def llm_timeout_handler(signum, frame):
    raise TimeoutException("LLM processing timed out.")

class EnhancedParser(BaseParser):
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = ConfigLoader.load_config()
        self.init_ner()
        self.init_layout_aware()
        self.init_sequence_model()
        self.init_validation_model()
        self.logger.info("EnhancedParser initialized successfully.")

    def init_ner(self):
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                tokenizer="dslim/bert-base-NER",
                aggregation_strategy="simple",
            )
            self.logger.info("Loaded NER model 'dslim/bert-base-NER' successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load NER model 'dslim/bert-base-NER': {e}")
            raise RuntimeError(f"Failed to load NER model 'dslim/bert-base-NER': {e}") from e

    def init_layout_aware(self):
        try:
            self.layoutlm_pipeline = pipeline(
                "token-classification",
                model="microsoft/layoutlmv3-base",
                tokenizer="microsoft/layoutlmv3-base",
            )
            self.logger.info("Loaded Layout-Aware model 'microsoft/layoutlmv3-base' successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Layout-Aware model 'microsoft/layoutlmv3-base': {e}")
            raise RuntimeError(f"Failed to load Layout-Aware model 'microsoft/layoutlmv3-base': {e}") from e

    def init_sequence_model(self):
        try:
            self.sequence_model_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large",
                tokenizer="facebook/bart-large",
            )
            self.logger.info("Loaded Sequence Model 'facebook/bart-large' successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Sequence Model 'facebook/bart-large': {e}")
            raise RuntimeError(f"Failed to load Sequence Model 'facebook/bart-large': {e}") from e

    def init_validation_model(self):
        try:
            # Retrieve the token from environment variables
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("Hugging Face token not found in environment variables.")
            
            # Login to Hugging Face Hub
            login(token=hf_token)
            self.logger.info("Logged in to Hugging Face Hub successfully.")

            self.validation_pipeline = pipeline(
                "text-generation",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
            )
            self.logger.info("Loaded Validation Model 'meta-llama/Meta-Llama-3-8B-Instruct' successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Validation Model 'meta-llama/Meta-Llama-3-8B-Instruct': {e}")
            raise RuntimeError(f"Failed to load Validation Model 'meta-llama/Meta-Llama-3-8B-Instruct': {e}") from e

    def parse(self, email_content: str) -> Dict[str, Any]:
        self.logger.info("Starting parsing process.")
        parsed_data: Dict[str, Any] = {}

        try:
            parsed_data.update(self._stage_ner_parsing(email_content))
            parsed_data.update(self._stage_layout_aware_parsing(email_content))
            parsed_data.update(self._stage_sequence_model_parsing(email_content))
            parsed_data.update(self._stage_validation(email_content, parsed_data))
            self._stage_schema_validation(parsed_data)
            parsed_data = self._stage_post_processing(parsed_data)
            self._stage_json_validation(parsed_data)
            self.logger.info("Parsing process completed successfully.")
            return parsed_data

        except TimeoutException as te:
            self.logger.error(f"Validation Model parsing timeout: {te}")
            raise RuntimeError(f"Validation Model parsing timeout: {te}") from te
        except ValueError as ve:
            self.logger.error(f"ValueError during parsing: {ve}")
            raise ve
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during parsing: {e}") from e

    def _stage_ner_parsing(self, email_content: str) -> Dict[str, Any]:
        ner_data = self.ner_parsing(email_content)
        return ner_data

    def _stage_layout_aware_parsing(self, email_content: str) -> Dict[str, Any]:
        layout_data = self.layout_aware_parsing(email_content)
        return layout_data

    def _stage_sequence_model_parsing(self, email_content: str) -> Dict[str, Any]:
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, llm_timeout_handler)
            signal.alarm(LLM_TIMEOUT_SECONDS)

        try:
            summary = self.sequence_model_pipeline(email_content, max_length=150, min_length=40, do_sample=False)
            summary_text = summary[0]['summary_text']
            extracted_sequence = self.sequence_model_extract(summary_text)
            if hasattr(signal, "alarm"):
                signal.alarm(0)
            return extracted_sequence
        except TimeoutException:
            self.logger.warning("Sequence Model parsing timed out.")
            return {}
        except Exception as e:
            self.logger.error(f"Error during Sequence Model parsing: {e}")
            return {}
        finally:
            if hasattr(signal, "alarm"):
                signal.alarm(0)

    def _stage_validation(self, email_content: str, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        validation_data = self.validation_parsing(email_content, parsed_data)
        return validation_data

    def ner_parsing(self, email_content: str) -> Dict[str, Any]:
        try:
            entities = self.ner_pipeline(email_content)
            extracted_entities: Dict[str, Any] = {}
            for entity in entities:
                label = entity.get("entity_group")
                text = entity.get("word")
                if label and text:
                    if label.lower() in ["name", "organization"]:
                        extracted_entities.setdefault("Insured Information", {}).setdefault("Name", []).append(text)
                    elif label.lower() == "date":
                        extracted_entities.setdefault("Assignment Information", {}).setdefault("Date of Loss/Occurrence", []).append(text)
                    elif label.lower() in ["claim_number", "policy_number"]:
                        field = "Carrier Claim Number" if label.lower() == "claim_number" else "Policy #"
                        extracted_entities.setdefault("Requesting Party", {}).setdefault(field, []).append(text)
                    elif label.lower() == "email":
                        extracted_entities.setdefault("Adjuster Information", {}).setdefault("Adjuster Email", []).append(text)
            return extracted_entities
        except Exception as e:
            self.logger.error(f"Error during NER parsing: {e}")
            return {}

    def layout_aware_parsing(self, email_content: str) -> Dict[str, Any]:
        try:
            entities = self.layoutlm_pipeline(email_content)
            extracted_layout: Dict[str, Any] = {}
            for entity in entities:
                label = entity.get("entity_group")
                text = entity.get("word")
                if label and text:
                    if label.lower() in ["address", "location"]:
                        extracted_layout.setdefault("Insured Information", {}).setdefault("Loss Address", []).append(text)
                    elif label.lower() in ["damage", "cost"]:
                        extracted_layout.setdefault("Assignment Information", {}).setdefault("Loss Description", []).append(text)
            return extracted_layout
        except Exception as e:
            self.logger.error(f"Error during Layout-Aware parsing: {e}")
            return {}

    def sequence_model_extract(self, summary_text: str) -> Dict[str, Any]:
        extracted_sequence: Dict[str, Any] = {}
        try:
            for item in summary_text.split(','):
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    for section, fields in QUICKBASE_SCHEMA.items():
                        if key in fields:
                            extracted_sequence.setdefault(section, {}).setdefault(key, []).append(value)
            return extracted_sequence
        except Exception as e:
            self.logger.error(f"Error during Sequence Model extraction: {e}")
            return {}

    def validation_parsing(self, email_content: str, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = (
                f"Validate the following extracted data against the original email content. "
                f"Ensure all fields are consistent and complete.\n\n"
                f"Email Content:\n{email_content}\n\n"
                f"Extracted Data:\n{parsed_data}\n\n"
                f"Provide a list of any missing or inconsistent fields."
            )
            validation_response = self.validation_pipeline(prompt, max_length=500, do_sample=False)
            validation_text = validation_response[0]['generated_text']
            issues = self.parse_validation_response(validation_text)
            if issues:
                parsed_data["validation_issues"] = issues
            return {}
        except TimeoutException:
            self.logger.warning("Validation Model parsing timed out.")
            return {}
        except Exception as e:
            self.logger.error(f"Error during Validation Model parsing: {e}")
            return {}

    def parse_validation_response(self, validation_text: str) -> List[str]:
        issues = []
        lines = validation_text.strip().split('\n')
        for line in lines:
            if line.strip():
                issues.append(line.strip())
        return issues

    def _stage_schema_validation(self, parsed_data: Dict[str, Any]):
        missing_fields: List[str] = []
        inconsistent_fields: List[str] = []

        for section, fields in QUICKBASE_SCHEMA.items():
            for field, _ in fields.items():
                value = parsed_data.get(section, {}).get(field)
                if not value or value == "N/A":
                    missing_fields.append(f"{section} -> {field}")
                    continue

                known_values = self.config.get("known_values", {}).get(field, [])
                if known_values:
                    best_match = max(
                        known_values,
                        key=lambda x: fuzz.partial_ratio(x.lower(), value.lower()),
                        default=None,
                    )
                    if best_match and fuzz.partial_ratio(best_match.lower(), value.lower()) >= self.config.get("fuzzy_threshold", 90):
                        parsed_data[section][field] = best_match
                    else:
                        inconsistent_fields.append(f"{section} -> {field}")
                else:
                    continue

        if missing_fields:
            parsed_data["missing_fields"] = missing_fields

        if inconsistent_fields:
            parsed_data["inconsistent_fields"] = inconsistent_fields

    def post_processing(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        skip_sections = [
            "TransformerEntities",
            "Entities",
            "missing_fields",
            "inconsistent_fields",
            "user_notifications",
            "validation_issues",
        ]

        for section, fields in parsed_data.items():
            if section in skip_sections:
                continue
            if not isinstance(fields, dict):
                continue

            for field, value in fields.items():
                if "Date" in field:
                    parsed_data[section][field] = self.format_date(value)
                if "Phone Number" in field or "Contact #" in field:
                    parsed_data[section][field] = self.format_phone_number(value)

        transformer_entities = parsed_data.get("TransformerEntities", {})
        for _, entities in transformer_entities.items():
            for entity in entities:
                text = entity.get("text", "").strip()
                confidence = entity.get("confidence", 0.0)
                if "policy" in text.lower():
                    parsed_data.setdefault("Adjuster Information", {}).setdefault("Policy #", []).append(text)

        attachments = parsed_data.get("Assignment Information", {}).get("Attachment(s)", [])
        if attachments and not self.verify_attachments(attachments):
            parsed_data["user_notifications"] = "Attachments mentioned but not found in the email."

        return parsed_data

    def format_date(self, date_str: str) -> str:
        if date_str == "N/A":
            return date_str

        for fmt in self.config.get("date_formats", []):
            try:
                date_obj = dateutil_parser.parse(date_str, fuzzy=True)
                standardized_date = date_obj.strftime("%Y-%m-%d")
                return standardized_date
            except (ValueError, TypeError):
                continue
        return "N/A"

    def format_phone_number(self, phone: str) -> str:
        if phone == "N/A":
            return phone

        try:
            parsed_number = phonenumbers.parse(phone, "US")
            if phonenumbers.is_valid_number(parsed_number):
                formatted_number = phonenumbers.format_number(
                    parsed_number, PhoneNumberFormat.E164
                )
                return formatted_number
            else:
                return "N/A"
        except phonenumbers.NumberParseException:
            return "N/A"

    def verify_attachments(self, attachments: List[str]) -> bool:
        return True
