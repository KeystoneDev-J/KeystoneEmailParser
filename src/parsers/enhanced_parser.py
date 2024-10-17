import logging
from typing import Optional, Dict, Any, List
import os
import re
from transformers import pipeline
from huggingface_hub import login
from thefuzz import fuzz
from dateutil import parser as dateutil_parser
import phonenumbers
from phonenumbers import PhoneNumberFormat
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from src.utils.validation import validate_json
from src.utils.config_loader import ConfigLoader
from src.utils.quickbase_schema import QUICKBASE_SCHEMA

# Define a timeout constant for long-running LLM processes
LLM_TIMEOUT_SECONDS = 500


# Custom exception for handling timeouts
class TimeoutException(Exception):
    pass


class EnhancedParser:
    """
    EnhancedParser class extends the BaseParser to implement advanced parsing techniques
    using various NLP models. It orchestrates the parsing stages and handles exceptions,
    logging, and validations throughout the process.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Initialize the logger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing EnhancedParser.")

        try:
            # Load configuration settings
            self.config = ConfigLoader.load_config()
            self.logger.debug(f"Loaded configuration: {self.config}")

            # Check for required environment variables
            self._check_environment_variables()

            # Initialize model attributes to None for lazy loading
            self.ner_pipeline = None
            self.layoutlm_pipeline = None
            self.sequence_model_pipeline = None
            self.validation_pipeline = None

            self.logger.info("EnhancedParser initialized successfully.")
        except Exception as e:
            self.logger.error(
                f"Error during EnhancedParser initialization: {e}", exc_info=True
            )
            # Re-raise the exception to prevent initialization in a degraded state
            raise

    def _check_environment_variables(self):
        required_vars = ["HF_TOKEN"]  # Add other required variables here
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            self.logger.error(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def _lazy_load_ner(self):
        if self.ner_pipeline is None:
            self.init_ner()

    def _lazy_load_layout_aware(self):
        if self.layoutlm_pipeline is None:
            self.init_layout_aware()

    def _lazy_load_sequence_model(self):
        if self.sequence_model_pipeline is None:
            self.init_sequence_model()

    def _lazy_load_validation_model(self):
        if self.validation_pipeline is None:
            self.init_validation_model()

    def init_ner(self):
        """
        Initialize the Named Entity Recognition (NER) pipeline using a pre-trained model.
        """
        try:
            self.logger.info("Initializing NER pipeline.")
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                tokenizer="dslim/bert-base-NER",
                aggregation_strategy="simple",
            )
            self.logger.info("Loaded NER model 'dslim/bert-base-NER' successfully.")
        except Exception as e:
            self.logger.error(
                f"Failed to load NER model 'dslim/bert-base-NER': {e}", exc_info=True
            )
            raise RuntimeError(
                f"Failed to load NER model 'dslim/bert-base-NER': {e}"
            ) from e

    def init_layout_aware(self):
        """
        Initialize the Layout-Aware pipeline using a pre-trained model.
        """
        try:
            self.logger.info("Initializing Layout-Aware pipeline.")
            self.layoutlm_pipeline = pipeline(
                "token-classification",
                model="microsoft/layoutlmv3-base",
                tokenizer="microsoft/layoutlmv3-base",
            )
            self.logger.info(
                "Loaded Layout-Aware model 'microsoft/layoutlmv3-base' successfully."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load Layout-Aware model 'microsoft/layoutlmv3-base': {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load Layout-Aware model 'microsoft/layoutlmv3-base': {e}"
            ) from e

    def init_sequence_model(self):
        """
        Initialize the Sequence Model pipeline using a pre-trained model for summarization.
        """
        try:
            self.logger.info("Initializing Sequence Model pipeline.")
            self.sequence_model_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large",
                tokenizer="facebook/bart-large",
            )
            self.logger.info(
                "Loaded Sequence Model 'facebook/bart-large' successfully."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load Sequence Model 'facebook/bart-large': {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load Sequence Model 'facebook/bart-large': {e}"
            ) from e

    def init_validation_model(self):
        """
        Initialize the Validation Model pipeline using a pre-trained model for text generation.
        """
        try:
            self.logger.info("Initializing Validation Model pipeline.")

            # Retrieve the Hugging Face token from environment variables
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError(
                    "Hugging Face token not found in environment variables."
                )

            # Login to Hugging Face Hub using the token
            login(token=hf_token)
            self.logger.info("Logged in to Hugging Face Hub successfully.")

            # Initialize the validation pipeline
            # Using a smaller model for testing or replace with your desired model
            self.validation_pipeline = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
            )
            self.logger.info("Loaded Validation Model 'gpt2' successfully.")
        except MemoryError as me:
            self.logger.critical(
                "MemoryError: Not enough memory to load the validation model.",
                exc_info=True,
            )
            raise RuntimeError(
                "MemoryError: Not enough memory to load the validation model."
            ) from me
        except Exception as e:
            self.logger.error(
                f"Failed to load Validation Model: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to load Validation Model: {e}") from e


    @lru_cache(maxsize=100)
    def parse(self, email_content: str) -> Dict[str, Any]:
        """
        Orchestrates the parsing process by executing each parsing stage sequentially.

        Args:
            email_content (str): The raw email content to be parsed.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed data.
        """
        self.logger.info("Starting parsing process.")
        parsed_data: Dict[str, Any] = {}

        try:
            # Stage 1: Regex Extraction
            self.logger.info("Stage 1: Regex Extraction.")
            regex_data = self.regex_extraction(email_content)
            parsed_data.update(regex_data)

            # Stage 2: NER Parsing
            self.logger.info("Stage 2: NER Parsing.")
            ner_data = self._stage_ner_parsing(email_content)
            parsed_data = self.merge_parsed_data(parsed_data, ner_data)

            # Stage 3: Layout-Aware Parsing
            self.logger.info("Stage 3: Layout-Aware Parsing.")
            layout_data = self._stage_layout_aware_parsing(email_content)
            parsed_data = self.merge_parsed_data(parsed_data, layout_data)

            # Stage 4: Sequence Model Parsing
            self.logger.info("Stage 4: Sequence Model Parsing.")
            sequence_data = self._stage_sequence_model_parsing(email_content)
            parsed_data = self.merge_parsed_data(parsed_data, sequence_data)

            # Stage 5: Validation Parsing
            self.logger.info("Stage 5: Validation Parsing.")
            self._stage_validation(email_content, parsed_data)

            # Stage 6: Schema Validation
            self.logger.info("Stage 6: Schema Validation.")
            self._stage_schema_validation(parsed_data)

            # Stage 7: Post Processing
            self.logger.info("Stage 7: Post Processing.")
            parsed_data = self._stage_post_processing(parsed_data)

            # Stage 8: JSON Validation
            self.logger.info("Stage 8: JSON Validation.")
            self._stage_json_validation(parsed_data)

            self.logger.info("Parsing process completed successfully.")
            return parsed_data

        except TimeoutException as te:
            self.logger.error(f"Validation Model parsing timeout: {te}", exc_info=True)
            raise RuntimeError(f"Validation Model parsing timeout: {te}") from te
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during parsing: {e}") from e

    def merge_parsed_data(
        self, original_data: Dict[str, Any], new_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merges new parsed data into the original data, combining lists and avoiding duplicates.

        Args:
            original_data (Dict[str, Any]): The original parsed data.
            new_data (Dict[str, Any]): The new parsed data to merge.

        Returns:
            Dict[str, Any]: The merged parsed data.
        """
        for section, fields in new_data.items():
            if section not in original_data:
                original_data[section] = fields
            else:
                for field, value in fields.items():
                    if field not in original_data[section]:
                        original_data[section][field] = value
                    else:
                        if isinstance(
                            original_data[section][field], list
                        ) and isinstance(value, list):
                            combined_list = original_data[section][field] + value
                            # Remove duplicates while preserving order
                            seen = set()
                            original_data[section][field] = [
                                x
                                for x in combined_list
                                if not (x in seen or seen.add(x))
                            ]
                        else:
                            # Overwrite with the new value
                            original_data[section][field] = value
        return original_data

    def _stage_ner_parsing(self, email_content: str) -> Dict[str, Any]:
        """
        Executes the NER parsing stage.

        Args:
            email_content (str): The email content to parse.

        Returns:
            Dict[str, Any]: Parsed data from NER.
        """
        self.logger.debug("Executing NER parsing stage.")
        self._lazy_load_ner()
        ner_data = self.ner_parsing(email_content)
        return ner_data

    def _stage_layout_aware_parsing(self, email_content: str) -> Dict[str, Any]:
        """
        Executes the Layout-Aware parsing stage.

        Args:
            email_content (str): The email content to parse.

        Returns:
            Dict[str, Any]: Parsed data from Layout-Aware parsing.
        """
        self.logger.debug("Executing Layout-Aware parsing stage.")
        self._lazy_load_layout_aware()
        layout_data = self.layout_aware_parsing(email_content)
        return layout_data

    def _stage_sequence_model_parsing(self, email_content: str) -> Dict[str, Any]:
        """
        Executes the Sequence Model parsing stage with timeout handling.

        Args:
            email_content (str): The email content to parse.

        Returns:
            Dict[str, Any]: Parsed data from Sequence Model.
        """
        self.logger.debug("Executing Sequence Model parsing stage.")
        self._lazy_load_sequence_model()
        try:
            # Tokenize and truncate the input text
            tokenizer = self.sequence_model_pipeline.tokenizer
            inputs = tokenizer.encode(
                email_content, max_length=1024, truncation=True, return_tensors="pt"
            )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.sequence_model_pipeline,
                    tokenizer.decode(inputs[0], skip_special_tokens=True),
                    max_length=150,
                    min_length=40,
                    do_sample=False,
                )
                # Wait for the result with a timeout
                summary = future.result(timeout=LLM_TIMEOUT_SECONDS)
                summary_text = summary[0]["summary_text"]
                self.logger.debug(f"Sequence Model Summary: {summary_text}")

                # Extract data from the summary
                extracted_sequence = self.sequence_model_extract(summary_text)
                return extracted_sequence
        except TimeoutError:
            self.logger.warning("Sequence Model parsing timed out.")
            return {}
        except Exception as e:
            self.logger.error(
                f"Error during Sequence Model parsing: {e}", exc_info=True
            )
            return {}

    def _stage_validation(self, email_content: str, parsed_data: Dict[str, Any]):
        """
        Executes the Validation parsing stage.

        Args:
            email_content (str): The original email content.
            parsed_data (Dict[str, Any]): The data parsed so far.
        """
        self.logger.debug("Executing Validation parsing stage.")
        self._lazy_load_validation_model()
        self.validation_parsing(email_content, parsed_data)

    def ner_parsing(self, email_content: str) -> Dict[str, Any]:
        """
        Performs Named Entity Recognition on the email content.

        Args:
            email_content (str): The email content to parse.

        Returns:
            Dict[str, Any]: Extracted entities.
        """
        try:
            self.logger.debug("Starting NER pipeline.")
            entities = self.ner_pipeline(email_content)
            extracted_entities: Dict[str, Any] = {}

            # Map NER labels to schema fields
            label_field_mapping = {
                "PER": ("Insured Information", "Name"),
                "ORG": ("Requesting Party", "Insurance Company"),
                "LOC": ("Insured Information", "Loss Address"),
                "MISC": ("Assignment Information", "Cause of loss"),
                "DATE": ("Assignment Information", "Date of Loss/Occurrence"),
                # Add custom labels if using a fine-tuned model
                # 'POLICY_NUMBER': ('Adjuster Information', 'Policy #'),
                # 'CLAIM_NUMBER': ('Requesting Party', 'Carrier Claim Number'),
                # Add more mappings as needed
            }

            for entity in entities:
                label = entity.get("entity_group")
                text = entity.get("word")

                if label and text:
                    mapping = label_field_mapping.get(label)
                    if mapping:
                        section, field = mapping
                        extracted_entities.setdefault(section, {}).setdefault(
                            field, []
                        ).append(text.strip())

            self.logger.debug(f"NER Parsing Result: {extracted_entities}")
            return extracted_entities
        except Exception as e:
            self.logger.error(f"Error during NER parsing: {e}", exc_info=True)
            return {}

    def regex_extraction(self, email_content: str) -> Dict[str, Any]:
        """
        Performs regex-based extraction on the email content.

        Args:
            email_content (str): The email content to parse.

        Returns:
            Dict[str, Any]: Extracted data using regex patterns.
        """
        extracted_data = {}
        try:
            self.logger.debug("Starting regex extraction.")

            # Patterns for extraction
            patterns = {
                "Policy #": r"Policy (?:Number|#):\s*(\S+)",
                "Carrier Claim Number": r"Claim (?:Number|#):\s*(\S+)",
                "Date of Loss/Occurrence": r"Date of Loss:\s*([^\n]+)",
                "Adjuster Name": r"Your adjuster, (.+?) \(",
                "Adjuster Email": r"Your adjuster, .+? \(([^)]+)\)",
                "Adjuster Phone Number": r"Phone:\s*([\d-]+)",
                "Public Adjuster": r"Best regards,\s*(.+?)\n",
                "Public Adjuster Phone": r"Phone:\s*([\d-]+)",
                "Public Adjuster Email": r"Email:\s*([^\s]+)",
                "Name": r"Policyholder:\s*([^\n]+)",
                "Loss Address": r"Property Address:\s*([^\n]+)",
                "Cause of loss": r"Peril:\s*([^\n]+)",
                "Loss Description": r"Claim Details:\s*\n(.*?)\n\n",
                "Attachment(s)": r"Please find attached (.+?)\.",
            }

            for field, pattern in patterns.items():
                matches = re.findall(pattern, email_content, re.DOTALL)
                if matches:
                    value = matches[0].strip()
                    # Map the field to the appropriate section in your schema
                    if field in [
                        "Policy #",
                        "Adjuster Name",
                        "Adjuster Email",
                        "Adjuster Phone Number",
                    ]:
                        section = "Adjuster Information"
                    elif field in ["Carrier Claim Number"]:
                        section = "Requesting Party"
                    elif field in [
                        "Public Adjuster",
                        "Public Adjuster Phone",
                        "Public Adjuster Email",
                    ]:
                        section = "Insured Information"
                    elif field in ["Name", "Loss Address"]:
                        section = "Insured Information"
                    elif field in [
                        "Date of Loss/Occurrence",
                        "Cause of loss",
                        "Loss Description",
                    ]:
                        section = "Assignment Information"
                    elif field in ["Attachment(s)"]:
                        section = "Assignment Information"
                    else:
                        section = "Additional Information"

                    extracted_data.setdefault(section, {}).setdefault(field, []).append(
                        value
                    )
                    self.logger.debug(f"Extracted {field}: {value}")

            self.logger.debug(f"Regex Extraction Result: {extracted_data}")
            return extracted_data
        except Exception as e:
            self.logger.error(f"Error during regex extraction: {e}", exc_info=True)
            return {}

    def layout_aware_parsing(self, email_content: str) -> Dict[str, Any]:
        """
        Performs Layout-Aware parsing on the email content.

        Args:
            email_content (str): The email content to parse.

        Returns:
            Dict[str, Any]: Extracted layout-aware entities.
        """
        # Since we don't have actual layout information, we can skip or simplify this method.
        return {}

    def sequence_model_extract(self, summary_text: str) -> Dict[str, Any]:
        """
        Extracts key-value pairs from the summary text generated by the Sequence Model.

        Args:
            summary_text (str): The summary text to parse.

        Returns:
            Dict[str, Any]: Extracted data from the summary.
        """
        extracted_sequence: Dict[str, Any] = {}
        try:
            self.logger.debug("Extracting data from Sequence Model summary.")
            # Split the summary text into items
            for item in summary_text.split(","):
                if ":" in item:
                    key, value = item.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    # Map keys to the appropriate sections and fields
                    for section, fields in QUICKBASE_SCHEMA.items():
                        for field in fields:
                            if key.lower() == field.lower():
                                extracted_sequence.setdefault(section, {}).setdefault(
                                    field, []
                                ).append(value)
                                self.logger.debug(
                                    f"Extracted {key}: {value} into section {section}"
                                )

            self.logger.debug(f"Sequence Model Extraction Result: {extracted_sequence}")
            return extracted_sequence
        except Exception as e:
            self.logger.error(
                f"Error during Sequence Model extraction: {e}", exc_info=True
            )
            return {}

    def validation_parsing(self, email_content: str, parsed_data: Dict[str, Any]):
        """
        Validates the extracted data against the original email content.

        Args:
            email_content (str): The original email content.
            parsed_data (Dict[str, Any]): The data parsed so far.
        """
        # Implement validation logic if necessary
        pass

    def _stage_schema_validation(self, parsed_data: Dict[str, Any]):
        """
        Validates the parsed data against the predefined schema.

        Args:
            parsed_data (Dict[str, Any]): The data to validate.
        """
        self.logger.debug("Starting schema validation.")
        missing_fields: List[str] = []
        inconsistent_fields: List[str] = []

        try:
            # Iterate over the schema to check for missing and inconsistent fields
            for section, fields in QUICKBASE_SCHEMA.items():
                for field in fields:
                    value = parsed_data.get(section, {}).get(field)
                    if not value or value == ["N/A"]:
                        missing_fields.append(f"{section} -> {field}")
                        self.logger.debug(f"Missing field: {section} -> {field}")
                        continue

                    # Perform fuzzy matching for known values
                    known_values = self.config.get("known_values", {}).get(field, [])
                    if known_values:
                        best_match = max(
                            known_values,
                            key=lambda x: fuzz.partial_ratio(
                                x.lower(), value[0].lower()
                            ),
                            default=None,
                        )
                        if best_match and fuzz.partial_ratio(
                            best_match.lower(), value[0].lower()
                        ) >= self.config.get("fuzzy_threshold", 90):
                            parsed_data[section][field] = [best_match]
                            self.logger.debug(
                                f"Updated {field} in {section} with best match: {best_match}"
                            )
                        else:
                            inconsistent_fields.append(f"{section} -> {field}")
                            self.logger.debug(
                                f"Inconsistent field: {section} -> {field} with value {value}"
                            )

            # Add missing and inconsistent fields to parsed data
            if missing_fields:
                parsed_data["missing_fields"] = missing_fields
                self.logger.info(f"Missing fields identified: {missing_fields}")

            if inconsistent_fields:
                parsed_data["inconsistent_fields"] = inconsistent_fields
                self.logger.info(
                    f"Inconsistent fields identified: {inconsistent_fields}"
                )
        except Exception as e:
            self.logger.error(f"Error during schema validation: {e}", exc_info=True)

    def _stage_post_processing(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs post-processing on the parsed data, including formatting and verification.

        Args:
            parsed_data (Dict[str, Any]): The data to post-process.

        Returns:
            Dict[str, Any]: The post-processed data.
        """
        self.logger.debug("Starting post-processing of parsed data.")
        skip_sections = [
            "TransformerEntities",
            "Entities",
            "missing_fields",
            "inconsistent_fields",
            "user_notifications",
            "validation_issues",
        ]

        try:
            # Iterate over sections and fields to format dates and phone numbers
            for section, fields in parsed_data.items():
                if section in skip_sections:
                    continue
                if not isinstance(fields, dict):
                    continue

                for field, value_list in fields.items():
                    if not isinstance(value_list, list):
                        continue
                    for idx, value in enumerate(value_list):
                        if "Date" in field:
                            formatted_date = self.format_date(value)
                            parsed_data[section][field][idx] = formatted_date
                            self.logger.debug(
                                f"Formatted date for {field} in {section}: {formatted_date}"
                            )
                        if "Phone Number" in field or "Contact #" in field:
                            formatted_phone = self.format_phone_number(value)
                            parsed_data[section][field][idx] = formatted_phone
                            self.logger.debug(
                                f"Formatted phone number for {field} in {section}: {formatted_phone}"
                            )

            # Verify attachments if mentioned
            attachments = parsed_data.get("Assignment Information", {}).get(
                "Attachment(s)", []
            )
            if attachments and not self.verify_attachments(attachments):
                parsed_data["user_notifications"] = (
                    "Attachments mentioned but not found in the email."
                )
                self.logger.info(
                    "Attachments mentioned in email but not found. User notification added."
                )

            self.logger.debug("Post-processing completed.")
            return parsed_data
        except Exception as e:
            self.logger.error(f"Error during post-processing: {e}", exc_info=True)
            return parsed_data

    def format_date(self, date_str: str) -> str:
        """
        Formats a date string into a standardized format.

        Args:
            date_str (str): The date string to format.

        Returns:
            str: The formatted date string or "N/A" if invalid.
        """
        if date_str == "N/A":
            return date_str

        self.logger.debug(f"Formatting date: {date_str}")
        try:
            date_obj = dateutil_parser.parse(date_str, fuzzy=True)
            standardized_date = date_obj.strftime("%Y-%m-%d")
            self.logger.debug(f"Standardized date: {standardized_date}")
            return standardized_date
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to format date: {date_str} - {e}")
            return "N/A"

    def format_phone_number(self, phone: str) -> str:
        """
        Formats a phone number into E.164 format.

        Args:
            phone (str): The phone number string to format.

        Returns:
            str: The formatted phone number or "N/A" if invalid.
        """
        if phone == "N/A":
            return phone

        self.logger.debug(f"Formatting phone number: {phone}")
        try:
            parsed_number = phonenumbers.parse(phone, "US")
            if phonenumbers.is_valid_number(parsed_number):
                formatted_number = phonenumbers.format_number(
                    parsed_number, PhoneNumberFormat.E164
                )
                self.logger.debug(f"Formatted phone number: {formatted_number}")
                return formatted_number
            else:
                self.logger.warning(f"Invalid phone number: {phone}")
                return "N/A"
        except phonenumbers.NumberParseException as e:
            self.logger.warning(f"Failed to parse phone number: {phone} - {e}")
            return "N/A"

    def verify_attachments(self, attachments: List[str]) -> bool:
        """
        Verifies if the mentioned attachments are present.

        Args:
            attachments (List[str]): A list of attachment names.

        Returns:
            bool: True if attachments are verified, False otherwise.
        """
        # Placeholder for actual attachment verification logic
        self.logger.debug(f"Verifying attachments: {attachments}")
        # Assuming all attachments are verified for now
        return True

    def _stage_json_validation(self, parsed_data: Dict[str, Any]):
        """
        Validates the final parsed data against a JSON schema.

        Args:
            parsed_data (Dict[str, Any]): The data to validate.
        """
        self.logger.debug("Starting JSON validation.")
        try:
            is_valid, error_message = validate_json(parsed_data)
            if is_valid:
                self.logger.info("JSON validation passed.")
            else:
                self.logger.error(f"JSON validation failed: {error_message}")
                parsed_data["validation_issues"] = parsed_data.get(
                    "validation_issues", []
                ) + [error_message]
        except Exception as e:
            self.logger.error(f"Error during JSON validation: {e}", exc_info=True)
            parsed_data["validation_issues"] = parsed_data.get(
                "validation_issues", []
            ) + [str(e)]

    def parse_email(
        self, email_content: str, parser_option: Any = None
    ) -> Dict[str, Any]:
        """
        Parses the email content using the enhanced parser.

        Args:
            email_content (str): The email content to parse.
            parser_option (Any, optional): Additional parser options.

        Returns:
            Dict[str, Any]: The parsed data.
        """
        self.logger.info("parse_email called.")
        return self.parse(email_content)
