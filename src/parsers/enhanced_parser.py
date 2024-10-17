import logging
from typing import Optional, Dict, Any, List
import os
from transformers import pipeline
from huggingface_hub import login
from thefuzz import fuzz
from dateutil import parser as dateutil_parser
import phonenumbers
from phonenumbers import PhoneNumberFormat
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys

from src.parsers.base_parser import BaseParser
from src.utils.validation import validate_json
from src.utils.config_loader import ConfigLoader
from src.utils.quickbase_schema import QUICKBASE_SCHEMA

# Define a timeout constant for long-running LLM processes
LLM_TIMEOUT_SECONDS = 500


# Custom exception for handling timeouts
class TimeoutException(Exception):
    pass


# Handler function for timeouts
def llm_timeout_handler():
    raise TimeoutException("LLM processing timed out.")


class EnhancedParser(BaseParser):
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
            # Stage 1: NER Parsing
            self.logger.info("Stage 1: NER Parsing.")
            parsed_data.update(self._stage_ner_parsing(email_content))

            # Stage 2: Layout-Aware Parsing
            self.logger.info("Stage 2: Layout-Aware Parsing.")
            parsed_data.update(self._stage_layout_aware_parsing(email_content))

            # Stage 3: Sequence Model Parsing
            self.logger.info("Stage 3: Sequence Model Parsing.")
            parsed_data.update(self._stage_sequence_model_parsing(email_content))

            # Stage 4: Validation Parsing
            self.logger.info("Stage 4: Validation Parsing.")
            self._stage_validation(email_content, parsed_data)

            # Stage 5: Schema Validation
            self.logger.info("Stage 5: Schema Validation.")
            self._stage_schema_validation(parsed_data)

            # Stage 6: Post Processing
            self.logger.info("Stage 6: Post Processing.")
            parsed_data = self._stage_post_processing(parsed_data)

            # Stage 7: JSON Validation
            self.logger.info("Stage 7: JSON Validation.")
            self._stage_json_validation(parsed_data)

            self.logger.info("Parsing process completed successfully.")
            return parsed_data

        except TimeoutException as te:
            self.logger.error(f"Validation Model parsing timeout: {te}", exc_info=True)
            raise RuntimeError(f"Validation Model parsing timeout: {te}") from te
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during parsing: {e}") from e

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

            # Process each recognized entity
            for entity in entities:
                label = entity.get("entity_group")
                text = entity.get("word")

                if label and text:
                    label_lower = label.lower()
                    self.logger.debug(f"Processing entity: {label} - {text}")

                    # Map entities to the appropriate fields
                    if label_lower == "per":
                        # Assuming 'PER' labels correspond to names
                        extracted_entities.setdefault(
                            "Insured Information", {}
                        ).setdefault("Name", []).append(text)
                    elif label_lower == "org":
                        # Assuming 'ORG' labels correspond to Insurance Company
                        extracted_entities.setdefault(
                            "Requesting Party", {}
                        ).setdefault("Insurance Company", []).append(text)
                    elif label_lower == "loc":
                        # Assuming 'LOC' labels correspond to Loss Address
                        extracted_entities.setdefault(
                            "Insured Information", {}
                        ).setdefault("Loss Address", []).append(text)
                    elif label_lower == "date":
                        extracted_entities.setdefault(
                            "Assignment Information", {}
                        ).setdefault("Date of Loss/Occurrence", []).append(text)
                    elif label_lower == "email":
                        extracted_entities.setdefault(
                            "Adjuster Information", {}
                        ).setdefault("Adjuster Email", []).append(text)

            self.logger.debug(f"NER Parsing Result: {extracted_entities}")
            return extracted_entities
        except Exception as e:
            self.logger.error(f"Error during NER parsing: {e}", exc_info=True)
            return {}

    def layout_aware_parsing(self, email_content: str) -> Dict[str, Any]:
        """
        Performs Layout-Aware parsing on the email content.

        Args:
            email_content (str): The email content to parse.

        Returns:
            Dict[str, Any]: Extracted layout-aware entities.
        """
        try:
            self.logger.debug("Starting Layout-Aware pipeline.")

            # Split the email content into words
            words = email_content.split()

            # Create dummy bounding boxes (since we don't have actual layout data)
            boxes = [[0, 0, 0, 0] for _ in words]

            # Prepare inputs for the pipeline
            inputs = {
                "words": words,
                "boxes": boxes,
            }

            entities = self.layoutlm_pipeline(**inputs)
            extracted_layout: Dict[str, Any] = {}

            # Process each recognized entity
            for entity in entities:
                label = entity.get("entity_group")
                text = entity.get("word")

                if label and text:
                    label_lower = label.lower()
                    self.logger.debug(f"Processing entity: {label} - {text}")

                    # Map entities to the appropriate fields
                    if label_lower in ["address", "location", "loc"]:
                        extracted_layout.setdefault(
                            "Insured Information", {}
                        ).setdefault("Loss Address", []).append(text)
                    elif label_lower in ["damage", "cost"]:
                        extracted_layout.setdefault(
                            "Assignment Information", {}
                        ).setdefault("Loss Description", []).append(text)

            self.logger.debug(f"Layout-Aware Parsing Result: {extracted_layout}")
            return extracted_layout
        except Exception as e:
            self.logger.error(f"Error during Layout-Aware parsing: {e}", exc_info=True)
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
        try:
            self.logger.debug("Starting validation parsing.")
            # Prepare the prompt for the validation model
            prompt = (
                f"Validate the following extracted data against the original email content. "
                f"Ensure all fields are consistent and complete.\n\n"
                f"Email Content:\n{email_content}\n\n"
                f"Extracted Data:\n{parsed_data}\n\n"
                f"Provide a list of any missing or inconsistent fields."
            )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.validation_pipeline,
                    prompt,
                    max_new_tokens=150,
                    do_sample=False,
                )
                try:
                    # Wait for the result with a timeout
                    validation_response = future.result(timeout=LLM_TIMEOUT_SECONDS)
                    validation_text = validation_response[0]["generated_text"]
                    self.logger.debug(f"Validation Model Response: {validation_text}")

                    # Parse the validation response
                    issues = self.parse_validation_response(validation_text)
                    if issues:
                        parsed_data["validation_issues"] = issues
                        self.logger.info(f"Validation issues found: {issues}")
                except TimeoutError:
                    self.logger.warning("Validation Model parsing timed out.")
                except Exception as e:
                    self.logger.error(
                        f"Error during Validation Model parsing: {e}", exc_info=True
                    )
        except Exception as e:
            self.logger.error(f"Error in validation_parsing: {e}", exc_info=True)

    def parse_validation_response(self, validation_text: str) -> List[str]:
        """
        Parses the validation response text into a list of issues.

        Args:
            validation_text (str): The response from the validation model.

        Returns:
            List[str]: A list of validation issues.
        """
        issues = []
        try:
            lines = validation_text.strip().split("\n")
            self.logger.debug("Parsing validation response.")

            for line in lines:
                if line.strip():
                    issues.append(line.strip())
                    self.logger.debug(f"Validation issue identified: {line.strip()}")

            self.logger.debug(f"Validation Issues: {issues}")
            return issues
        except Exception as e:
            self.logger.error(f"Error parsing validation response: {e}", exc_info=True)
            return issues

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

            # Handle TransformerEntities for additional data
            transformer_entities = parsed_data.get("TransformerEntities", {})
            for _, entities in transformer_entities.items():
                for entity in entities:
                    text = entity.get("text", "").strip()
                    confidence = entity.get("confidence", 0.0)
                    self.logger.debug(
                        f"Processing TransformerEntity: {text} with confidence {confidence}"
                    )
                    if "policy" in text.lower():
                        parsed_data.setdefault("Adjuster Information", {}).setdefault(
                            "Policy #", []
                        ).append(text)
                        self.logger.debug(
                            f"Added Policy # to Adjuster Information: {text}"
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
