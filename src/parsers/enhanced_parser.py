# src/parsers/enhanced_parser.py

import logging
import re
import signal
from datetime import datetime
from typing import Optional, Dict, Any, List

from transformers import pipeline
from thefuzz import fuzz

from src.parsers.base_parser import BaseParser
from src.utils.validation import validate_json
from src.utils.config_loader import ConfigLoader

# Constants for timeout handling
LLM_TIMEOUT_SECONDS = 5


class TimeoutException(Exception):
    """Custom exception for handling LLM processing timeouts."""
    pass


def llm_timeout_handler(signum, frame):
    """Signal handler for LLM processing timeout."""
    raise TimeoutException("LLM processing timed out.")


class EnhancedParser(BaseParser):
    """An enhanced email parser with multi-stage processing and risk management."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initializes the EnhancedParser.

        Args:
            logger (logging.Logger, optional): Custom logger instance. If None, a default logger is created.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = ConfigLoader.load_config()
        self.init_rule_based_patterns()
        self.init_llm()
        self.logger.info("EnhancedParser initialized successfully.")

    def init_rule_based_patterns(self):
        """Initialize regex patterns for rule-based parsing from configuration."""
        self.patterns = {}
        patterns_config = self.config.get('patterns', {})
        for section, fields in patterns_config.items():
            self.patterns[section] = {}
            for field, pattern in fields.items():
                try:
                    self.patterns[section][field] = re.compile(pattern, re.IGNORECASE)
                    self.logger.debug(f"Compiled regex for '{section} -> {field}': {pattern}")
                except re.error as e:
                    self.logger.error(f"Invalid regex pattern for '{section} -> {field}': {e}")
                    raise ValueError(f"Invalid regex pattern for '{section} -> {field}': {e}")

    def init_llm(self):
        """Initialize the Language Model for LLM-based parsing."""
        try:
            self.llm = pipeline(
                "ner",
                model=self.config.get('llm_model', 'distilbert-base-uncased'),
                tokenizer=self.config.get('llm_model', 'distilbert-base-uncased'),
                aggregation_strategy="simple"
            )
            self.logger.info(f"Loaded LLM model '{self.config.get('llm_model', 'distilbert-base-uncased')}' successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load LLM model '{self.config.get('llm_model', 'distilbert-base-uncased')}': {e}")
            raise RuntimeError(f"Failed to load LLM model '{self.config.get('llm_model', 'distilbert-base-uncased')}': {e}")

    def parse(self, email_content: str) -> Dict[str, Any]:
        """
        Parse the email content through multiple stages.

        Args:
            email_content (str): The raw email content to parse.

        Returns:
            dict: Parsed data as a dictionary.

        Raises:
            ValueError: If JSON schema validation fails.
            RuntimeError: For any other parsing errors.
        """
        self.logger.info("Starting parsing process.")
        parsed_data = {}

        try:
            # Stage 1: Rule-Based Parsing
            self.logger.debug("Stage 1: Rule-Based Parsing started.")
            rule_based_data = self.rule_based_parsing(email_content)
            parsed_data.update(rule_based_data)
            self.logger.debug(f"Rule-Based Parsed Data: {rule_based_data}")

            # Stage 2: LLM-Based Parsing
            self.logger.debug("Stage 2: LLM-Based Parsing started.")
            llm_data = self.llm_parsing(email_content)
            parsed_data.update(llm_data)
            self.logger.debug(f"LLM-Based Parsed Data: {llm_data}")

            # Stage 3: Schema Validation and Cross-Check
            self.logger.debug("Stage 3: Schema Validation and Cross-Check started.")
            self.schema_validation(parsed_data)

            # Stage 4: Post-Processing and Final Refinement
            self.logger.debug("Stage 4: Post-Processing and Final Refinement started.")
            parsed_data = self.post_processing(parsed_data)

            # Validate the extracted data against the JSON schema
            is_valid, error_message = validate_json(parsed_data)
            if not is_valid:
                self.logger.error(f"JSON Schema Validation Error: {error_message}")
                raise ValueError(f"JSON Schema Validation Error: {error_message}")

            self.logger.info("Parsing process completed successfully.")
            return parsed_data

        except TimeoutException as te:
            self.logger.error(f"LLM parsing timeout: {te}")
            raise RuntimeError(f"LLM parsing timeout: {te}") from te
        except ValueError as ve:
            self.logger.error(f"ValueError during parsing: {ve}")
            raise ve
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during parsing: {e}") from e

    def rule_based_parsing(self, email_content: str) -> Dict[str, Any]:
        """
        Extract structured data using rule-based parsing.

        Args:
            email_content (str): The raw email content.

        Returns:
            dict: Extracted structured data.
        """
        parsed_data = {}
        for section, fields in self.patterns.items():
            parsed_data[section] = {}
            for field, pattern in fields.items():
                match = pattern.search(email_content)
                if match:
                    parsed_data[section][field] = match.group(1).strip()
                    self.logger.debug(f"Rule-Based Parsing - Found '{section} -> {field}': {parsed_data[section][field]}")
                else:
                    parsed_data[section][field] = "N/A"
                    self.logger.debug(f"Rule-Based Parsing - '{section} -> {field}' not found, set to 'N/A'")
        return parsed_data

    def llm_parsing(self, email_content: str) -> Dict[str, Any]:
        """
        Extract unstructured data using LLM-based parsing.

        Args:
            email_content (str): The raw email content.

        Returns:
            dict: Extracted unstructured data with confidence scores.
        """
        # Set up the timeout handler
        signal.signal(signal.SIGALRM, llm_timeout_handler)
        signal.alarm(LLM_TIMEOUT_SECONDS)

        try:
            entities = self.llm(email_content)
            signal.alarm(0)  # Disable the alarm
            extracted_entities = {}
            for entity in entities:
                entity_label = entity.get("entity_group")
                entity_text = entity.get("word")
                confidence = entity.get("score", 0.0)
                if entity_label and entity_text:
                    extracted_entities.setdefault(entity_label, []).append({
                        "text": entity_text,
                        "confidence": confidence
                    })
            self.logger.debug(f"LLM-Based Entities Extracted: {extracted_entities}")
            return {"TransformerEntities": extracted_entities}
        except TimeoutException:
            self.logger.warning("LLM parsing timed out.")
            return {}
        except Exception as e:
            self.logger.error(f"Error during LLM parsing: {e}")
            return {}
        finally:
            signal.alarm(0)  # Ensure the alarm is disabled

    def schema_validation(self, parsed_data: Dict[str, Any]):
        """
        Validate extracted data against the QuickBase schema and perform fuzzy matching.

        Args:
            parsed_data (dict): The parsed data to validate.

        Modifies:
            parsed_data (dict): Updates with corrected values or flags inconsistencies.
        """
        missing_fields = []
        inconsistent_fields = []

        for section, fields in self.config.get('patterns', {}).items():
            for field in fields.keys():
                value = parsed_data.get(section, {}).get(field)
                if not value or value == "N/A":
                    missing_fields.append(f"{section} -> {field}")
                    continue

                known_values = self.config.get('known_values', {}).get(field, [])
                best_match = max(
                    known_values,
                    key=lambda x: fuzz.partial_ratio(x.lower(), value.lower()),
                    default=None
                )
                if best_match and fuzz.partial_ratio(best_match.lower(), value.lower()) >= self.config.get('fuzzy_threshold', 90):
                    parsed_data[section][field] = best_match
                    self.logger.debug(f"Fuzzy matched '{section} -> {field}' to '{best_match}'.")
                else:
                    inconsistent_fields.append(f"{section} -> {field}")
                    self.logger.debug(f"Inconsistent field '{section} -> {field}': '{value}'.")

        if missing_fields:
            parsed_data["missing_fields"] = missing_fields
            self.logger.warning(f"Missing fields: {missing_fields}")

        if inconsistent_fields:
            parsed_data["inconsistent_fields"] = inconsistent_fields
            self.logger.warning(f"Inconsistent fields: {inconsistent_fields}")

    def post_processing(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize data formats, perform sanity checks, and prepare data for QuickBase.

        Args:
            parsed_data (dict): The parsed data to refine.

        Returns:
            dict: The refined and standardized data.
        """
        # Example: Format dates and phone numbers
        for section, fields in parsed_data.items():
            for field, value in fields.items():
                if 'Date' in field:
                    parsed_data[section][field] = self.format_date(value)
                if 'Phone Number' in field:
                    parsed_data[section][field] = self.format_phone_number(value)

        # Example: Attachment verification
        attachments = parsed_data.get("Attachment(s)", [])
        if attachments and not self.verify_attachments(attachments):
            parsed_data["user_notifications"] = "Attachments mentioned but not found in the email."
            self.logger.warning("Attachments mentioned but not found.")

        return parsed_data

    def format_date(self, date_str: str) -> str:
        """
        Parse and standardize date formats to 'YYYY-MM-DD'.

        Args:
            date_str (str): The raw date string.

        Returns:
            str: The standardized date or 'N/A' if parsing fails.
        """
        if date_str == "N/A":
            return date_str

        for fmt in self.config.get('date_formats', []):
            try:
                date_obj = datetime.strptime(date_str, fmt)
                standardized_date = date_obj.strftime("%Y-%m-%d")
                self.logger.debug(f"Formatted date '{date_str}' to '{standardized_date}' using format '{fmt}'.")
                return standardized_date
            except ValueError:
                continue
        # Fallback to dateutil if available
        try:
            from dateutil import parser as dateutil_parser
            date_obj = dateutil_parser.parse(date_str, fuzzy=True)
            standardized_date = date_obj.strftime("%Y-%m-%d")
            self.logger.debug(f"Formatted date '{date_str}' to '{standardized_date}' using dateutil.")
            return standardized_date
        except Exception as e:
            self.logger.warning(f"Failed to parse date '{date_str}': {e}")
            return "N/A"

    def format_phone_number(self, phone: str) -> str:
        """
        Format phone numbers to a standard format.

        Args:
            phone (str): The raw phone number.

        Returns:
            str: The formatted phone number or 'N/A' if invalid.
        """
        if phone == "N/A":
            return phone

        try:
            import phonenumbers
            from phonenumbers import PhoneNumberFormat

            parsed_number = phonenumbers.parse(phone, "US")  # Default region
            if phonenumbers.is_valid_number(parsed_number):
                formatted_number = phonenumbers.format_number(parsed_number, PhoneNumberFormat.E164)
                self.logger.debug(f"Formatted phone number '{phone}' to '{formatted_number}'.")
                return formatted_number
            else:
                self.logger.warning(f"Invalid phone number format: '{phone}'.")
                return "N/A"
        except phonenumbers.NumberParseException as e:
            self.logger.warning(f"Error parsing phone number '{phone}': {e}")
            return "N/A"

    def verify_attachments(self, attachments: List[str]) -> bool:
        """
        Verify if the mentioned attachments are present in the email.

        Args:
            attachments (List[str]): List of attachment names or URLs.

        Returns:
            bool: True if all attachments are present, False otherwise.
        """
        # Placeholder implementation
        # Actual implementation would depend on how attachments are handled/stored
        # For now, we'll assume attachments are always present
        self.logger.debug("Verifying attachments: %s", attachments)
        return True  # Modify as per actual attachment verification logic
