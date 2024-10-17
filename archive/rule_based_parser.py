# src\parsers\rule_based_parser.py

"""Module containing the RuleBasedParser class for parsing emails using a rule-based approach."""

# Standard library imports
import logging
import os
import re
import time
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Dict, Any

# Third-party imports
import yaml
import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat
import spacy
from spacy.cli import download as spacy_download
from spacy.util import is_package
from dateutil import parser as dateutil_parser
from dateutil.parser import ParserError

# Local application imports
from src.parsers.base_parser import BaseParser
from src.utils.validation import validate_json

# Constants for repeated strings
REQUESTING_PARTY = "Requesting Party"
INSURED_INFORMATION = "Insured Information"
ADJUSTER_INFORMATION = "Adjuster Information"
ASSIGNMENT_INFORMATION = "Assignment Information"
ASSIGNMENT_TYPE = "Assignment Type"
ADDITIONAL_DETAILS = "Additional details/Special Instructions"
ATTACHMENTS = "Attachment(s)"
CARRIER_CLAIM_NUMBER = "Carrier Claim Number"
OWNER_OR_TENANT = "Owner or Tenant"
ADJUSTER_PHONE_NUMBER = "Adjuster Phone Number"
ADJUSTER_EMAIL = "Adjuster Email"
POLICY_NUMBER = "Policy #"
DATE_OF_LOSS = "Date of Loss/Occurrence"
RESIDENCE_OCCUPIED_DURING_LOSS = "Residence Occupied During Loss"
WAS_SOMEONE_HOME = "Was Someone home at time of damage"

# Constants for field labels to avoid duplication
INSURANCE_COMPANY_LABEL = "Insurance Company"
LOSS_ADDRESS_LABEL = "Loss Address"
ADJUSTER_NAME_LABEL = "Adjuster Name"
DATE_OF_LOSS_LABEL = "Date of Loss"

# Constants for log messages
FOUND_MESSAGE = "Found %s: %s"
FOUND_ADDITIONAL_PATTERN_MESSAGE = "Found %s using additional pattern: %s"
NOT_FOUND_MESSAGE = "%s not found, set to 'N/A'"


class RuleBasedParser(BaseParser):
    """An improved and enhanced rule-based parser for comprehensive email parsing."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        nlp_model: str = "en_core_web_sm",
    ) -> None:
        """
        Initializes the RuleBasedParser.

        Args:
            config_path (str, optional): Path to the YAML configuration file. Defaults to None.
            logger (logging.Logger, optional): Custom logger instance. If None, a default logger is created.
            nlp_model (str, optional): spaCy model name to load. Defaults to "en_core_web_sm".

        Raises:
            FileNotFoundError: If the provided config_path does not exist.
            OSError: If the spaCy model fails to load or download.
            ValueError: If the configuration is invalid.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.enhance_logging()
        self.ensure_spacy_model(nlp_model)

        try:
            self.nlp = spacy.load(nlp_model)
            self.logger.info("spaCy model '%s' loaded successfully.", nlp_model)
        except OSError as e:
            self.logger.error("Failed to load spaCy model '%s': %s", nlp_model, e)
            raise OSError(f"spaCy model '{nlp_model}' could not be loaded.") from e

        # Load configuration for patterns if provided
        if config_path:
            if not os.path.exists(config_path):
                self.logger.error(
                    "Configuration file not found at path: %s", config_path
                )
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)
                self.logger.info("Loaded parser configuration from %s.", config_path)
            except yaml.YAMLError as e:
                self.logger.error("Error parsing YAML configuration: %s", e)
                raise ValueError("Invalid YAML configuration.") from e
        else:
            config = self.default_config()
            self.logger.info("Loaded default parser configuration.")

        # Validate configuration structure
        required_keys = {"section_headers", "patterns", "entity_labels"}
        if not required_keys.issubset(config.keys()):
            missing = required_keys - config.keys()
            self.logger.error("Configuration missing keys: %s", missing)
            raise ValueError(f"Configuration missing keys: {missing}")

        self.config = config  # Assigning config to self.config for later use

        # Precompile regular expressions for performance
        self.section_headers = config["section_headers"]
        try:
            headers_pattern = r"(?i)^\s*({})\s*:?\s*$".format(
                "|".join(map(re.escape, self.section_headers))
            )
            self.section_pattern = re.compile(headers_pattern, re.MULTILINE)
        except re.error as e:
            self.logger.error("Invalid regex pattern in section headers: %s", e)
            raise ValueError("Invalid regex pattern in section headers.") from e

        # Define patterns for each section
        self.patterns = {}
        for section, fields in config["patterns"].items():
            self.patterns[section] = {}
            for field, pattern in fields.items():
                try:
                    self.patterns[section][field] = re.compile(
                        pattern, re.IGNORECASE | re.DOTALL
                    )
                except re.error as e:
                    self.logger.error(
                        "Invalid regex for %s - %s: %s", section, field, e
                    )
                    raise ValueError(f"Invalid regex for {section} - {field}") from e

        # Additional patterns for common edge cases
        self.additional_patterns = {}
        for section, fields in config.get("additional_patterns", {}).items():
            self.additional_patterns[section] = {}
            for field, pattern in fields.items():
                try:
                    self.additional_patterns[section][field] = re.compile(
                        pattern, re.IGNORECASE | re.DOTALL
                    )
                except re.error as e:
                    self.logger.error(
                        "Invalid additional regex for %s - %s: %s", section, field, e
                    )
                    raise ValueError(
                        f"Invalid additional regex for {section} - {field}"
                    ) from e

        # Load date formats and boolean values from config if available
        self.date_formats = config.get(
            "date_formats",
            [
                "%m/%d/%Y",
                "%m-%d-%Y",
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
                "%Y.%m.%d",
                "%d.%m.%Y",
                "%Y%m%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
            ],
        )
        self.boolean_values = config.get(
            "boolean_values",
            {
                "positive": [
                    "yes",
                    "y",
                    "true",
                    "t",
                    "1",
                    "x",
                    "[x]",
                    "[X]",
                    "(x)",
                    "(X)",
                ],
                "negative": [
                    "no",
                    "n",
                    "false",
                    "f",
                    "0",
                    "[ ]",
                    "()",
                    "[N/A]",
                    "(N/A)",
                ],
            },
        )

    def enhance_logging(self) -> None:
        """
        Enhances logging by setting up structured logging and log levels.
        """
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Enhanced logging setup complete.")

    def ensure_spacy_model(
        self, model_name: str = "en_core_web_sm", retries: int = 3, delay: int = 5
    ) -> None:
        """
        Ensures that the specified spaCy model is installed.
        Downloads the model if it's not present.

        Args:
            model_name (str, optional): Name of the spaCy model to ensure. Defaults to "en_core_web_sm".
            retries (int, optional): Number of retry attempts for downloading the model. Defaults to 3.
            delay (int, optional): Initial delay in seconds between retries. Defaults to 5.

        Raises:
            OSError: If the spaCy model cannot be downloaded after retries.
        """
        if not is_package(model_name):
            self.logger.warning(
                "spaCy model '%s' not found. Initiating download...", model_name
            )
            attempt = 0
            current_delay = delay
            while attempt < retries:
                try:
                    spacy_download(model_name)
                    self.logger.info(
                        "Successfully downloaded spaCy model '%s'.", model_name
                    )
                    return
                except Exception as e:
                    attempt += 1
                    self.logger.error(
                        "Attempt %d/%d: Failed to download spaCy model '%s': %s",
                        attempt,
                        retries,
                        model_name,
                        e,
                    )
                    if attempt < retries:
                        self.logger.info("Retrying in %d seconds...", current_delay)
                        time.sleep(current_delay)
                        current_delay *= 2  # Exponential backoff
                    else:
                        self.logger.critical(
                            "Exceeded maximum retries. Unable to download spaCy model '%s'.",
                            model_name,
                        )
                        raise OSError(
                            "Failed to download spaCy model '%s' after %d attempts."
                            % (model_name, retries)
                        ) from e
        else:
            self.logger.info("spaCy model '%s' is already installed.", model_name)

    def default_config(self) -> Dict[str, Any]:
        """
        Provides the default configuration for the parser with improved error handling and flexible regex patterns.

        Returns:
            dict: Default parser configuration.

        Raises:
            ValueError: If the default configuration is invalid.
        """
        default_configuration: Dict[str, Any] = {
            "version": "1.2",
            "section_headers": [
                REQUESTING_PARTY,
                INSURED_INFORMATION,
                ADJUSTER_INFORMATION,
                ASSIGNMENT_INFORMATION,
                ASSIGNMENT_TYPE,
                ADDITIONAL_DETAILS,
                ATTACHMENTS,
            ],
            "patterns": {
                REQUESTING_PARTY: {
                    INSURANCE_COMPANY_LABEL: r"Insurance Company\s*:\s*(.*)",
                    "Handler": r"Handler\s*:\s*(.*)",
                    CARRIER_CLAIM_NUMBER: r"Carrier Claim Number\s*:\s*(.*)",
                },
                INSURED_INFORMATION: {
                    "Name": r"Name\s*:\s*(.*)",
                    "Contact #": r"Contact #\s*:\s*(.*)",
                    LOSS_ADDRESS_LABEL: r"Loss Address\s*:\s*(.*)",
                    "Public Adjuster": r"Public Adjuster\s*:\s*(.*)",
                    OWNER_OR_TENANT: (
                        r"Is the insured (?:an )?(Owner|Tenant) of the loss location\?\s*(Yes|No)"
                    ),
                },
                ADJUSTER_INFORMATION: {
                    ADJUSTER_NAME_LABEL: r"Adjuster Name\s*:\s*(.*)",
                    ADJUSTER_PHONE_NUMBER: r"Adjuster Phone Number\s*:\s*(\+?\d[\d\s\-().]{7,}\d)",
                    ADJUSTER_EMAIL: r"Adjuster Email\s*:\s*([\w\.-]+@[\w\.-]+\.\w+)",
                    "Job Title": r"Job Title\s*:\s*(.*)",
                    "Address": r"Address\s*:\s*(.*)",
                    POLICY_NUMBER: r"Policy #\s*:\s*(\w+)",
                },
                ASSIGNMENT_INFORMATION: {
                    DATE_OF_LOSS_LABEL: (
                        r"Date of Loss(?:/Occurrence)?\s*:\s*"
                        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
                    ),
                    "Cause of loss": r"Cause of loss\s*:\s*(.*)",
                    "Facts of Loss": r"Facts of Loss\s*:\s*(.*)",
                    "Loss Description": r"Loss Description\s*:\s*(.*)",
                    RESIDENCE_OCCUPIED_DURING_LOSS: (
                        r"Residence Occupied During Loss\s*:\s*(Yes|No)"
                    ),
                    WAS_SOMEONE_HOME: (
                        r"Was Someone home at time of damage\s*:\s*(Yes|No)"
                    ),
                    "Repair or Mitigation Progress": r"Repair or Mitigation Progress\s*:\s*(.*)",
                    "Type": r"Type\s*:\s*(.*)",
                    "Inspection type": r"Inspection type\s*:\s*(.*)",
                },
                ASSIGNMENT_TYPE: {
                    "Wind": r"Wind\s*\[\s*([xX])\s*\]",
                    "Structural": r"Structural\s*\[\s*([xX])\s*\]",
                    "Hail": r"Hail\s*\[\s*([xX])\s*\]",
                    "Foundation": r"Foundation\s*\[\s*([xX])\s*\]",
                    "Other": r"Other\s*\[\s*([xX])\s*\]\s*-\s*provide details\s*:\s*(.*)",
                },
                ADDITIONAL_DETAILS: {
                    ADDITIONAL_DETAILS: r"Additional details/Special Instructions\s*:\s*(.*)"
                },
                ATTACHMENTS: {ATTACHMENTS: r"Attachment\(s\)\s*:\s*(.*)"},
            },
            "additional_patterns": {
                REQUESTING_PARTY: {
                    POLICY_NUMBER: r"Policy\s*Number\s*:\s*(\w+)",
                    CARRIER_CLAIM_NUMBER: r"Claim\s*Number\s*:\s*(.*)",
                },
                ASSIGNMENT_INFORMATION: {
                    DATE_OF_LOSS_LABEL: (
                        r"Date of Loss(?:/Occurrence)?\s*:\s*"
                        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
                    )
                },
            },
            "date_formats": [
                "%m/%d/%Y",
                "%m-%d-%Y",
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
                "%Y.%m.%d",
                "%d.%m.%Y",
                "%Y%m%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
            ],
            "boolean_values": {
                "positive": [
                    "yes",
                    "y",
                    "true",
                    "t",
                    "1",
                    "x",
                    "[x]",
                    "[X]",
                    "(x)",
                    "(X)",
                ],
                "negative": [
                    "no",
                    "n",
                    "false",
                    "f",
                    "0",
                    "[ ]",
                    "()",
                    "[N/A]",
                    "(N/A)",
                ],
            },
            "entity_labels": ["PERSON", "ORG", "GPE", "DATE", "PRODUCT"],
        }

        # Validate the default configuration structure
        required_top_level_keys = {
            "version",
            "section_headers",
            "patterns",
            "entity_labels",
        }
        if not required_top_level_keys.issubset(default_configuration.keys()):
            missing_keys = required_top_level_keys - default_configuration.keys()
            self.logger.error("Default configuration missing keys: %s", missing_keys)
            raise ValueError(f"Default configuration missing keys: {missing_keys}")

        # Validate regex patterns
        for section, fields in default_configuration["patterns"].items():
            for field, pattern in fields.items():
                try:
                    re.compile(pattern, re.IGNORECASE | re.DOTALL)
                except re.error as e:
                    self.logger.error(
                        "Invalid regex pattern for %s - %s: %s", section, field, e
                    )
                    raise ValueError(
                        "Invalid regex pattern for %s - %s" % (section, field)
                    ) from e

        # Validate additional_patterns
        for section, fields in default_configuration.get(
            "additional_patterns", {}
        ).items():
            for field, pattern in fields.items():
                try:
                    re.compile(pattern, re.IGNORECASE | re.DOTALL)
                except re.error as e:
                    self.logger.error(
                        "Invalid additional regex pattern for %s - %s: %s",
                        section,
                        field,
                        e,
                    )
                    raise ValueError(
                        "Invalid additional regex pattern for %s - %s"
                        % (section, field)
                    ) from e

        # Validate entity_labels
        if not isinstance(default_configuration["entity_labels"], list) or not all(
            isinstance(label, str) for label in default_configuration["entity_labels"]
        ):
            self.logger.error(
                "Invalid 'entity_labels' in default configuration. It must be a list of strings."
            )
            raise ValueError(
                "Invalid 'entity_labels' in default configuration. It must be a list of strings."
            )

        self.logger.info("Default configuration loaded and validated successfully.")
        return default_configuration

    def preprocess_email_content(self, email_content: str) -> str:
        """
        Cleans up email content to remove extraneous spaces and newlines.

        Args:
            email_content (str): The raw email content.

        Returns:
            str: Preprocessed email content.
        """
        email_content = re.sub(
            r"\s+", " ", email_content
        )  # Remove excess spaces/newlines
        email_content = email_content.strip()  # Remove leading/trailing spaces
        return email_content

    def extract_entities_from_paragraphs(self, email_content: str) -> Dict[str, Any]:
        """
        Use NLP (spaCy) to extract key entities from unstructured email content.

        Args:
            email_content (str): The raw email content.

        Returns:
            dict: A dictionary with extracted entities like names, organizations, dates.
        """
        self.logger.info("Extracting entities using spaCy NLP.")
        doc = self.nlp(email_content)  # Use the pre-loaded model

        extracted_data: Dict[str, Optional[str]] = {
            INSURANCE_COMPANY_LABEL: None,
            "Handler": None,
            ADJUSTER_NAME_LABEL: None,
            DATE_OF_LOSS_LABEL: None,
            LOSS_ADDRESS_LABEL: None,
        }

        for ent in doc.ents:
            if ent.label_ == "ORG":  # Organization could be an insurance company
                if not extracted_data[INSURANCE_COMPANY_LABEL]:
                    extracted_data[INSURANCE_COMPANY_LABEL] = ent.text
            elif ent.label_ == "PERSON":  # Persons could be a handler or adjuster
                if not extracted_data["Handler"]:
                    extracted_data["Handler"] = ent.text
                elif not extracted_data[ADJUSTER_NAME_LABEL]:
                    extracted_data[ADJUSTER_NAME_LABEL] = ent.text
            elif ent.label_ == "DATE":  # Date entity
                if not extracted_data[DATE_OF_LOSS_LABEL]:
                    extracted_data[DATE_OF_LOSS_LABEL] = ent.text
            elif ent.label_ == "GPE":  # Geopolitical Entity for addresses
                if not extracted_data[LOSS_ADDRESS_LABEL]:
                    extracted_data[LOSS_ADDRESS_LABEL] = ent.text

        # Replace None with 'N/A' if not found
        for key, value in extracted_data.items():
            if value is None:
                extracted_data[key] = "N/A"

        self.logger.info("NLP-based extraction completed: %s", extracted_data)
        return extracted_data

    def parse(self, email_content: str) -> Dict[str, Any]:
        """
        Parses the email content using regular expressions and NLP techniques to extract key information.

        Args:
            email_content (str): The raw email content to parse.

        Returns:
            dict: Parsed data as a dictionary.

        Raises:
            ValueError: If JSON schema validation fails.
            RuntimeError: For any other parsing errors.
        """
        self.logger.info("Starting parsing of email content.")
        extracted_data: Dict[str, Any] = {}

        try:
            # Step 1: Preprocess email content
            preprocessed_content = self.preprocess_email_content(email_content)

            # Step 2: Extract sections based on the assignment schema (structured data)
            sections = self.split_into_sections(preprocessed_content)

            # Step 3: Extract data from each section using section-based methods
            for section, content in sections.items():
                method_name = "extract_" + self.snake_case(section)
                extract_method = getattr(self, method_name, None)
                if callable(extract_method):
                    try:
                        self.logger.debug("Extracting data for section: %s", section)
                        data = extract_method(content)
                        extracted_data.update(data)
                    except Exception as e:
                        self.logger.error(
                            "Error extracting section '%s': %s",
                            section,
                            e,
                            exc_info=True,
                        )
                        extracted_data.update(self.default_section_data(section))
                else:
                    self.logger.warning(
                        "No extraction method found for section: %s", section
                    )
                    extracted_data.update(self.default_section_data(section))

            # Step 4: Ensure 'Additional details/Special Instructions' is always present
            if ADDITIONAL_DETAILS not in extracted_data:
                self.logger.info(
                    "Ensuring 'Additional details/Special Instructions' section is present."
                )
                extracted_data.update(self.default_section_data(ADDITIONAL_DETAILS))

            # Step 5: Apply NLP-based extraction for unstructured paragraphs
            self.logger.info("Starting NLP-based entity extraction.")
            nlp_extracted_data = self.extract_entities_from_paragraphs(
                preprocessed_content
            )
            extracted_data.update(nlp_extracted_data)

            # Step 6: Validate the extracted data against the JSON schema
            is_valid, error_message = validate_json(extracted_data)
            if not is_valid:
                self.logger.error("JSON Schema Validation Error: %s", error_message)
                raise ValueError(f"JSON Schema Validation Error: {error_message}")

            self.logger.debug("Extracted Data: %s", extracted_data)
            self.logger.info("Successfully parsed email content.")
            return extracted_data

        except ValueError as ve:
            # Re-raise ValueError for JSON schema validation errors
            raise ve

        except RuntimeError as re_err:
            # Re-raise RuntimeError for parsing errors
            raise re_err

        except Exception as e:
            # Log the error with more context and re-raise the exception
            self.logger.error(
                "Unexpected error during parsing email content: %s",
                str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to parse email content: {str(e)}") from e

    def snake_case(self, text: str) -> str:
        """
        Converts the given text to snake_case by removing non-word characters and replacing spaces with underscores.

        Args:
            text (str): The input text to convert.

        Returns:
            str: The converted snake_case text.
        """
        if not isinstance(text, str):
            self.logger.error("Input to snake_case is not a string: %s", type(text))
            raise TypeError("Input must be a string.")

        original_text = text.strip().lower()
        converted_text = re.sub(r"[^\w\s]", "", original_text, flags=re.UNICODE)
        converted_text = re.sub(r"\s+", "_", converted_text, flags=re.UNICODE)
        self.logger.debug(
            "Converted '%s' to snake_case: '%s'", original_text, converted_text
        )
        return converted_text

    def split_into_sections(self, email_content: str) -> Dict[str, Any]:
        """
        Splits the email content into sections based on the assignment schema headers.

        Args:
            email_content (str): The raw email content.

        Returns:
            dict: Ordered dictionary mapping sections to their content.
        """
        self.logger.debug("Starting to split email content into sections.")
        sections: Dict[str, Any] = OrderedDict()

        # Find all section headers and their positions
        matches = list(self.section_pattern.finditer(email_content))
        self.logger.debug("Found %d section headers.", len(matches))

        for i, match in enumerate(matches):
            section = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(email_content)
            content = email_content[start:end].strip()

            if section in sections:
                # Handle duplicate sections by appending content
                if isinstance(sections[section], list):
                    sections[section].append(content)
                    self.logger.warning(
                        "Duplicate section header '%s' found. Appending content.",
                        section,
                    )
                else:
                    sections[section] = [sections[section], content]
                    self.logger.warning(
                        "Duplicate section header '%s' found. Converting content to list.",
                        section,
                    )
            else:
                sections[section] = content
                self.logger.debug(
                    "Captured section '%s' with content length %d.",
                    section,
                    len(content),
                )

        # Handle sections that were not found by setting default data
        for section in self.section_headers:
            if section not in sections:
                self.logger.warning("Section '%s' not found in email content.", section)
                sections[section] = ""  # Or set to a default value as needed

        self.logger.debug("Sections Found: %s", list(sections.keys()))
        return sections

    def default_section_data(self, section: str) -> Dict[str, Any]:
        """
        Provides default data structure for missing sections.

        Args:
            section (str): The name of the section.

        Returns:
            dict: Default data for the section.
        """
        default_data: Dict[str, Any] = {}
        if section == REQUESTING_PARTY:
            default_data[REQUESTING_PARTY] = {
                INSURANCE_COMPANY_LABEL: "N/A",
                "Handler": "N/A",
                CARRIER_CLAIM_NUMBER: "N/A",
            }
        elif section == INSURED_INFORMATION:
            default_data[INSURED_INFORMATION] = {
                "Name": "N/A",
                "Contact #": "N/A",
                LOSS_ADDRESS_LABEL: "N/A",
                "Public Adjuster": "N/A",
                OWNER_OR_TENANT: "N/A",
            }
        elif section == ADJUSTER_INFORMATION:
            default_data[ADJUSTER_INFORMATION] = {
                ADJUSTER_NAME_LABEL: "N/A",
                ADJUSTER_PHONE_NUMBER: "N/A",
                ADJUSTER_EMAIL: "N/A",
                "Job Title": "N/A",
                "Address": "N/A",
                POLICY_NUMBER: "N/A",
            }
        elif section == ASSIGNMENT_INFORMATION:
            default_data[ASSIGNMENT_INFORMATION] = {
                DATE_OF_LOSS_LABEL: "N/A",
                "Cause of loss": "N/A",
                "Facts of Loss": "N/A",
                "Loss Description": "N/A",
                RESIDENCE_OCCUPIED_DURING_LOSS: "N/A",
                WAS_SOMEONE_HOME: "N/A",
                "Repair or Mitigation Progress": "N/A",
                "Type": "N/A",
                "Inspection type": "N/A",
            }
        elif section == ASSIGNMENT_TYPE:
            default_data[ASSIGNMENT_TYPE] = {
                "Wind": False,
                "Structural": False,
                "Hail": False,
                "Foundation": False,
                "Other": {"Checked": False, "Details": "N/A"},
            }
        elif section == ADDITIONAL_DETAILS:
            default_data[ADDITIONAL_DETAILS] = "N/A"
        elif section == ATTACHMENTS:
            default_data[ATTACHMENTS] = []
        return default_data

    def extract_requesting_party(self, text: str) -> Dict[str, Any]:
        """
        Extracts data from the 'Requesting Party' section.

        Args:
            text (str): Content of the 'Requesting Party' section.

        Returns:
            dict: Extracted 'Requesting Party' data.
        """
        self.logger.debug("Extracting Requesting Party information.")
        data: Dict[str, Any] = {}
        for key, pattern in self.patterns[REQUESTING_PARTY].items():
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                # Handle alternative patterns
                if not value and key in self.additional_patterns.get(
                    REQUESTING_PARTY, {}
                ):
                    alt_pattern = self.additional_patterns[REQUESTING_PARTY][key]
                    alt_match = alt_pattern.search(text)
                    value = alt_match.group(1).strip() if alt_match else "N/A"
                data[key] = value if value else "N/A"
                self.logger.debug(FOUND_MESSAGE, key, value)
            else:
                # Attempt to find using additional patterns if applicable
                if key in self.additional_patterns.get(REQUESTING_PARTY, {}):
                    alt_pattern = self.additional_patterns[REQUESTING_PARTY][key]
                    alt_match = alt_pattern.search(text)
                    value = alt_match.group(1).strip() if alt_match else "N/A"
                    data[key] = value if value else "N/A"
                    if value != "N/A":
                        self.logger.debug(FOUND_ADDITIONAL_PATTERN_MESSAGE, key, value)
                    else:
                        self.logger.debug(NOT_FOUND_MESSAGE, key)
                else:
                    data[key] = "N/A"
                    self.logger.debug(NOT_FOUND_MESSAGE, key)
        return {REQUESTING_PARTY: data}

    def extract_insured_information(self, text: str) -> Dict[str, Any]:
        """
        Extracts data from the 'Insured Information' section.

        Args:
            text (str): Content of the 'Insured Information' section.

        Returns:
            dict: Extracted 'Insured Information' data.
        """
        self.logger.debug("Extracting Insured Information.")
        data: Dict[str, Any] = {}
        for key, pattern in self.patterns[INSURED_INFORMATION].items():
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                if key == OWNER_OR_TENANT:
                    value = (
                        value.capitalize()
                        if value.lower() in ["owner", "tenant"]
                        else "N/A"
                    )
                data[key] = value if value else "N/A"
                self.logger.debug(FOUND_MESSAGE, key, value)
            else:
                # Attempt to find using additional patterns if applicable
                if key in self.additional_patterns.get(INSURED_INFORMATION, {}):
                    alt_pattern = self.additional_patterns[INSURED_INFORMATION][key]
                    alt_match = alt_pattern.search(text)
                    value = alt_match.group(1).strip() if alt_match else "N/A"
                    data[key] = value if value else "N/A"
                    if value != "N/A":
                        self.logger.debug(FOUND_ADDITIONAL_PATTERN_MESSAGE, key, value)
                    else:
                        self.logger.debug(NOT_FOUND_MESSAGE, key)
                else:
                    data[key] = "N/A"
                    self.logger.debug(NOT_FOUND_MESSAGE, key)
        return {INSURED_INFORMATION: data}

    def extract_adjuster_information(self, text: str) -> Dict[str, Any]:
        """
        Extracts data from the 'Adjuster Information' section.

        Args:
            text (str): Content of the 'Adjuster Information' section.

        Returns:
            dict: Extracted 'Adjuster Information' data.
        """
        self.logger.debug("Extracting Adjuster Information.")
        data: Dict[str, Any] = {}
        for key, pattern in self.patterns[ADJUSTER_INFORMATION].items():
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                # Specific handling for phone numbers and emails
                if key == ADJUSTER_PHONE_NUMBER:
                    formatted_phone = self.format_phone_number(value)
                    value = formatted_phone if formatted_phone else "N/A"
                elif key == ADJUSTER_EMAIL:
                    value = value.lower() if value else "N/A"
                data[key] = value if value else "N/A"
                self.logger.debug(FOUND_MESSAGE, key, value)
            else:
                # Attempt to find using additional patterns if applicable
                if key in self.additional_patterns.get(ADJUSTER_INFORMATION, {}):
                    alt_pattern = self.additional_patterns[ADJUSTER_INFORMATION][key]
                    alt_match = alt_pattern.search(text)
                    value = alt_match.group(1).strip() if alt_match else "N/A"
                    if key == ADJUSTER_PHONE_NUMBER and value != "N/A":
                        formatted_phone = self.format_phone_number(value)
                        value = formatted_phone if formatted_phone else "N/A"
                    elif key == ADJUSTER_EMAIL and value != "N/A":
                        value = value.lower()
                    data[key] = value if value else "N/A"
                    if value != "N/A":
                        self.logger.debug(FOUND_ADDITIONAL_PATTERN_MESSAGE, key, value)
                    else:
                        self.logger.debug(NOT_FOUND_MESSAGE, key)
                else:
                    data[key] = "N/A"
                    self.logger.debug(NOT_FOUND_MESSAGE, key)
        return {ADJUSTER_INFORMATION: data}

    def format_phone_number(
        self, phone: str, default_region: str = "US"
    ) -> Optional[str]:
        """
        Formats the phone number to the E.164 standard format.

        Args:
            phone (str): Raw phone number.
            default_region (str, optional): Default region code for parsing. Defaults to "US".

        Returns:
            Optional[str]: Formatted phone number in E.164 format or None if parsing fails.
        """
        self.logger.debug("Attempting to format phone number: '%s'", phone)
        try:
            parsed_number = phonenumbers.parse(phone, default_region)
            if phonenumbers.is_valid_number(parsed_number):
                formatted_number = phonenumbers.format_number(
                    parsed_number, PhoneNumberFormat.E164
                )
                self.logger.debug(
                    "Formatted phone number '%s' as '%s'.", phone, formatted_number
                )
                return formatted_number
            else:
                self.logger.warning("Invalid phone number format: '%s'.", phone)
                return None
        except NumberParseException as e:
            self.logger.warning("Error parsing phone number '%s': %s", phone, e)
            return None

    def extract_assignment_information(self, text: str) -> Dict[str, Any]:
        """
        Extracts data from the 'Assignment Information' section.

        Args:
            text (str): Content of the 'Assignment Information' section.

        Returns:
            dict: Extracted 'Assignment Information' data.
        """
        self.logger.debug("Extracting Assignment Information.")
        data: Dict[str, Any] = {}
        for key, pattern in self.patterns[ASSIGNMENT_INFORMATION].items():
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                # Specific handling for dates and boolean fields
                if key == DATE_OF_LOSS_LABEL:
                    parsed_date = self.parse_date(value)
                    value = parsed_date if parsed_date else "N/A"
                elif key in [RESIDENCE_OCCUPIED_DURING_LOSS, WAS_SOMEONE_HOME]:
                    value = (
                        value.capitalize() if value.lower() in ["yes", "no"] else "N/A"
                    )
                data[key] = value if value else "N/A"
                self.logger.debug(FOUND_MESSAGE, key, value)
            else:
                # Attempt to find using additional patterns if applicable
                if key in self.additional_patterns.get(ASSIGNMENT_INFORMATION, {}):
                    alt_pattern = self.additional_patterns[ASSIGNMENT_INFORMATION][key]
                    alt_match = alt_pattern.search(text)
                    value = alt_match.group(1).strip() if alt_match else "N/A"
                    if value and key == DATE_OF_LOSS_LABEL:
                        parsed_date = self.parse_date(value)
                        value = parsed_date if parsed_date else "N/A"
                    elif key in [RESIDENCE_OCCUPIED_DURING_LOSS, WAS_SOMEONE_HOME]:
                        value = (
                            value.capitalize()
                            if value.lower() in ["yes", "no"]
                            else "N/A"
                        )
                    data[key] = value if value else "N/A"
                    if value != "N/A":
                        self.logger.debug(FOUND_ADDITIONAL_PATTERN_MESSAGE, key, value)
                    else:
                        self.logger.debug(NOT_FOUND_MESSAGE, key)
                else:
                    data[key] = "N/A"
                    self.logger.debug(NOT_FOUND_MESSAGE, key)
        return {ASSIGNMENT_INFORMATION: data}

    def parse_date(self, date_str: str) -> Optional[str]:
        """
        Parses and standardizes date formats to YYYY-MM-DD.

        Args:
            date_str (str): Raw date string.

        Returns:
            Optional[str]: Standardized date in YYYY-MM-DD format or None if parsing fails.
        """
        self.logger.debug("Attempting to parse date: '%s'", date_str)

        # First, try using the predefined date formats
        for fmt in self.date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                standardized_date = date_obj.strftime("%Y-%m-%d")
                self.logger.debug(
                    "Parsed date '%s' as '%s' using format '%s'.",
                    date_str,
                    standardized_date,
                    fmt,
                )
                return standardized_date
            except ValueError:
                continue

        # If predefined formats fail, try using dateutil's parser
        try:
            date_obj = dateutil_parser.parse(date_str, dayfirst=False, fuzzy=True)
            standardized_date = date_obj.strftime("%Y-%m-%d")
            self.logger.debug(
                "Parsed date '%s' as '%s' using dateutil parser.",
                date_str,
                standardized_date,
            )
            return standardized_date
        except ParserError as e:
            self.logger.warning("Unable to parse date '%s': %s", date_str, e)
            return None  # or "Invalid Date" based on requirements

    def extract_assignment_type(self, text: str) -> Dict[str, Any]:
        """
        Extracts the assignment type by checking the corresponding boxes.

        Args:
            text (str): Content of the 'Assignment Type' section.

        Returns:
            dict: Extracted 'Assignment Type' data.
        """
        self.logger.debug("Extracting Assignment Type.")
        data: Dict[str, Any] = {
            "Wind": False,
            "Structural": False,
            "Hail": False,
            "Foundation": False,
            "Other": {"Checked": False, "Details": "N/A"},
        }

        for key, pattern in self.patterns[ASSIGNMENT_TYPE].items():
            match = pattern.search(text)
            if key != "Other":
                if match:
                    data[key] = True
                    self.logger.debug("Assignment Type '%s' checked.", key)
            else:
                if match:
                    data["Other"]["Checked"] = True
                    details = match.group(2).strip() if match.lastindex >= 2 else "N/A"
                    data["Other"]["Details"] = details if details else "N/A"
                    self.logger.debug(
                        "Assignment Type 'Other' checked with details: %s", details
                    )
        return {ASSIGNMENT_TYPE: data}

    def extract_additional_details_special_instructions(
        self, text: str
    ) -> Dict[str, Any]:
        """
        Extracts additional details or special instructions.

        Args:
            text (str): Content of the 'Additional details/Special Instructions' section.

        Returns:
            dict: Extracted additional details.
        """
        self.logger.debug("Extracting Additional Details/Special Instructions.")
        data: Dict[str, Any] = {}
        pattern = self.patterns[ADDITIONAL_DETAILS][ADDITIONAL_DETAILS]
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            data[ADDITIONAL_DETAILS] = value if value else "N/A"
            self.logger.debug(
                "Found Additional details/Special Instructions: %s", value
            )
        else:
            data[ADDITIONAL_DETAILS] = "N/A"
            self.logger.debug(
                "Additional details/Special Instructions not found, set to 'N/A'"
            )
        return data

    def extract_attachments(self, text: str) -> Dict[str, Any]:
        """
        Extracts attachment information.

        Args:
            text (str): Content of the 'Attachment(s)' section.

        Returns:
            dict: Extracted attachment details.
        """
        self.logger.debug("Extracting Attachment(s).")
        data: Dict[str, Any] = {}
        pattern = self.patterns[ATTACHMENTS][ATTACHMENTS]
        match = pattern.search(text)
        if match:
            attachments = match.group(1).strip()
            if attachments.lower() != "n/a" and attachments:
                # Split by multiple delimiters
                attachment_list = re.split(r",|;|\n|•|–|-", attachments)
                # Further filter and validate attachment entries
                attachment_list = [
                    att.strip()
                    for att in attachment_list
                    if att.strip()
                    and (
                        self.is_valid_attachment(att.strip())
                        or self.is_valid_url(att.strip())
                    )
                ]
                data[ATTACHMENTS] = attachment_list if attachment_list else []
                self.logger.debug("Found Attachments: %s", attachment_list)
            else:
                data[ATTACHMENTS] = []
                self.logger.debug("Attachments marked as 'N/A' or empty.")
        else:
            data[ATTACHMENTS] = []
            self.logger.debug("Attachment(s) not found, set to empty list")
        return data

    def is_valid_attachment(self, attachment: str) -> bool:
        """Simple validation for file extensions."""
        valid_extensions = [".pdf", ".docx", ".xlsx", ".zip", ".png", ".jpg"]
        return any(attachment.lower().endswith(ext) for ext in valid_extensions)

    def is_valid_url(self, attachment: str) -> bool:
        """Simple URL validation."""
        url_pattern = re.compile(
            r"^(?:http|ftp)s?://"  # http:// or https://
            r"(?:\S+(?::\S*)?@)?"  # user:pass@
            r"(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])\."  # IP...
            r"(?:1?\d{1,2}|2[0-4]\d|25[0-5])\."
            r"(?:1?\d{1,2}|2[0-4]\d|25[0-5])\."
            r"(?:1?\d{1,2}|2[0-4]\d|25[0-5]))|"  # ...or
            r"(?:(?:[a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))"  # domain...
            r"(?::\d{2,5})?"  # optional port
            r"(?:/\S*)?$",
            re.IGNORECASE,
        )
        return re.match(url_pattern, attachment) is not None

    def extract_entities(self, email_content: str) -> Dict[str, Any]:
        """
        Extracts named entities from the email content using spaCy.

        Args:
            email_content (str): The raw email content.

        Returns:
            dict: Extracted entities categorized by their labels.
        """
        self.logger.debug("Extracting Named Entities using spaCy.")
        try:
            doc = self.nlp(email_content)
        except Exception as e:
            self.logger.error("spaCy failed to process the email content: %s", e)
            return {}

        entities: Dict[str, Any] = {}
        # Fetch relevant labels from configuration or use default
        relevant_labels = set(
            self.config.get(
                "entity_labels", ["PERSON", "ORG", "GPE", "DATE", "PRODUCT"]
            )
        )

        for ent in doc.ents:
            if ent.label_ in relevant_labels:
                entities.setdefault(ent.label_, set()).add(ent.text.strip())

        # Convert sets to sorted lists for consistency
        entities = {label: sorted(list(texts)) for label, texts in entities.items()}

        self.logger.debug("Extracted Entities: %s", entities)
        return entities
