#  src\parsers\hybrid_parser.py

"""Module containing the HybridParser class for parsing emails using a hybrid approach."""

import logging
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

import yaml
import spacy
from spacy.cli import download as spacy_download
from spacy.util import is_package
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from thefuzz import fuzz  

from src.parsers.base_parser import BaseParser
from src.parsers.rule_based_parser import RuleBasedParser
from src.utils.validation import validate_json

from urllib.parse import urlparse

# Constants for section headers and field names
SECTION_REQUESTING_PARTY = "Requesting Party"
SECTION_INSURED_INFORMATION = "Insured Information"
SECTION_ADJUSTER_INFORMATION = "Adjuster Information"
SECTION_ASSIGNMENT_INFORMATION = "Assignment Information"
SECTION_ADDITIONAL_DETAILS = "Additional details/Special Instructions"
SECTION_ATTACHMENTS = "Attachment(s)"

FIELD_INSURANCE_COMPANY = "Insurance Company"
FIELD_HANDLER = "Handler"
FIELD_CARRIER_CLAIM_NUMBER = "Carrier Claim Number"
FIELD_NAME = "Name"
FIELD_CONTACT_NUMBER = "Contact #"
FIELD_LOSS_ADDRESS = "Loss Address"
FIELD_PUBLIC_ADJUSTER = "Public Adjuster"
FIELD_OWNER_OR_TENANT = "Owner or Tenant"
FIELD_ADJUSTER_NAME = "Adjuster Name"
FIELD_ADJUSTER_PHONE_NUMBER = "Adjuster Phone Number"
FIELD_ADJUSTER_EMAIL = "Adjuster Email"
FIELD_JOB_TITLE = "Job Title"
FIELD_ADDRESS = "Address"
FIELD_POLICY_NUMBER = "Policy Number"
FIELD_DATE_OF_LOSS = "Date of Loss/Occurrence"
FIELD_CAUSE_OF_LOSS = "Cause of Loss"
FIELD_FACTS_OF_LOSS = "Facts of Loss"
FIELD_LOSS_DESCRIPTION = "Loss Description"
FIELD_RESIDENCE_OCCUPIED = "Residence Occupied During Loss"
FIELD_SOMEONE_HOME = "Was Someone Home at Time of Damage"
FIELD_REPAIR_PROGRESS = "Repair or Mitigation Progress"
FIELD_TYPE = "Type"
FIELD_INSPECTION_TYPE = "Inspection Type"
FIELD_WIND = "Wind"
FIELD_STRUCTURAL = "Structural"
FIELD_HAIL = "Hail"
FIELD_FOUNDATION = "Foundation"
FIELD_OTHER = "Other"
FIELD_ADDITIONAL_DETAILS = "Additional Details/Special Instructions"
FIELD_ATTACHMENTS = "Attachments"

# Constants for logging messages
LOG_FOUND = "Found %s: %s"
LOG_FOUND_ADDITIONAL = "Found %s using additional pattern: %s"
LOG_NOT_FOUND = "%s not found, set to 'N/A'"
LOG_UNEXPECTED_PHONE_FORMAT = "Unexpected phone number format: %s"
LOG_PARSED_DATE = "Parsed date '%s' as '%s' using format '%s'."
LOG_FAILED_DATE_PARSE = "Unable to parse date: %s"
LOG_ENTITIES = "Extracted Entities: %s"
LOG_TRANSFORMER_ENTITIES = "Transformer Extracted Entities: %s"
LOG_FUZZY_MATCHED = "Fuzzy matched field '%s' to '%s'."
LOG_APPLIED_RULE = "Applied rule on field '%s'. New value: %s"
LOG_FAILED_PARSE = "Failed to parse email content: %s"
LOG_FAILED_LOAD_MODEL = "Failed to load spaCy model: %s"
LOG_FAILED_TRANSFORMER = "Failed to initialize transformer model: %s"
LOG_FAILED_EXTRACT_ENTITIES_SPACY = "Failed to extract entities using spaCy: %s"
LOG_FAILED_EXTRACT_ENTITIES_TRANSFORMER = (
    "Failed to extract entities using Transformer model: %s"
)
LOG_SECTION_NOT_FOUND = "Section '%s' not found in email content."
LOG_FALLBACK = "Falling back to RuleBasedParser for parsing."
LOG_SECTION_HEADER = "Detected section header: %s"
LOG_LOADING_CONFIG = "Loaded parser configuration from %s."
LOG_LOADING_DEFAULT_CONFIG = "Loaded default parser configuration."
LOG_SPACY_MODEL_LOADED = "spaCy model '%s' loaded successfully."
LOG_SPACY_MODEL_DOWNLOADED = "Successfully downloaded spaCy model '%s'."
LOG_SPACY_MODEL_DOWNLOAD_FAILED = "Failed to download spaCy model '%s': %s"
LOG_EXTRACTING_SECTIONS = "Splitting email content into sections."
LOG_SECTION_FOUND = "Sections Found: %s"
LOG_EXTRACTING_REQUESTING_PARTY = "Extracting Requesting Party information."
LOG_EXTRACTING_INSURED_INFORMATION = "Extracting Insured Information."
LOG_EXTRACTING_ADJUSTER_INFORMATION = "Extracting Adjuster Information."
LOG_EXTRACTING_ASSIGNMENT_INFORMATION = "Extracting Assignment Information."
LOG_EXTRACTING_ADDITIONAL_DETAILS = (
    "Extracting Additional Details/Special Instructions."
)
LOG_EXTRACTING_ATTACHMENTS = "Extracting Attachments."
LOG_FAILED_FUZZY_MATCHING = "Failed during fuzzy matching: %s"
LOG_FAILED_POST_PROCESSING = "Failed during post-processing rules: %s"
LOG_VALIDATION_ERROR = "JSON Schema Validation Error: %s"
LOG_PARSER_SUCCESS = "Successfully parsed email with HybridParser."
LOG_PARSER_DEBUG = "Extracted Data: %s"
LOG_MODEL_DOWNLOAD_START = "Starting download of %s model..."
LOG_MODEL_DOWNLOAD_COMPLETE = "Successfully downloaded %s model."
LOG_MODEL_DOWNLOAD_FAILED = "Failed to download %s model: %s"
LOG_STARTUP_CHECK_START = "Starting model availability check..."
LOG_STARTUP_CHECK_COMPLETE = "All required models are available."
LOG_STARTUP_CHECK_FAILED = "Some required models are not available: %s"


class HybridParser(BaseParser):
    """Hybrid parser for comprehensive email parsing."""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self.load_config(config_path)
        self.nlp = None
        self.tokenizer = None
        self.transformer_model = None
        self.transformer = None
        self.initialize_models()
        self.compile_patterns()

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use default."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)
                self.logger.info(LOG_LOADING_CONFIG, config_path)
            except (IOError, yaml.YAMLError) as e:
                self.logger.error("Error loading config file '%s': %s", config_path, e)
                raise
        else:
            config = self.default_config()
            self.logger.info(LOG_LOADING_DEFAULT_CONFIG)
        return config

    def initialize_models(self):
        """Initialize and download required models."""
        self.ensure_spacy_model()
        self.ensure_transformer_model()
        self.startup_check()

    def ensure_spacy_model(self):
        """Ensure spaCy model is downloaded and loaded."""
        model_name = "en_core_web_sm"
        self.logger.info(LOG_MODEL_DOWNLOAD_START, "spaCy")
        if not is_package(model_name):
            try:
                spacy_download(model_name)
                self.logger.info(LOG_SPACY_MODEL_DOWNLOADED, model_name)
            except Exception as e:
                self.logger.error(LOG_SPACY_MODEL_DOWNLOAD_FAILED, model_name, str(e))
                raise
        try:
            self.nlp = spacy.load(model_name)
            self.logger.info(LOG_SPACY_MODEL_LOADED, model_name)
        except OSError as e:
            self.logger.error(LOG_FAILED_LOAD_MODEL, str(e))
            raise

    def ensure_transformer_model(self):
        """Ensure transformer model is downloaded and loaded."""
        model_name = "dslim/bert-base-NER"  # Using a more suitable NER model
        self.logger.info(LOG_MODEL_DOWNLOAD_START, "Transformer")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForTokenClassification.from_pretrained(
                model_name
            )
            self.transformer = pipeline(
                "ner",
                model=self.transformer_model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
            )
            self.logger.info(LOG_MODEL_DOWNLOAD_COMPLETE, "Transformer")
        except Exception as e:
            self.logger.error(LOG_FAILED_TRANSFORMER, "Transformer", str(e))
            raise

    def startup_check(self):
        """Verify all required models are available."""
        self.logger.info(LOG_STARTUP_CHECK_START)
        missing_models = []
        if not self.nlp:
            missing_models.append("spaCy")
        if not self.transformer:
            missing_models.append("Transformer")
        if missing_models:
            self.logger.error(LOG_STARTUP_CHECK_FAILED, ", ".join(missing_models))
            raise RuntimeError(f"Missing required models: {', '.join(missing_models)}")
        self.logger.info(LOG_STARTUP_CHECK_COMPLETE)

    def compile_patterns(self):
        """Compile regex patterns for performance."""
        self.section_headers = self.config["section_headers"]
        self.section_pattern = re.compile(
            rf'(?i)^\s*(?:\*+\s*)?({"|".join(map(re.escape, self.section_headers))})(?:\s*\*+)?:?\s*$'
        )
        self.patterns = {}
        self.additional_patterns = {}
        for section, fields in self.config["patterns"].items():
            self.patterns[section] = {
                field: re.compile(pattern, re.IGNORECASE | re.DOTALL)
                for field, pattern in fields.items()
            }
        for section, fields in self.config.get("additional_patterns", {}).items():
            self.additional_patterns[section] = {
                field: re.compile(pattern, re.IGNORECASE | re.DOTALL)
                for field, pattern in fields.items()
            }

    def default_config(self) -> Dict[str, Any]:
        """Provide default configuration for the parser."""
        return {
            "section_headers": [
                SECTION_REQUESTING_PARTY,
                SECTION_INSURED_INFORMATION,
                SECTION_ADJUSTER_INFORMATION,
                SECTION_ASSIGNMENT_INFORMATION,
                SECTION_ADDITIONAL_DETAILS,
                SECTION_ATTACHMENTS,
            ],
            "patterns": {
                SECTION_REQUESTING_PARTY: {
                    FIELD_INSURANCE_COMPANY: r"(?i)insurance\s*company\s*:?\s*(.*)",
                    FIELD_HANDLER: r"(?i)handler\s*:?\s*(.*)",
                    FIELD_CARRIER_CLAIM_NUMBER: r"(?i)(?:carrier\s*)?claim\s*number\s*:?\s*(.*)",
                },
                SECTION_INSURED_INFORMATION: {
                    FIELD_NAME: r"(?i)name\s*:?\s*(.*)",
                    FIELD_CONTACT_NUMBER: r"(?i)contact(?:\s*#)?\s*:?\s*(.*)",
                    FIELD_LOSS_ADDRESS: r"(?i)loss\s*address\s*:?\s*(.*)",
                    FIELD_PUBLIC_ADJUSTER: r"(?i)public\s*adjuster\s*:?\s*(.*)",
                    FIELD_OWNER_OR_TENANT: r"(?i)(?:is\s*the\s*insured\s*an?\s*)?owner\s*or\s*tenant.*?:?\s*(.*)",
                },
                SECTION_ADJUSTER_INFORMATION: {
                    FIELD_ADJUSTER_NAME: r"(?i)adjuster\s*name\s*:?\s*(.*)",
                    FIELD_ADJUSTER_PHONE_NUMBER: r"(?i)adjuster\s*phone\s*number\s*:?\s*(\+?[\d\s\-().]{7,})",
                    FIELD_ADJUSTER_EMAIL: r"(?i)adjuster\s*email\s*:?\s*([\w\.-]+@[\w\.-]+\.\w+)",
                    FIELD_JOB_TITLE: r"(?i)job\s*title\s*:?\s*(.*)",
                    FIELD_ADDRESS: r"(?i)address\s*:?\s*(.*)",
                    FIELD_POLICY_NUMBER: r"(?i)policy(?:\s*#)?\s*:?\s*(\w+)",
                },
                SECTION_ASSIGNMENT_INFORMATION: {
                    FIELD_DATE_OF_LOSS: r"(?i)date\s*of\s*loss(?:/occurrence)?\s*:?\s*(.*)",
                    FIELD_CAUSE_OF_LOSS: r"(?i)cause\s*of\s*loss\s*:?\s*(.*)",
                    FIELD_FACTS_OF_LOSS: r"(?i)facts\s*of\s*loss\s*:?\s*(.*)",
                    FIELD_LOSS_DESCRIPTION: r"(?i)loss\s*description\s*:?\s*(.*)",
                    FIELD_RESIDENCE_OCCUPIED: r"(?i)residence\s*occupied\s*during\s*loss\s*:?\s*(.*)",
                    FIELD_SOMEONE_HOME: r"(?i)was\s*someone\s*home\s*at\s*time\s*of\s*damage\s*:?\s*(.*)",
                    FIELD_REPAIR_PROGRESS: r"(?i)repair(?:\s*or\s*mitigation)?\s*progress\s*:?\s*(.*)",
                    FIELD_TYPE: r"(?i)type\s*:?\s*(.*)",
                    FIELD_INSPECTION_TYPE: r"(?i)inspection\s*type\s*:?\s*(.*)",
                    FIELD_WIND: r"(?i)wind\s*\[(.*?)\]",
                    FIELD_STRUCTURAL: r"(?i)structural\s*\[(.*?)\]",
                    FIELD_HAIL: r"(?i)hail\s*\[(.*?)\]",
                    FIELD_FOUNDATION: r"(?i)foundation\s*\[(.*?)\]",
                    FIELD_OTHER: r"(?i)other\s*\[(.*?)\].*?(?:details?)?\s*:?\s*(.*)",
                },
                SECTION_ADDITIONAL_DETAILS: {
                    FIELD_ADDITIONAL_DETAILS: r"(?i)additional\s*details?(?:/special\s*instructions?)?\s*:?\s*(.*)"
                },
                SECTION_ATTACHMENTS: {FIELD_ATTACHMENTS: r"(?i)attachment(?:s)?\s*:?\s*(.*)"},
            },
            "additional_patterns": {
                SECTION_REQUESTING_PARTY: {
                    FIELD_POLICY_NUMBER: r"(?i)policy(?:\s*#)?\s*:?\s*(\w+)",
                    FIELD_CARRIER_CLAIM_NUMBER: r"(?i)(?:carrier\s*)?claim\s*number\s*:?\s*(.*)",
                },
                SECTION_ASSIGNMENT_INFORMATION: {
                    FIELD_DATE_OF_LOSS: r"(?i)date\s*of\s*loss(?:/occurrence)?\s*:?\s*(.*)"
                },
            },
            "date_formats": [
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%Y-%m-%d",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%Y.%m.%d",
                "%d.%m.%Y",
                "%m-%d-%Y",
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
            "fuzzy_match_fields": [
                FIELD_INSURANCE_COMPANY,
                FIELD_HANDLER,
                FIELD_ADJUSTER_NAME,
                FIELD_POLICY_NUMBER,
            ],
            "known_values": {
                FIELD_INSURANCE_COMPANY: [
                    "State Farm",
                    "Allstate",
                    "Geico",
                    "Progressive",
                    "Nationwide",
                    "Liberty Mutual",
                    "Farmers",
                    "Travelers",
                    "American Family",
                    "USAA",
                ],
                FIELD_HANDLER: [
                    "John Doe",
                    "Jane Smith",
                    "Emily Davis",
                    "Michael Brown",
                    "Sarah Johnson",
                    "David Wilson",
                ],
                FIELD_ADJUSTER_NAME: [
                    "Michael Brown",
                    "Sarah Johnson",
                    "David Wilson",
                    "Laura Martinez",
                    "James Anderson",
                ],
                FIELD_POLICY_NUMBER: [
                    "ABC123",
                    "XYZ789",
                    "DEF456",
                    "GHI101",
                    "JKL202",
                ],
            },
            "post_processing_rules": [
                {
                    "field": FIELD_ADJUSTER_EMAIL,
                    "condition_field": FIELD_ADJUSTER_EMAIL,
                    "condition_value": "N/A",
                    "action_value": "unknown@example.com",
                },
            ],
        }

    def parse(self, email_content: str) -> Dict[str, Any]:
        """
        Parse the email content using various techniques to extract key information.

        Args:
            email_content (str): The raw email content to parse.

        Returns:
            dict: Parsed data as a dictionary.
        """
        self.logger.info("Parsing email content with HybridParser.")
        extracted_data = {}
        try:
            sections = self.split_into_sections(email_content)
            for section, content in sections.items():
                extract_method = getattr(
                    self, f"extract_{self.snake_case(section)}", None
                )
                if extract_method:
                    try:
                        data = extract_method(content)
                        extracted_data.update(data)
                    except (ValueError, RuntimeError) as e:
                        self.logger.error(LOG_FAILED_PARSE, e)
                        extracted_data.update(self.default_section_data())
                else:
                    self.logger.warning(
                        "No extraction method found for section: %s", section
                    )
                    extracted_data.update(self.default_section_data())

            # Ensure all required fields are present
            for field in self.get_all_fields():
                if field not in extracted_data:
                    self.logger.warning(LOG_NOT_FOUND, field)
                    extracted_data[field] = "N/A"

            # Extract entities using spaCy
            entities = self.extract_entities(email_content)
            extracted_data["Entities"] = entities

            # Extract entities using Transformer model and merge
            transformer_entities = self.transformer_extraction(email_content)
            extracted_data["TransformerEntities"] = transformer_entities.get(
                "TransformerEntities", {}
            )

            # Apply fuzzy matching to improve data accuracy
            extracted_data = self.fuzzy_match_fields(email_content, extracted_data)

            # Apply post-processing rules
            extracted_data = self.apply_rules(extracted_data)

            # Validate the parsed data against the JSON schema
            is_valid, error_message = validate_json(extracted_data)
            if not is_valid:
                self.logger.error(LOG_VALIDATION_ERROR, error_message)
                raise ValueError(LOG_VALIDATION_ERROR % error_message)

            self.logger.debug(LOG_PARSER_DEBUG, extracted_data)
            self.logger.info(LOG_PARSER_SUCCESS)
            return extracted_data

        except (IOError, yaml.YAMLError, ValueError, RuntimeError) as e:
            self.logger.error(LOG_FAILED_PARSE, e)
            extracted_data = self.fallback_to_rule_based_parser(email_content)
            return extracted_data

    def snake_case(self, text: str) -> str:
        """Convert text to snake_case, replacing special characters with underscores."""
        text = re.sub(r'[\/\(\)]', '_', text)  # Replace /, (, ) with _
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", "_", text.strip().lower())
        text = re.sub(r'__+', '_', text)  # Replace multiple underscores with single
        return text.strip('_')

    def split_into_sections(self, email_content: str) -> Dict[str, str]:
        """Split the email content into sections based on headers."""
        self.logger.debug(LOG_EXTRACTING_SECTIONS)
        sections = {}
        current_section = None
        content_buffer = []

        for line in email_content.splitlines():
            line = line.strip()
            if not line:
                continue

            header_match = self.section_pattern.match(line)
            if header_match:
                if current_section:
                    sections[current_section] = "\n".join(content_buffer).strip()
                    content_buffer = []
                current_section = header_match.group(1)
                self.logger.debug(LOG_SECTION_HEADER, current_section)
            elif current_section:
                content_buffer.append(line)

        if current_section and content_buffer:
            sections[current_section] = "\n".join(content_buffer).strip()

        self.logger.debug(LOG_SECTION_FOUND, list(sections.keys()))
        return sections

    def get_all_fields(self) -> List[str]:
        """Retrieve a list of all field names from the configuration."""
        fields = []
        for section in self.config["patterns"]:
            fields.extend(self.config["patterns"][section].keys())
        for section in self.config.get("additional_patterns", {}):
            fields.extend(self.config["additional_patterns"][section].keys())
        return list(set(fields))

    def default_section_data(self) -> Dict[str, Any]:
        """Provide default data structure with flat fields."""
        default_data = {
            FIELD_INSURANCE_COMPANY: "N/A",
            FIELD_HANDLER: "N/A",
            FIELD_CARRIER_CLAIM_NUMBER: "N/A",
            FIELD_NAME: "N/A",
            FIELD_CONTACT_NUMBER: "N/A",
            FIELD_LOSS_ADDRESS: "N/A",
            FIELD_PUBLIC_ADJUSTER: "N/A",
            FIELD_OWNER_OR_TENANT: "N/A",
            FIELD_ADJUSTER_NAME: "N/A",
            FIELD_ADJUSTER_PHONE_NUMBER: "N/A",
            FIELD_ADJUSTER_EMAIL: "N/A",
            FIELD_JOB_TITLE: "N/A",
            FIELD_ADDRESS: "N/A",
            FIELD_POLICY_NUMBER: "N/A",
            FIELD_DATE_OF_LOSS: "N/A",
            FIELD_CAUSE_OF_LOSS: "N/A",
            FIELD_FACTS_OF_LOSS: "N/A",
            FIELD_LOSS_DESCRIPTION: "N/A",
            FIELD_RESIDENCE_OCCUPIED: False,
            FIELD_SOMEONE_HOME: False,
            FIELD_REPAIR_PROGRESS: "N/A",
            FIELD_TYPE: "N/A",
            FIELD_INSPECTION_TYPE: "N/A",
            FIELD_WIND: False,
            FIELD_STRUCTURAL: False,
            FIELD_HAIL: False,
            FIELD_FOUNDATION: False,
            FIELD_OTHER: {"Checked": False, "Details": "N/A"},
            FIELD_ADDITIONAL_DETAILS: "N/A",
            FIELD_ATTACHMENTS: [],
        }
        return default_data

    def extract_requesting_party(self, text: str) -> Dict[str, Any]:
        """Extract data from the 'Requesting Party' section."""
        self.logger.debug(LOG_EXTRACTING_REQUESTING_PARTY)
        return self._extract_section_data(SECTION_REQUESTING_PARTY, text)

    def extract_insured_information(self, text: str) -> Dict[str, Any]:
        """Extract data from the 'Insured Information' section."""
        self.logger.debug(LOG_EXTRACTING_INSURED_INFORMATION)
        return self._extract_section_data(SECTION_INSURED_INFORMATION, text)

    def extract_adjuster_information(self, text: str) -> Dict[str, Any]:
        """Extract data from the 'Adjuster Information' section."""
        self.logger.debug(LOG_EXTRACTING_ADJUSTER_INFORMATION)
        return self._extract_section_data(SECTION_ADJUSTER_INFORMATION, text)

    def extract_assignment_information(self, text: str) -> Dict[str, Any]:
        """Extract data from the 'Assignment Information' section, including Assignment Type."""
        self.logger.debug(LOG_EXTRACTING_ASSIGNMENT_INFORMATION)
        data = self._extract_section_data(SECTION_ASSIGNMENT_INFORMATION, text)

        # Handle Assignment Type within Assignment Information
        assignment_type = {
            FIELD_WIND: False,
            FIELD_STRUCTURAL: False,
            FIELD_HAIL: False,
            FIELD_FOUNDATION: False,
            FIELD_OTHER: {"Checked": False, "Details": "N/A"},
        }

        for key in [FIELD_WIND, FIELD_STRUCTURAL, FIELD_HAIL, FIELD_FOUNDATION, FIELD_OTHER]:
            pattern = self.patterns[SECTION_ASSIGNMENT_INFORMATION].get(key)
            if pattern:
                match = pattern.search(text)
                if match:
                    if key != FIELD_OTHER:
                        assignment_type[key] = bool(match.group(1).strip().lower() == 'x')
                        self.logger.debug(LOG_FOUND, f"Assignment Type '{key}'", "Checked" if assignment_type[key] else "Unchecked")
                    else:
                        assignment_type[key]["Checked"] = bool(match.group(1).strip().lower() == 'x')
                        details = match.group(2).strip() if match.lastindex >= 2 else "N/A"
                        assignment_type[key]["Details"] = details if details else "N/A"
                        self.logger.debug(LOG_FOUND_ADDITIONAL, f"Assignment Type '{key}'", details)
                else:
                    if key != FIELD_OTHER:
                        assignment_type[key] = False
                        self.logger.debug(LOG_FOUND, f"Assignment Type '{key}'", "Unchecked")
                    else:
                        assignment_type[key]["Checked"] = False
                        assignment_type[key]["Details"] = "N/A"
                        self.logger.debug(LOG_FOUND_ADDITIONAL, f"Assignment Type '{key}'", "N/A")
            else:
                if key != FIELD_OTHER:
                    assignment_type[key] = False
                else:
                    assignment_type[key]["Checked"] = False
                    assignment_type[key]["Details"] = "N/A"

        # Flatten assignment_type into data
        for key, value in assignment_type.items():
            if isinstance(value, dict):
                data[f"{key}_Checked"] = value["Checked"]
                data[f"{key}_Details"] = value["Details"]
            else:
                data[key] = value

        return data

    def extract_additional_details_special_instructions(
        self, text: str
    ) -> Dict[str, Any]:
        """Extract additional details or special instructions."""
        self.logger.debug(LOG_EXTRACTING_ADDITIONAL_DETAILS)
        pattern = self.patterns.get(SECTION_ADDITIONAL_DETAILS, {}).get(
            FIELD_ADDITIONAL_DETAILS
        )
        if pattern:
            match = pattern.search(text)
            value = match.group(1).strip() if match else "N/A"
            self.logger.debug(
                LOG_FOUND if value != "N/A" else LOG_NOT_FOUND,
                FIELD_ADDITIONAL_DETAILS,
                value,
            )
            return {FIELD_ADDITIONAL_DETAILS: value}
        self.logger.debug(LOG_NOT_FOUND, FIELD_ADDITIONAL_DETAILS)
        return {FIELD_ADDITIONAL_DETAILS: "N/A"}

    def extract_attachments(self, text: str) -> Dict[str, Any]:
        """Extract attachment information."""
        self.logger.debug(LOG_EXTRACTING_ATTACHMENTS)
        pattern = self.patterns.get(SECTION_ATTACHMENTS, {}).get(FIELD_ATTACHMENTS)
        if pattern:
            match = pattern.search(text)
            if match:
                attachments = match.group(1).strip()
                if attachments.lower() != "n/a" and attachments:
                    attachment_list = [
                        att.strip()
                        for att in re.split(r"[,;\n•–-]", attachments)
                        if att.strip()
                        and (
                            self.is_valid_attachment(att.strip())
                            or self.is_valid_url(att.strip())
                        )
                    ]
                    self.logger.debug(LOG_FOUND, FIELD_ATTACHMENTS, attachment_list)
                else:
                    attachment_list = []
                    self.logger.debug("Attachments marked as 'N/A' or empty.")
            else:
                attachment_list = []
                self.logger.debug("Attachment(s) not found, set to empty list")
            return {FIELD_ATTACHMENTS: attachment_list}
        self.logger.debug(LOG_NOT_FOUND, FIELD_ATTACHMENTS)
        return {FIELD_ATTACHMENTS: []}

    def is_valid_attachment(self, attachment: str) -> bool:
        """Validate file extensions."""
        valid_extensions = [
            ".pdf",
            ".docx",
            ".xlsx",
            ".zip",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
        ]
        return any(attachment.lower().endswith(ext) for ext in valid_extensions)

    def is_valid_url(self, attachment: str) -> bool:
        """Validate URLs using urllib.parse."""
        try:
            result = urlparse(attachment)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def extract_entities(self, email_content: str) -> Dict[str, List[str]]:
        """Extract named entities from the email content using spaCy."""
        self.logger.debug("Extracting Named Entities using spaCy.")
        try:
            doc = self.nlp(email_content)
            entities = {}
            relevant_labels = {"PERSON", "ORG", "GPE", "DATE", "PRODUCT"}
            for ent in doc.ents:
                if ent.label_ in relevant_labels:
                    entities.setdefault(ent.label_, []).append(ent.text)
            # Remove duplicates
            for label in entities:
                entities[label] = list(set(entities[label]))
            self.logger.debug(LOG_ENTITIES, entities)
            return entities
        except (ValueError, RuntimeError, AttributeError) as e:
            self.logger.error(LOG_FAILED_EXTRACT_ENTITIES_SPACY, e)
            return {}
        except Exception as e:
            self.logger.error(LOG_FAILED_EXTRACT_ENTITIES_SPACY, e)
            return {}

    def transformer_extraction(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        """Extract named entities using the transformer-based model."""
        self.logger.debug("Extracting Named Entities using Transformer model.")
        try:
            transformer_entities = self.transformer(text)
            entities = {}
            for entity in transformer_entities:
                label = entity.get("entity_group") or entity.get("entity")
                word = entity.get("word")
                if not label or not word:
                    continue  # Skip if no label or word is found
                entities.setdefault(label, []).append(word)
            # Remove duplicates
            for label in entities:
                entities[label] = list(set(entities[label]))
            self.logger.debug(LOG_TRANSFORMER_ENTITIES, entities)
            return {"TransformerEntities": entities}
        except (ValueError, RuntimeError) as e:
            self.logger.error(LOG_FAILED_EXTRACT_ENTITIES_TRANSFORMER, e)
            return {"TransformerEntities": {}}
        except Exception as e:
            self.logger.error(LOG_FAILED_EXTRACT_ENTITIES_TRANSFORMER, e)
            return {"TransformerEntities": {}}

    def fuzzy_match_fields(
        self, text: str, parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply fuzzy matching to specified fields to improve data accuracy."""
        self.logger.debug("Applying fuzzy matching to specified fields.")
        try:
            fuzzy_fields = self.config.get("fuzzy_match_fields", [])
            for field in fuzzy_fields:
                if parsed_data.get(field) == "N/A":
                    best_match = max(
                        self.config.get("known_values", {}).get(field, []),
                        key=lambda x: fuzz.partial_ratio(x.lower(), text.lower()),
                        default=None,
                    )
                    if (
                        best_match
                        and fuzz.partial_ratio(best_match.lower(), text.lower()) > 80
                    ):
                        parsed_data[field] = best_match
                        self.logger.debug(LOG_FUZZY_MATCHED, field, best_match)
            return parsed_data
        except (ValueError, RuntimeError) as e:
            self.logger.error(LOG_FAILED_FUZZY_MATCHING, e)
            return parsed_data
        except Exception as e:
            self.logger.error(LOG_FAILED_FUZZY_MATCHING, e)
            return parsed_data

    def apply_rules(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule-based post-processing to the parsed data."""
        self.logger.debug("Applying post-processing rules.")
        try:
            for rule in self.config.get("post_processing_rules", []):
                field = rule["field"]
                condition_field = rule["condition_field"]
                condition_value = rule["condition_value"]
                action_value = rule["action_value"]
                if parsed_data.get(condition_field) == condition_value:
                    parsed_data[field] = action_value
                    self.logger.debug(LOG_APPLIED_RULE, field, parsed_data[field])
            return parsed_data
        except (KeyError, TypeError) as e:
            self.logger.error(LOG_FAILED_POST_PROCESSING, e)
            return parsed_data
        except Exception as e:
            self.logger.error(LOG_FAILED_POST_PROCESSING, e)
            return parsed_data

    def format_phone_number(self, phone: str) -> str:
        """Format the phone number to a standard format."""
        digits = re.sub(r"\D", "", phone)
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits.startswith("1"):
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            self.logger.warning(LOG_UNEXPECTED_PHONE_FORMAT, phone)
            return phone

    def parse_date(self, date_str: str) -> str:
        """Parse and standardize date formats using predefined formats."""
        for fmt in self.config.get("date_formats", []):
            try:
                date_obj = datetime.strptime(date_str, fmt)
                standardized_date = date_obj.strftime("%Y-%m-%d")
                self.logger.debug(LOG_PARSED_DATE, date_str, standardized_date, fmt)
                return standardized_date
            except ValueError:
                continue
        # Fallback to dateutil_parse
        try:
            date_obj = datetime.fromisoformat(date_str)
            standardized_date = date_obj.strftime("%Y-%m-%d")
            self.logger.debug(LOG_PARSED_DATE, date_str, standardized_date, "dateutil")
            return standardized_date
        except ValueError:
            self.logger.warning(LOG_FAILED_DATE_PARSE, date_str)
            return date_str

    def parse_boolean(self, value: str) -> Optional[bool]:
        """Parse boolean values."""
        value = value.lower()
        if value in self.config.get("boolean_values", {}).get("positive", []):
            return True
        elif value in self.config.get("boolean_values", {}).get("negative", []):
            return False
        else:
            self.logger.warning("Unknown boolean value '%s'. Setting to None.", value)
            return None

    def fallback_to_rule_based_parser(self, email_content: str) -> Dict[str, Any]:
        """Fallback mechanism to use RuleBasedParser if hybrid parsing fails."""
        self.logger.info(LOG_FALLBACK)
        try:
            rule_based_parser = RuleBasedParser(config=self.config)  # Pass config if needed
            return rule_based_parser.parse(email_content)
        except Exception as e:
            self.logger.error("Fallback to RuleBasedParser failed: %s", e)
            return self.default_section_data()

    def _extract_section_data(self, section: str, text: str) -> Dict[str, Any]:
        """Generic method to extract data from a section using patterns."""
        data = {}
        for key, pattern in self.patterns.get(section, {}).items():
            match = pattern.search(text)
            if match:
                if key == FIELD_DATE_OF_LOSS:
                    value = self.parse_date(match.group(1).strip())
                elif key in [
                    FIELD_RESIDENCE_OCCUPIED,
                    FIELD_SOMEONE_HOME,
                ]:
                    value = self.parse_boolean(match.group(1).strip())
                elif key == FIELD_OWNER_OR_TENANT:
                    value = match.group(1).strip()
                elif key == FIELD_ADJUSTER_PHONE_NUMBER:
                    value = self.format_phone_number(match.group(1).strip())
                elif key == FIELD_ADJUSTER_EMAIL:
                    value = match.group(1).strip().lower()
                elif key == FIELD_OTHER:
                    checked = bool(match.group(1).strip().lower() == 'x')
                    details = match.group(2).strip() if match.lastindex >= 2 else "N/A"
                    value = {"Checked": checked, "Details": details if details else "N/A"}
                else:
                    value = match.group(1).strip()
                if key == FIELD_OTHER:
                    data[f"{key}_Checked"] = value["Checked"]
                    data[f"{key}_Details"] = value["Details"]
                else:
                    data[key] = value if value else "N/A"
                self.logger.debug(LOG_FOUND, key, value)
            else:
                # Handle alternative patterns
                alt_pattern = self.additional_patterns.get(section, {}).get(key)
                if alt_pattern:
                    alt_match = alt_pattern.search(text)
                    value = alt_match.group(1).strip() if alt_match else "N/A"
                    if key == FIELD_DATE_OF_LOSS and value != "N/A":
                        value = self.parse_date(value)
                    elif key == FIELD_OWNER_OR_TENANT and value != "N/A":
                        value = value.strip()
                    elif key in [
                        FIELD_RESIDENCE_OCCUPIED,
                        FIELD_SOMEONE_HOME,
                    ] and value != "N/A":
                        value = self.parse_boolean(value)
                    elif key == FIELD_ADJUSTER_PHONE_NUMBER and value != "N/A":
                        value = self.format_phone_number(value)
                    elif key == FIELD_ADJUSTER_EMAIL and value != "N/A":
                        value = value.lower()
                    elif key == FIELD_OTHER and value != "N/A":
                        parts = value.split(None, 1)
                        checked = bool(parts[0].strip().lower() == 'x') if parts else False
                        details = parts[1].strip() if len(parts) > 1 else "N/A"
                        value = {"Checked": checked, "Details": details if details else "N/A"}
                    else:
                        value = value if value else "N/A"

                    if key == FIELD_OTHER and value != "N/A":
                        data[f"{key}_Checked"] = value["Checked"]
                        data[f"{key}_Details"] = value["Details"]
                    else:
                        data[key] = value

                    if value != "N/A":
                        self.logger.debug(LOG_FOUND_ADDITIONAL, key, value)
                    else:
                        self.logger.debug(LOG_NOT_FOUND, key)
                else:
                    if key == FIELD_OTHER:
                        data[f"{key}_Checked"] = False
                        data[f"{key}_Details"] = "N/A"
                    else:
                        data[key] = "N/A"
                    self.logger.debug(LOG_NOT_FOUND, key)
        return data
