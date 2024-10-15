# src/utils/validation.py

import jsonschema
from jsonschema import Draft7Validator
import logging
from src.utils.config_loader import ConfigLoader

# Load configuration
config = ConfigLoader.load_config()

# Define the QuickBase schema based on parser_config.yaml
assignment_schema = {
    "type": "object",
    "properties": {
        "Requesting Party": {
            "type": "object",
            "properties": {
                "Insurance Company": {"type": ["string", "null"]},
                "Handler": {"type": ["string", "null"]},
                "Carrier Claim Number": {"type": ["string", "null"]},
            },
            "required": ["Insurance Company", "Handler", "Carrier Claim Number"],
        },
        "Insured Information": {
            "type": "object",
            "properties": {
                "Name": {"type": ["string", "null"]},
                "Contact #": {"type": ["string", "null"]},
                "Loss Address": {"type": ["string", "null"]},
                "Public Adjuster": {"type": ["string", "null"]},
                "Owner or Tenant": {"type": ["string", "null"]},
            },
            "required": ["Name", "Contact #", "Loss Address", "Public Adjuster", "Owner or Tenant"],
        },
        "Adjuster Information": {
            "type": "object",
            "properties": {
                "Adjuster Name": {"type": ["string", "null"]},
                "Adjuster Phone Number": {"type": ["string", "null"]},
                "Adjuster Email": {"type": ["string", "null"]},
                "Job Title": {"type": ["string", "null"]},
                "Address": {"type": ["string", "null"]},
                "Policy #": {"type": ["string", "null"]},
            },
            "required": ["Adjuster Name", "Adjuster Phone Number", "Adjuster Email", "Job Title", "Address", "Policy #"],
        },
        "Assignment Information": {
            "type": "object",
            "properties": {
                "Date of Loss/Occurrence": {"type": ["string", "null"]},
                "Cause of loss": {"type": ["string", "null"]},
                "Facts of Loss": {"type": ["string", "null"]},
                "Loss Description": {"type": ["string", "null"]},
                "Residence Occupied During Loss": {"type": ["string", "boolean", "null"]},
                "Was Someone home at time of damage": {"type": ["string", "boolean", "null"]},
                "Repair or Mitigation Progress": {"type": ["string", "null"]},
                "Type": {"type": ["string", "null"]},
                "Inspection type": {"type": ["string", "null"]},
            },
            "required": [
                "Date of Loss/Occurrence",
                "Cause of loss",
                "Facts of Loss",
                "Loss Description",
                "Residence Occupied During Loss",
                "Was Someone home at time of damage",
                "Repair or Mitigation Progress",
                "Type",
                "Inspection type",
            ],
        },
        "Assignment Type": {
            "type": "object",
            "properties": {
                "Wind": {"type": ["boolean", "null"]},
                "Structural": {"type": ["boolean", "null"]},
                "Hail": {"type": ["boolean", "null"]},
                "Foundation": {"type": ["boolean", "null"]},
                "Other": {
                    "type": "object",
                    "properties": {
                        "Checked": {"type": ["boolean", "null"]},
                        "Details": {"type": ["string", "null"]},
                    },
                    "required": ["Checked", "Details"],
                },
            },
            "required": ["Wind", "Structural", "Hail", "Foundation", "Other"],
        },
        "Additional details/Special Instructions": {"type": ["string", "null"]},
        "Attachment(s)": {
            "type": "array",
            "items": {"type": "string"},
        },
        "Entities": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "TransformerEntities": {  # New Field Added
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "missing_fields": {  # Optional Fields for Parser Feedback
            "type": "array",
            "items": {"type": "string"}
        },
        "inconsistent_fields": {  # Optional Fields for Parser Feedback
            "type": "array",
            "items": {"type": "string"}
        },
        "user_notifications": {  # Optional Field for Parser Feedback
            "type": "string",
            "enum": ["Attachments mentioned but not found in the email."]
        },
    },
    "required": [
        "Requesting Party",
        "Insured Information",
        "Adjuster Information",
        "Assignment Information",
        "Assignment Type",
        "Additional details/Special Instructions",
        "Attachment(s)",
        "Entities",
        "TransformerEntities",
    ],
    "additionalProperties": False
}

def validate_json(parsed_data: dict) -> (bool, str):
    """
    Validate the parsed data against the QuickBase schema.

    Args:
        parsed_data (dict): The parsed data to validate.

    Returns:
        tuple: (is_valid (bool), error_message (str))
    """
    logger = logging.getLogger("Validation")
    validator = Draft7Validator(assignment_schema)
    errors = sorted(validator.iter_errors(parsed_data), key=lambda e: e.path)

    if errors:
        error_messages = [f"{'.'.join(map(str, error.path))}: {error.message}" for error in errors]
        logger.error(f"Validation failed with errors: {error_messages}")
        return False, "\n".join(error_messages)
    logger.debug("Validation successful. Parsed data conforms to the schema.")
    return True, ""
