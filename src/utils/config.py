# src/utils/config.py

import os
import logging
import sys

class ConfigLoader:
    """Configuration loader class."""

    @staticmethod
    def load_config() -> dict:
        """
        Load configuration settings from the Config class.

        Returns:
            dict: Configuration settings.
        """
        logger = logging.getLogger("ConfigLoader")
        try:
            config = Config.load_config()
            ConfigLoader._validate_config(config, logger)
            logger.info("Configuration loaded and validated successfully.")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            sys.exit(1)

    @staticmethod
    def _validate_config(config: dict, logger: logging.Logger):
        """
        Validate the loaded configuration.

        Args:
            config (dict): The configuration dictionary.
            logger (logging.Logger): The logger instance.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        logger.debug("Validating configuration settings.")

        # Example validations
        if not isinstance(config.get("fuzzy_threshold"), int) or not (0 <= config["fuzzy_threshold"] <= 100):
            raise ValueError("fuzzy_threshold must be an integer between 0 and 100.")

        if not isinstance(config.get("valid_extensions"), list):
            raise ValueError("valid_extensions must be a list.")

        if not isinstance(config.get("log_level"), str) or config["log_level"] not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError("log_level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")

        # Add more validations as needed based on the configuration structure

class Config:
    """Configuration settings."""

    # ----------------------------
    # Enhanced Parser Configuration
    # ----------------------------
    # Fuzzy Matching Threshold
    FUZZY_THRESHOLD = int(os.getenv('FUZZY_THRESHOLD', '90'))
    
    # Known Values for Fuzzy Matching
    KNOWN_VALUES = {
        # Requesting Party
        "Insurance Company": [
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
            # Add more known insurance companies as needed
        ],
        "Handler": [
            "John Doe",
            "Jane Smith",
            "Emily Davis",
            "Michael Brown",
            "Sarah Johnson",
            "David Wilson",
            # Add more known handlers as needed
        ],
        "Carrier Claim Number": [
            "ABC123",
            "XYZ789",
            "DEF456",
            "GHI101",
            "JKL202",
            # Add more known carrier claim numbers as needed
        ],
        
        # Insured Information
        "Name": [
            "Alice Johnson",
            "Bob Lee",
            "Charlie Kim",
            # Add more known insured names as needed
        ],
        "Contact #": [
            # Assuming phone numbers are validated and formatted, known values can be omitted or included as needed
        ],
        "Loss Address": [
            # Addresses can vary widely; consider omitting or using patterns instead
        ],
        "Public Adjuster": [
            "Best Adjusters Inc.",
            "Adjuster Pros",
            # Add more known public adjusters as needed
        ],
        "Owner or Tenant": [
            "Owner",
            "Tenant",
            # Add more as needed
        ],
        
        # Adjuster Information
        "Adjuster Name": [
            "Michael Brown",
            "Sarah Johnson",
            "David Wilson",
            "Laura Martinez",
            "James Anderson",
            # Add more known adjuster names as needed
        ],
        "Adjuster Phone Number": [
            # Similar to "Contact #", phone numbers are dynamic
        ],
        "Adjuster Email": [
            # Emails are unique; known values may not be feasible
        ],
        "Job Title": [
            "Senior Adjuster",
            "Field Adjuster",
            "Claims Manager",
            # Add more as needed
        ],
        "Address": [
            # Dynamic; consider using patterns
        ],
        "Policy #": [
            "POL123456",
            "POL654321",
            # Add more known policy numbers as needed
        ],
        
        # Assignment Information
        "Date of Loss/Occurrence": [
            # Dates are dynamic; use date parsing
        ],
        "Cause of loss": [
            "Fire",
            "Water Damage",
            "Storm",
            "Theft",
            # Add more as needed
        ],
        "Facts of Loss": [
            # Dynamic; use summarization or extraction
        ],
        "Loss Description": [
            # Dynamic; use summarization or extraction
        ],
        "Residence Occupied During Loss": [
            "Yes",
            "No",
            # Add more as needed
        ],
        "Was Someone Home at Time of Damage": [
            "Yes",
            "No",
            # Add more as needed
        ],
        "Repair or Mitigation Progress": [
            "Completed",
            "In Progress",
            "Not Started",
            # Add more as needed
        ],
        "Type": [
            "Residential",
            "Commercial",
            # Add more as needed
        ],
        "Inspection type": [
            "Initial",
            "Follow-up",
            # Add more as needed
        ],
        "Assignment Type": [
            "Standard",
            "Expedited",
            # Add more as needed
        ],
        "Additional details/Special Instructions": [
            # Dynamic; use extraction
        ],
        "Attachment(s)": [
            # File names; use validation
        ],
    }

    # Date Formats for Parsing
    DATE_FORMATS = [
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y-%m-%d',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%Y.%m.%d',
        '%d.%m.%Y',
        '%m-%d-%Y',
        '%Y%m%d',
        '%B %-d, %Y',
        '%b %-d, %Y',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%fZ',
    ]

    # Boolean Values
    BOOLEAN_VALUES = {
        "positive": [
            'yes', 'y', 'true', 't', '1', 'x', '[x]', '[X]', '(x)', '(X)',
        ],
        "negative": [
            'no', 'n', 'false', 'f', '0', '[ ]', '()', '[N/A]', '(N/A)',
        ],
    }

    # Valid File Extensions for Attachments
    VALID_EXTENSIONS = [
        '.pdf', '.docx', '.xlsx', '.zip', '.png', '.jpg', '.jpeg', '.gif', '.txt', '.csv',
    ]

    # URL Validation Setting
    URL_VALIDATION = {
        "use_external_library": os.getenv('URL_VALIDATION_USE_EXTERNAL_LIBRARY', 'true').lower() in ['true', '1', 'yes']
    }

    # ----------------------------
    # Logging Configuration
    # ----------------------------
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG').upper()  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # ----------------------------
    # Load Configuration
    # ----------------------------
    @classmethod
    def load_config(cls) -> dict:
        """
        Load configuration settings.

        Returns:
            dict: Configuration settings.
        """
        logger = logging.getLogger("Config")
        config = {
            "fuzzy_threshold": cls.FUZZY_THRESHOLD,
            "known_values": cls.KNOWN_VALUES,
            "date_formats": cls.DATE_FORMATS,
            "boolean_values": cls.BOOLEAN_VALUES,
            "valid_extensions": cls.VALID_EXTENSIONS,
            "url_validation": cls.URL_VALIDATION,
            "log_level": cls.LOG_LEVEL,
        }
        logger.info("Loaded configuration from Config class variables.")
        return config
