# src/utils/config.py

import os
import yaml
import logging

class Config:
    """Configuration settings."""

    # ----------------------------
    # General Configuration
    # ----------------------------
    # Path to the YAML configuration file
    CONFIG_FILE_PATH = os.getenv('CONFIG_FILE_PATH', 'src/parsers/parser_config.yaml')

    # ----------------------------
    # Enhanced Parser Configuration
    # ----------------------------
    # Fuzzy Matching Threshold
    FUZZY_THRESHOLD = int(os.getenv('FUZZY_THRESHOLD', '90'))

    # Known Values for Fuzzy Matching
    KNOWN_VALUES = {
        "Claim Number": [
            "ABC123",
            "XYZ789",
            "DEF456",
            "GHI101",
            "JKL202",
            # Add more known claim numbers as needed
        ],
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
        "Adjuster Name": [
            "Michael Brown",
            "Sarah Johnson",
            "David Wilson",
            "Laura Martinez",
            "James Anderson",
            # Add more known adjuster names as needed
        ],
        # Add other fields and their known values as needed
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
            'yes',
            'y',
            'true',
            't',
            '1',
            'x',
            '[x]',
            '[X]',
            '(x)',
            '(X)',
        ],
        "negative": [
            'no',
            'n',
            'false',
            'f',
            '0',
            '[ ]',
            '()',
            '[N/A]',
            '(N/A)',
        ],
    }

    # Valid File Extensions for Attachments
    VALID_EXTENSIONS = [
        '.pdf',
        '.docx',
        '.xlsx',
        '.zip',
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
        '.txt',
        '.csv',
    ]

    # URL Validation Setting
    URL_VALIDATION = {
        "use_external_library": bool(os.getenv('URL_VALIDATION_USE_EXTERNAL_LIBRARY', 'true').lower() in ['true', '1', 'yes'])
    }

    # ----------------------------
    # Logging Configuration
    # ----------------------------
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # ----------------------------
    # Load Configuration from YAML
    # ----------------------------
    @classmethod
    def load_parser_config(cls) -> dict:
        """
        Load parser configurations from the YAML file.

        Returns:
            dict: Parsed configuration.
        """
        logger = logging.getLogger("Config")
        config = {}
        try:
            with open(cls.CONFIG_FILE_PATH, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded parser configuration from {cls.CONFIG_FILE_PATH}.")
        except FileNotFoundError:
            logger.error(f"Configuration file {cls.CONFIG_FILE_PATH} not found.")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        return config
