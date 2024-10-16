# src/utils/config_loader.py

import logging
import yaml
from src.utils.config import Config

class ConfigLoader:
    """Utility class for loading and merging configurations."""

    @staticmethod
    def load_config() -> dict:
        """
        Load configuration from the YAML file and merge with Config class settings.

        Returns:
            dict: Merged configuration dictionary.
        """
        logger = logging.getLogger("ConfigLoader")
        config = {}

        # Load from YAML file
        try:
            with open(Config.CONFIG_FILE_PATH, 'r', encoding='utf-8') as file:
                parser_config = yaml.safe_load(file)
            logger.info(f"Loaded parser configuration from {Config.CONFIG_FILE_PATH}.")
            config.update(parser_config)
        except FileNotFoundError:
            logger.error(f"Configuration file {Config.CONFIG_FILE_PATH} not found.")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

        # Validate essential configuration sections
        essential_sections = ['patterns', 'additional_patterns']
        for section in essential_sections:
            if section not in config:
                logger.error(f"Missing essential configuration section: {section}")
                raise KeyError(f"Missing essential configuration section: {section}")

        # Merge with additional configurations from config.py
        config['fuzzy_threshold'] = Config.FUZZY_THRESHOLD
        config['known_values'] = Config.KNOWN_VALUES if hasattr(Config, 'KNOWN_VALUES') else {}
        config['date_formats'] = Config.DATE_FORMATS
        config['boolean_values'] = Config.BOOLEAN_VALUES
        config['valid_extensions'] = Config.VALID_EXTENSIONS
        config['url_validation'] = Config.URL_VALIDATION
        config['log_level'] = Config.LOG_LEVEL

        logger.debug(f"Merged configuration: {config}")
        return config
