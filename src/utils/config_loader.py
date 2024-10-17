# src/utils/config_loader.py

from src.utils.config import Config

class ConfigLoader:
    @classmethod
    def load_config(cls) -> dict:
        """
        Load configuration settings from the Config class variables.

        Returns:
            dict: Configuration settings.
        """
        return Config.load_config()
