from abc import ABC, abstractmethod

class BaseParser(ABC):
    @abstractmethod
    def parse_email(self, email_content: str, parser_option):
        pass