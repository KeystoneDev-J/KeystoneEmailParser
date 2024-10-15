# src/parsers/parser_options.py

from enum import Enum

class ParserOption(Enum):
    """
    Enumeration of available parser options.

    Attributes:
        RULE_BASED (str): Identifier for the rule-based parser.
        HYBRID_PARSER (str): Identifier for the hybrid parser.
        ENHANCED_PARSER (str): Identifier for the enhanced parser.
    """
    RULE_BASED = 'rule_based'
    HYBRID_PARSER = 'hybrid_parser'
    ENHANCED_PARSER = 'enhanced_parser' 