# src/utils/security.py

import bleach
import logging
import re

def sanitize_content(email_content: str) -> str:
    """
    Sanitize the email content to remove potentially malicious code and unwanted HTML,
    while preserving essential formatting for data extraction.

    Args:
        email_content (str): The raw email content.

    Returns:
        str: The sanitized email content.
    """
    logger = logging.getLogger("Security")

    try:
        # Define allowed tags and attributes
        allowed_tags = list(bleach.sanitizer.ALLOWED_TAGS) + [
            'p', 'br', 'span', 'div', 'ul', 'li', 'ol', 'strong', 'em', 'table', 'tr', 'td', 'th'
        ]
        allowed_attributes = {
            '*': ['style', 'class', 'id'],
            'a': ['href', 'title'],
            'img': ['src', 'alt'],
        }
        allowed_protocols = set(bleach.sanitizer.ALLOWED_PROTOCOLS) | {'mailto', 'tel'}

        # Define CSS sanitizer with allowed CSS properties
        css_sanitizer = bleach.css_sanitizer.CSSSanitizer(
            allowed_css_properties=['color', 'font-weight', 'background-color', 'text-align']
        )

        # Clean the content with bleach
        sanitized = bleach.clean(
            email_content,
            tags=allowed_tags,
            attributes=allowed_attributes,
            protocols=allowed_protocols,
            css_sanitizer=css_sanitizer,
            strip=True
        )

        # Remove asterisks used for bolding and other markdown-like syntax
        sanitized = re.sub(r'\*{1,2}', '', sanitized)

        logger.debug("Email content sanitized successfully.")
        return sanitized

    except Exception as e:
        logger.error(f"Error during sanitizing content: {e}", exc_info=True)
        # In case of sanitization failure, return a safe default
        return ""