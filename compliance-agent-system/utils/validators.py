"""Data validation utilities."""

from typing import Any, Dict
import re

def validate_email(email: str) -> bool:
    """Validate email address."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_document(document: Dict[str, Any]) -> bool:
    """Validate document structure."""
    required_fields = ['content']
    return all(field in document for field in required_fields)
