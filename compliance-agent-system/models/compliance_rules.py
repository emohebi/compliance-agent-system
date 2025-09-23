"""Compliance rule models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class ComplianceRule(BaseModel):
    """Model for a compliance rule."""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    category: str = Field(..., description="Rule category")
    severity: str = Field("medium", description="Rule severity: low, medium, high")
    patterns: Optional[List[str]] = Field(None, description="Regex patterns to match")
    keywords: Optional[List[str]] = Field(None, description="Keywords to check")
    enabled: bool = Field(True, description="Whether rule is enabled")
    
class ComplianceResult(BaseModel):
    """Model for compliance check results."""
    is_compliant: bool = Field(..., description="Overall compliance status")
    score: float = Field(..., description="Compliance score (0-1)")
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    checked_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    document_metadata: Optional[Dict[str, Any]] = None
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_compliant': self.is_compliant,
            'score': self.score,
            'violations': self.violations,
            'warnings': self.warnings,
            'checked_at': self.checked_at,
            'document_metadata': self.document_metadata
        }