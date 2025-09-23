"""Compliance rule models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class AgentComplianceResult(BaseModel):
    """Simplified compliance result for agent structured output."""
    is_compliant: bool = Field(
        description="Set to true if document has no major violations, false otherwise"
    )
    score: float = Field(
        description="Rate compliance from 0.0 (many violations) to 1.0 (no violations)",
        ge=0.0,
        le=1.0
    )
    violations: List[str] = Field(
        default_factory=list,
        description="List of violation descriptions (e.g., 'Found SSN in document', 'Password exposed')"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning descriptions (e.g., 'Potential PII', 'Weak security practice')"
    )

class RemediationPlan(BaseModel):
    """Model for remediation suggestions."""
    priority_actions: List[Dict[str, str]] = Field(
        description="Priority actions with 'action', 'priority', and 'effort' keys"
    )
    prevention_measures: List[str] = Field(description="Measures to prevent future violations")
    estimated_timeline: str = Field(description="Estimated timeline for remediation")

class ComplianceReport(BaseModel):
    """Model for compliance report."""
    executive_summary: str = Field(description="Executive summary of compliance status")
    total_documents: int = Field(description="Total documents analyzed")
    compliance_rate: float = Field(description="Overall compliance rate")
    key_risks: List[str] = Field(description="Key risks identified")
    immediate_actions: List[str] = Field(description="Recommended immediate actions")
    long_term_strategy: str = Field(description="Long-term compliance strategy")

class BatchComplianceResult(BaseModel):
    """Model for batch compliance check results."""
    total_documents: int = Field(description="Total number of documents checked")
    compliant_documents: int = Field(description="Number of compliant documents")
    non_compliant_documents: int = Field(description="Number of non-compliant documents")
    compliance_rate: float = Field(description="Overall compliance rate (0.0-1.0)")
    summary: str = Field(description="Summary of findings across all documents")
    common_violations: List[str] = Field(description="Most common violations found")

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