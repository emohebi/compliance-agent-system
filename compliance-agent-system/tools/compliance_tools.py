"""Compliance checking tools."""

from typing import Dict, Any, List, Optional
from strands import tool
import json
import re
import logging
from datetime import datetime
from models.compliance_rules import ComplianceRule, ComplianceResult

logger = logging.getLogger(__name__)

class ComplianceTools:
    """Tools for compliance checking."""
    
    def __init__(self, rules_file: Optional[str] = None):
        """Initialize compliance tools with rules."""
        self.rules = self._load_rules(rules_file)
    
    def _load_rules(self, rules_file: Optional[str]) -> List[ComplianceRule]:
        """Load compliance rules from file."""
        if not rules_file:
            return self._get_default_rules()
        
        try:
            with open(rules_file, 'r') as f:
                rules_data = json.load(f)
                return [ComplianceRule(**rule) for rule in rules_data]
        except Exception as e:
            logger.warning(f"Failed to load rules from {rules_file}: {e}")
            return self._get_default_rules()
    
    def _get_default_rules(self) -> List[ComplianceRule]:
        """Get default compliance rules."""
        return [
            ComplianceRule(
                rule_id="PII_001",
                name="PII Detection",
                description="Check for personally identifiable information",
                category="data_protection",
                severity="high",
                patterns=[
                    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"  # Credit card
                ]
            ),
            ComplianceRule(
                rule_id="SEC_001",
                name="Security Keywords",
                description="Check for security-sensitive keywords",
                category="security",
                severity="medium",
                keywords=["password", "api_key", "secret", "token", "credential"]
            ),
            ComplianceRule(
                rule_id="REG_001",
                name="Regulatory Compliance",
                description="Check for regulatory compliance keywords",
                category="regulatory",
                severity="high",
                keywords=["gdpr", "hipaa", "sox", "pci-dss"]
            )
        ]
    
    @tool
    def check_document_compliance(
        self,
        document_content: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> ComplianceResult:
        """
        Check document for compliance violations.
        
        Args:
            document_content: The document content to check
            document_metadata: Optional document metadata
            
        Returns:
            ComplianceResult with findings
        """
        violations = []
        warnings = []
        score = 1.0  # Start with perfect score
        
        for rule in self.rules:
            # Check patterns
            if rule.patterns:
                for pattern in rule.patterns:
                    matches = re.findall(pattern, document_content, re.IGNORECASE)
                    if matches:
                        violation = {
                            'rule_id': rule.rule_id,
                            'rule_name': rule.name,
                            'severity': rule.severity,
                            'matches': len(matches),
                            'category': rule.category,
                            'description': f"Found {len(matches)} instances matching pattern"
                        }
                        
                        if rule.severity == "high":
                            violations.append(violation)
                            score -= 0.2
                        else:
                            warnings.append(violation)
                            score -= 0.1
            
            # Check keywords
            if rule.keywords:
                for keyword in rule.keywords:
                    if keyword.lower() in document_content.lower():
                        violation = {
                            'rule_id': rule.rule_id,
                            'rule_name': rule.name,
                            'severity': rule.severity,
                            'keyword': keyword,
                            'category': rule.category,
                            'description': f"Found keyword: {keyword}"
                        }
                        
                        if rule.severity == "high":
                            violations.append(violation)
                            score -= 0.15
                        else:
                            warnings.append(violation)
                            score -= 0.05
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        return ComplianceResult(
            is_compliant=len(violations) == 0,
            score=score,
            violations=violations,
            warnings=warnings,
            checked_at=datetime.now().isoformat(),
            document_metadata=document_metadata
        )
    
    @tool
    def check_batch_compliance(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check multiple documents for compliance.
        
        Args:
            documents: List of documents to check
            
        Returns:
            Batch compliance results
        """
        results = []
        total_compliant = 0
        
        for doc in documents:
            result = self.check_document_compliance(
                document_content=doc.get('content', ''),
                document_metadata=doc.get('metadata', {})
            )
            
            if result.is_compliant:
                total_compliant += 1
            
            results.append(result.dict())
        
        return {
            'total_documents': len(documents),
            'compliant_documents': total_compliant,
            'non_compliant_documents': len(documents) - total_compliant,
            'compliance_rate': total_compliant / len(documents) if documents else 0,
            'results': results
        }
    
    @tool
    def generate_compliance_report(
        self,
        compliance_results: List[ComplianceResult]
    ) -> str:
        """
        Generate a compliance report from results.
        
        Args:
            compliance_results: List of compliance results
            
        Returns:
            Formatted compliance report
        """
        report = "# Compliance Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        
        # Summary
        total = len(compliance_results)
        compliant = sum(1 for r in compliance_results if r.is_compliant)
        
        report += "## Summary\n"
        report += f"- Total Documents: {total}\n"
        report += f"- Compliant: {compliant}\n"
        report += f"- Non-Compliant: {total - compliant}\n"
        report += f"- Compliance Rate: {(compliant/total)*100:.1f}%\n\n"
        
        # Violations by category
        violations_by_category = {}
        for result in compliance_results:
            for violation in result.violations:
                category = violation.get('category', 'unknown')
                if category not in violations_by_category:
                    violations_by_category[category] = 0
                violations_by_category[category] += 1
        
        if violations_by_category:
            report += "## Violations by Category\n"
            for category, count in violations_by_category.items():
                report += f"- {category}: {count}\n"
            report += "\n"
        
        # High severity violations
        high_severity = []
        for result in compliance_results:
            for violation in result.violations:
                if violation.get('severity') == 'high':
                    high_severity.append(violation)
        
        if high_severity:
            report += "## High Severity Violations\n"
            for violation in high_severity[:10]:  # Limit to top 10
                report += f"- {violation['rule_name']}: {violation['description']}\n"
            report += "\n"
        
        return report

# Create singleton instance
compliance_tools = ComplianceTools()