"""Compliance checking tools."""

from typing import Dict, Any, List, Optional
from strands import tool
import json
import logging
from datetime import datetime
from models.compliance_rules import ComplianceRule, ComplianceResult
from pathlib import Path

logger = logging.getLogger(__name__)

class ComplianceTools:
    """Tools for compliance checking."""
    
    def __init__(self, rules_file: Optional[str] = None):
        """Initialize compliance tools with rules."""
        self.rules = self._load_rules(rules_file)
    
    def _load_rules(self, rules_file: Optional[str]) -> List[ComplianceRule]:
        """Load compliance rules from file or use defaults."""
        if rules_file:
            try:
                with open(rules_file, 'r') as f:
                    rules_data = json.load(f)
                    return [ComplianceRule(**rule) for rule in rules_data.get('rules', [])]
            except Exception as e:
                logger.warning(f"Failed to load rules from {rules_file}: {e}")
        
        return self._get_default_rules()
    
    def _get_default_rules(self) -> List[ComplianceRule]:
        """Get default compliance rules - now more generic and descriptive."""
        return [
            ComplianceRule(
                rule_id="PII_001",
                name="Personal Identifiable Information",
                description="Check for any personal identifiable information that could identify individuals",
                category="data_protection",
                severity="high",
                patterns=None,  # No regex patterns
                keywords=[
                    "social security", "ssn", "passport", "driver license",
                    "date of birth", "dob", "personal address", "home address",
                    "phone number", "email address", "bank account", "credit card",
                    "medical record", "health information", "biometric"
                ]
            ),
            ComplianceRule(
                rule_id="SEC_001",
                name="Security and Credentials",
                description="Check for exposed security credentials or sensitive authentication data",
                category="security",
                severity="critical",
                patterns=None,
                keywords=[
                    "password", "api key", "api_key", "apikey", "secret", "token",
                    "credential", "private key", "private_key", "auth", "authentication",
                    "bearer", "oauth", "jwt", "access_token", "refresh_token",
                    "client_secret", "database connection", "connection string"
                ]
            ),
            ComplianceRule(
                rule_id="REG_001",
                name="Regulatory Compliance",
                description="Check for regulatory compliance requirements and violations",
                category="regulatory",
                severity="high",
                patterns=None,
                keywords=[
                    "gdpr", "hipaa", "sox", "sarbanes", "pci", "pci-dss",
                    "ccpa", "ferpa", "glba", "fcra", "ada", "coppa",
                    "data protection", "privacy policy", "consent", "opt-out",
                    "data retention", "right to be forgotten", "data portability"
                ]
            ),
            ComplianceRule(
                rule_id="FIN_001",
                name="Financial Information",
                description="Check for sensitive financial data that requires protection",
                category="financial",
                severity="high",
                patterns=None,
                keywords=[
                    "account number", "routing number", "iban", "swift",
                    "tax id", "ein", "financial statement", "salary",
                    "compensation", "bonus", "stock option", "insider information",
                    "earnings", "revenue", "profit", "loss", "budget"
                ]
            ),
            ComplianceRule(
                rule_id="PROP_001",
                name="Proprietary Information",
                description="Check for proprietary or confidential business information",
                category="proprietary",
                severity="medium",
                patterns=None,
                keywords=[
                    "confidential", "proprietary", "trade secret", "internal only",
                    "restricted", "classified", "not for distribution", "nda",
                    "intellectual property", "patent pending", "copyright",
                    "competitive advantage", "strategic plan", "roadmap"
                ]
            )
        ]
    
    @tool
    def check_document_compliance(
        self,
        document_content: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        check_context: bool = True
    ) -> ComplianceResult:
        """
        Check document for compliance violations using intelligent analysis.
        
        This tool performs context-aware compliance checking rather than
        simple pattern matching. It understands the context and meaning
        of the content to identify real violations.
        
        Args:
            document_content: The document content to check
            document_metadata: Optional document metadata
            check_context: Whether to consider context (True) or flag any occurrence (False)
            
        Returns:
            ComplianceResult with findings
        """
        violations = []
        warnings = []
        score = 1.0  # Start with perfect score
        
        # Convert content to lowercase for checking
        content_lower = document_content.lower()
        
        for rule in self.rules:
            rule_triggered = False
            findings = []
            
            # Check for keywords in context
            if rule.keywords:
                for keyword in rule.keywords:
                    if keyword.lower() in content_lower:
                        # Found keyword - analyze context if requested
                        if check_context:
                            # Extract context around keyword (50 chars before and after)
                            keyword_pos = content_lower.find(keyword.lower())
                            start = max(0, keyword_pos - 50)
                            end = min(len(document_content), keyword_pos + len(keyword) + 50)
                            context = document_content[start:end].strip()
                            
                            # Check if it's actually sensitive based on context
                            if self._is_sensitive_in_context(keyword, context, rule.category):
                                findings.append({
                                    'keyword': keyword,
                                    'context': context,
                                    'position': keyword_pos
                                })
                                rule_triggered = True
                        else:
                            # No context check - flag any occurrence
                            findings.append({'keyword': keyword})
                            rule_triggered = True
            
            # Create violation or warning based on findings
            if rule_triggered and findings:
                finding = {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'severity': rule.severity,
                    'category': rule.category,
                    'description': f"{rule.description}. Found {len(findings)} potential issue(s).",
                    'details': findings[:5]  # Limit details to prevent overflow
                }
                
                if rule.severity in ["critical", "high"]:
                    violations.append(finding)
                    score -= 0.2 * (len(findings) / 10)  # Scale penalty by number of findings
                else:
                    warnings.append(finding)
                    score -= 0.05 * (len(findings) / 10)
        
        # Additional intelligent checks
        additional_issues = self._perform_intelligent_checks(document_content, document_metadata)
        violations.extend(additional_issues.get('violations', []))
        warnings.extend(additional_issues.get('warnings', []))
        score -= additional_issues.get('score_penalty', 0)
        
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
    
    def _is_sensitive_in_context(self, keyword: str, context: str, category: str) -> bool:
        """
        Determine if a keyword represents actual sensitive data in its context.
        This method uses heuristics to reduce false positives.
        
        Args:
            keyword: The keyword found
            context: The surrounding text
            category: The rule category
            
        Returns:
            True if the context suggests actual sensitive data
        """
        context_lower = context.lower()
        
        # Check for common false positive indicators
        false_positive_indicators = [
            "example", "sample", "test", "demo", "placeholder",
            "xxx", "****", "[redacted]", "dummy", "fake",
            "documentation", "template", "mock"
        ]
        
        for indicator in false_positive_indicators:
            if indicator in context_lower:
                return False  # Likely not real sensitive data
        
        # Category-specific context checks
        if category == "data_protection":
            # Check if it looks like real PII
            if keyword in ["ssn", "social security"]:
                # Look for number patterns that could be SSN
                import re
                if re.search(r'\b\d{3}-?\d{2}-?\d{4}\b', context):
                    return True
            elif keyword in ["email", "email address"]:
                # Look for email pattern
                import re
                if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', context):
                    return True
            elif keyword in ["credit card", "card number"]:
                # Look for credit card pattern
                import re
                if re.search(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', context):
                    return True
        
        elif category == "security":
            # Check if it looks like actual credentials
            if any(kw in keyword for kw in ["password", "token", "key", "secret"]):
                # Look for patterns that suggest real credentials
                import re
                # Check for base64-like strings, hex strings, or long random strings
                if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', context) or \
                   re.search(r'[0-9a-fA-F]{32,}', context) or \
                   re.search(r'[A-Za-z0-9]{20,}', context):
                    return True
        
        # Default: consider it sensitive if we're not sure
        return True
    
    def _perform_intelligent_checks(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform additional intelligent compliance checks beyond keyword matching.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Dictionary with additional violations, warnings, and score penalty
        """
        additional_violations = []
        additional_warnings = []
        score_penalty = 0
        
        # Check for suspicious patterns that might indicate data dumps
        lines = content.split('\n')
        
        # Check for CSV-like data that might contain bulk PII
        if len(lines) > 10:
            comma_lines = sum(1 for line in lines if line.count(',') > 3)
            if comma_lines > len(lines) * 0.7:
                additional_warnings.append({
                    'rule_id': 'INTEL_001',
                    'rule_name': 'Bulk Data Detection',
                    'severity': 'medium',
                    'category': 'data_protection',
                    'description': 'Document appears to contain bulk structured data that may include PII'
                })
                score_penalty += 0.1
        
        # Check for JSON structures with sensitive field names
        if '{' in content and '}' in content:
            sensitive_json_fields = [
                '"password"', '"api_key"', '"apiKey"', '"secret"',
                '"ssn"', '"socialSecurity"', '"creditCard"', '"token"'
            ]
            for field in sensitive_json_fields:
                if field in content:
                    additional_violations.append({
                        'rule_id': 'INTEL_002',
                        'rule_name': 'Sensitive JSON Fields',
                        'severity': 'high',
                        'category': 'security',
                        'description': f'JSON structure contains sensitive field: {field}'
                    })
                    score_penalty += 0.15
                    break
        
        # Check for Base64 encoded content that might hide sensitive data
        import re
        base64_pattern = r'[A-Za-z0-9+/]{50,}={0,2}'
        base64_matches = re.findall(base64_pattern, content)
        if len(base64_matches) > 3:
            additional_warnings.append({
                'rule_id': 'INTEL_003',
                'rule_name': 'Encoded Content',
                'severity': 'medium',
                'category': 'security',
                'description': 'Document contains multiple Base64-like encoded strings that may contain sensitive data'
            })
            score_penalty += 0.05
        
        # Check document metadata for sensitivity indicators
        if metadata:
            if metadata.get('classification') in ['confidential', 'restricted', 'internal']:
                additional_warnings.append({
                    'rule_id': 'INTEL_004',
                    'rule_name': 'Classified Document',
                    'severity': 'medium',
                    'category': 'proprietary',
                    'description': f"Document is classified as: {metadata.get('classification')}"
                })
                score_penalty += 0.1
        
        return {
            'violations': additional_violations,
            'warnings': additional_warnings,
            'score_penalty': score_penalty
        }
    
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
            Batch compliance results with statistics and patterns
        """
        results = []
        total_compliant = 0
        violation_patterns = {}
        
        for doc in documents:
            result = self.check_document_compliance(
                document_content=doc.get('content', ''),
                document_metadata=doc.get('metadata', {}),
                check_context=True  # Use context-aware checking
            )
            
            if result.is_compliant:
                total_compliant += 1
            
            # Track violation patterns
            for violation in result.violations:
                category = violation.get('category', 'unknown')
                violation_patterns[category] = violation_patterns.get(category, 0) + 1
            
            results.append(result.dict())
        
        # Identify most common issues
        common_issues = sorted(violation_patterns.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_documents': len(documents),
            'compliant_documents': total_compliant,
            'non_compliant_documents': len(documents) - total_compliant,
            'compliance_rate': total_compliant / len(documents) if documents else 0,
            'common_issues': common_issues[:5],  # Top 5 issue categories
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
        avg_score = sum(r.score for r in compliance_results) / total if total > 0 else 0
        
        report += "## Executive Summary\n"
        report += f"- Total Documents: {total}\n"
        report += f"- Compliant: {compliant} ({(compliant/total)*100:.1f}%)\n"
        report += f"- Non-Compliant: {total - compliant} ({((total-compliant)/total)*100:.1f}%)\n"
        report += f"- Average Compliance Score: {avg_score:.2f}\n\n"
        
        # Risk Assessment
        report += "## Risk Assessment\n"
        if avg_score >= 0.9:
            report += "✅ **Low Risk** - Strong compliance posture\n"
        elif avg_score >= 0.7:
            report += "⚠️ **Medium Risk** - Some compliance issues need attention\n"
        else:
            report += "❌ **High Risk** - Significant compliance violations detected\n"
        report += "\n"
        
        # Violations by category
        violations_by_category = {}
        all_violations = []
        
        for result in compliance_results:
            for violation in result.violations:
                category = violation.get('category', 'unknown')
                severity = violation.get('severity', 'unknown')
                if category not in violations_by_category:
                    violations_by_category[category] = {'count': 0, 'critical': 0, 'high': 0, 'medium': 0}
                violations_by_category[category]['count'] += 1
                if severity == 'critical':
                    violations_by_category[category]['critical'] += 1
                elif severity == 'high':
                    violations_by_category[category]['high'] += 1
                else:
                    violations_by_category[category]['medium'] += 1
                all_violations.append(violation)
        
        if violations_by_category:
            report += "## Violations by Category\n"
            for category, stats in sorted(violations_by_category.items(), 
                                        key=lambda x: x[1]['count'], 
                                        reverse=True):
                report += f"\n### {category.title()}\n"
                report += f"- Total: {stats['count']}\n"
                if stats['critical'] > 0:
                    report += f"- Critical: {stats['critical']} ⚠️\n"
                if stats['high'] > 0:
                    report += f"- High: {stats['high']}\n"
                if stats['medium'] > 0:
                    report += f"- Medium: {stats['medium']}\n"
            report += "\n"
        
        # Top violations
        if all_violations:
            report += "## Most Frequent Violations\n"
            violation_counts = {}
            for v in all_violations:
                name = v.get('rule_name', 'Unknown')
                violation_counts[name] = violation_counts.get(name, 0) + 1
            
            for name, count in sorted(violation_counts.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:5]:
                report += f"- {name}: {count} occurrences\n"
            report += "\n"
        
        # Recommendations
        report += "## Recommendations\n"
        if violations_by_category.get('data_protection', {}).get('count', 0) > 0:
            report += "1. **Data Protection**: Implement data masking and encryption for PII\n"
        if violations_by_category.get('security', {}).get('count', 0) > 0:
            report += "2. **Security**: Rotate exposed credentials and implement secrets management\n"
        if violations_by_category.get('regulatory', {}).get('count', 0) > 0:
            report += "3. **Regulatory**: Review regulatory requirements and update policies\n"
        if violations_by_category.get('financial', {}).get('count', 0) > 0:
            report += "4. **Financial**: Enhance controls around financial data handling\n"
        if violations_by_category.get('proprietary', {}).get('count', 0) > 0:
            report += "5. **Proprietary**: Review document classification and access controls\n"
        
        return report
    
    @tool
    def check_keyword_compliance(
        self,
        required_keywords: List[str],
        match_type: str, # = "all",  # "all" or "any"
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if required keywords are present in documents for compliance.
        
        Args:
            required_keywords: List of keywords that must be present
            documents: List of documents to check
            match_type: "all" (must contain all keywords) or "any" (must contain at least one)
            
        Returns:
            Compliance results showing which documents have/lack keywords
        """
        # kb_file = Path(__file__).parent.parent / "data/local_knowledge_base.json"
        # try:
        #     with open(kb_file, 'r', encoding='utf-8') as f:
        #         kb_data = json.load(f)
        #         documents = kb_data.get('documents', [])
        #         logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
        # except Exception as e:
        #     logger.error(f"Error loading knowledge base: {e}")
        #     documents = []
        match_type = 'all' if match_type not in ['all', 'any'] else match_type
        results = {
            'required_keywords': required_keywords,
            'match_type': match_type,
            'total_documents': len(documents),
            'compliant_documents': [],
            'non_compliant_documents': [],
            'compliance_rate': 0.0,
            'summary': {}
        }
        
        for i, doc in enumerate(documents):
            content = doc.get('content', '').lower()
            doc_id = doc.get('id', f'doc_{i}')
            doc_metadata = doc.get('metadata', {})
            
            # Check which keywords are present
            found_keywords = []
            missing_keywords = []
            
            for keyword in required_keywords:
                if keyword.lower() in content:
                    found_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)
            
            # Determine compliance based on match_type
            is_compliant = False
            if match_type == "all":
                is_compliant = len(missing_keywords) == 0
            elif match_type == "any":
                is_compliant = len(found_keywords) > 0
            
            doc_result = {
                'document_id': doc_id,
                'metadata': doc_metadata,
                'found_keywords': found_keywords,
                'missing_keywords': missing_keywords,
                'is_compliant': is_compliant,
                'compliance_percentage': len(found_keywords) / len(required_keywords) * 100
            }
            
            if is_compliant:
                results['compliant_documents'].append(doc_result)
            else:
                results['non_compliant_documents'].append(doc_result)
        
        # Calculate overall compliance rate
        results['compliance_rate'] = len(results['compliant_documents']) / len(documents) if documents else 0
        
        # Create summary
        results['summary'] = {
            'total_compliant': len(results['compliant_documents']),
            'total_non_compliant': len(results['non_compliant_documents']),
            'compliance_percentage': results['compliance_rate'] * 100,
            'most_missing_keyword': self._find_most_missing_keyword(results['non_compliant_documents']),
            'least_found_keyword': self._find_least_found_keyword(documents, required_keywords)
        }
        
        return results

    def _find_most_missing_keyword(self, non_compliant_docs: List[Dict]) -> Optional[str]:
        """Find the keyword that is missing most often."""
        missing_counts = {}
        for doc in non_compliant_docs:
            for keyword in doc.get('missing_keywords', []):
                missing_counts[keyword] = missing_counts.get(keyword, 0) + 1
        
        if missing_counts:
            return max(missing_counts, key=missing_counts.get)
        return None

    def _find_least_found_keyword(self, documents: List[Dict], keywords: List[str]) -> Optional[str]:
        """Find the keyword that appears in fewest documents."""
        keyword_counts = {kw: 0 for kw in keywords}
        
        for doc in documents:
            content = doc.get('content', '').lower()
            for keyword in keywords:
                if keyword.lower() in content:
                    keyword_counts[keyword] += 1
        
        if keyword_counts:
            return min(keyword_counts, key=keyword_counts.get)
        return None

# Create singleton instance
compliance_tools = ComplianceTools()