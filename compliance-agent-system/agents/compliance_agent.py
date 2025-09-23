"""Compliance Agent for checking document compliance."""

import logging
from typing import Dict, Any, List, Optional
from strands import Agent
from strands.models import BedrockModel
from config.settings import settings
from tools.compliance_tools import compliance_tools
from models.compliance_rules import ComplianceResult, BatchComplianceResult, ComplianceReport, RemediationPlan, AgentComplianceResult
from datetime import datetime

logger = logging.getLogger(__name__)

class ComplianceAgent:
    """Agent for compliance checking and reporting."""
    
    def __init__(self):
        """Initialize the Compliance agent."""
        # Setup Bedrock model
        self.model = BedrockModel(**settings.get_bedrock_config())
        
        # Create agent with compliance tools
        self.agent = Agent(
            model=self.model,
            tools=[
                compliance_tools.check_document_compliance,
                compliance_tools.check_batch_compliance,
                compliance_tools.generate_compliance_report
            ],
            system_prompt="""You are a Compliance specialist agent.
            When using structured output:
            - Always provide is_compliant as boolean (true/false)
            - Always provide score as float between 0.0 and 1.0
            - Provide violations and warnings as lists of strings
            """
        )
    
    def check_document(
        self,
        document_content: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> ComplianceResult:
        """Check a single document for compliance."""
        logger.info("Checking document compliance")
        
        prompt = f"""
        Check this document for compliance violations:
        
        Document: {document_content[:1000]}...
        
        Analyze for:
        1. PII (SSN format: XXX-XX-XXXX, emails, credit cards)  
        2. Security issues (passwords, API keys, tokens)
        3. Regulatory keywords (GDPR, HIPAA, PCI-DSS)
        
        Determine:
        - is_compliant: true if no major issues, false if violations found
        - score: 0.0 (many violations) to 1.0 (no violations)
        - violations: list each violation found
        - warnings: list potential issues
        """
        
        # Get simplified result from agent
        agent_result = self.agent.structured_output(
            AgentComplianceResult,
            prompt
        )
        
        # Convert to full ComplianceResult
        return ComplianceResult(
            is_compliant=agent_result.is_compliant,
            score=agent_result.score,
            violations=[
                {
                    "rule_name": "AI Detection",
                    "description": v,
                    "severity": "high",
                    "category": "compliance"
                }
                for v in agent_result.violations
            ],
            warnings=[
                {
                    "rule_name": "AI Detection",
                    "description": w,
                    "severity": "medium",
                    "category": "warning"
                }
                for w in agent_result.warnings
            ],
            checked_at=datetime.now().isoformat(),
            document_metadata=document_metadata
        )
    
    def check_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check multiple documents for compliance."""
        logger.info(f"Checking batch of {len(documents)} documents")
        
        # Prepare documents summary for agent
        docs_summary = "\n".join([
            f"Document {i+1}: {doc.get('content', '')[:200]}..."
            for i, doc in enumerate(documents[:10])  # Limit to first 10 for prompt
        ])
        
        prompt = f"""
        Perform batch compliance checking on {len(documents)} documents.
        
        Sample of documents:
        {docs_summary}
        
        Analyze all documents and provide:
        1. Total count and compliance statistics
        2. Common violation patterns
        3. Overall compliance rate
        4. Summary of key findings
        """
        
        # Get structured batch result
        batch_result = self.agent.structured_output(
            BatchComplianceResult,
            prompt
        )
        
        return {
            'analysis': batch_result.summary,
            'results': batch_result.dict()
        }
        
    def generate_report(
        self,
        compliance_results: List[ComplianceResult]
    ) -> str:
        """Generate a detailed compliance report."""
        logger.info("Generating compliance report")
        
        # Prepare summary of results
        results_summary = {
            'total': len(compliance_results),
            'compliant': sum(1 for r in compliance_results if r.is_compliant),
            'average_score': sum(r.score for r in compliance_results) / len(compliance_results) if compliance_results else 0
        }
        
        prompt = f"""
        Generate an executive compliance report based on these results:
        
        Total documents checked: {results_summary['total']}
        Compliant documents: {results_summary['compliant']}
        Average compliance score: {results_summary['average_score']:.2f}
        
        Provide:
        1. Executive summary
        2. Key risks and their business impact
        3. Recommended immediate actions
        4. Long-term compliance strategy
        """
        
        # Get structured report
        report = self.agent.structured_output(
            ComplianceReport,
            prompt
        )
        
        # Format as markdown report
        final_report = f"""
    # Compliance Report

    ## Executive Summary
    {report.executive_summary}

    ## Statistics
    - Total Documents: {report.total_documents}
    - Compliance Rate: {report.compliance_rate:.1%}

    ## Key Risks
    {chr(10).join(f'- {risk}' for risk in report.key_risks)}

    ## Immediate Actions Required
    {chr(10).join(f'1. {action}' for i, action in enumerate(report.immediate_actions, 1))}

    ## Long-Term Strategy
    {report.long_term_strategy}

    ---
    Generated: {datetime.now().isoformat()}
    """
        
        return final_report
    
    def suggest_remediation(
        self,
        violations: List[Dict[str, Any]]
    ) -> str:
        """Suggest remediation steps for violations."""
        logger.info("Generating remediation suggestions")
        
        violations_summary = "\n".join([
            f"- {v.get('rule_name', 'Unknown')}: {v.get('description', '')}"
            for v in violations[:20]  # Limit to prevent token overflow
        ])
        
        prompt = f"""
        Analyze these compliance violations and provide a remediation plan:
        
        Violations:
        {violations_summary}
        
        Create an actionable remediation plan with:
        1. Specific priority actions with effort estimates
        2. Prevention measures for the future
        3. Realistic timeline for implementation
        """
        
        plan = self.agent.structured_output(
            RemediationPlan,
            prompt
        )
        
        # Format as actionable text
        formatted_plan = "## Remediation Plan\n\n"
        formatted_plan += "### Priority Actions\n"
        for action in plan.priority_actions:
            formatted_plan += f"- **{action.get('priority', 'Medium')}**: {action.get('action', '')}\n"
            formatted_plan += f"  *Effort: {action.get('effort', 'Unknown')}*\n"
        
        formatted_plan += "\n### Prevention Measures\n"
        for measure in plan.prevention_measures:
            formatted_plan += f"- {measure}\n"
        
        formatted_plan += f"\n### Timeline\n{plan.estimated_timeline}"
        
        return formatted_plan