"""Compliance Agent for checking document compliance."""

import logging
from typing import Dict, Any, List, Optional
from strands import Agent
from strands.models import BedrockModel
from config.settings import settings
from tools.compliance_tools import compliance_tools
from models.compliance_rules import ComplianceResult, BatchComplianceResult, ComplianceReport, RemediationPlan, AgentComplianceResult, KeywordComplianceReport
from datetime import datetime
import json

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
                compliance_tools.generate_compliance_report,
                compliance_tools.check_keyword_compliance
                # compliance_tools.suggest_remediation
            ],
            system_prompt="""You are a Compliance specialist agent.
            When using structured output:
            - Always provide is_compliant as boolean (true/false)
            - Always provide score as float between 0.0 and 1.0
            - Provide violations and warnings as lists of strings
            """
        )
    
    def check_document(self, document_content: str, document_metadata: Optional[Dict[str, Any]] = None) -> ComplianceResult:
        logger.info("Checking document compliance")
        
        # Step 1: Tool invocation
        tool_prompt = f"""
        Use the check_document_compliance tool to analyze this document.
        
        Tool parameters:
        - document_content: {document_content[:500]}...
        - document_metadata: {document_metadata}
        
        Execute the check_document_compliance tool now.
        """
        
        tool_response = self.agent(tool_prompt)
        
        # Step 2: Format with structured output
        format_prompt = f"""
        Based on the compliance check results: {str(tool_response)}
        
        Create a compliance result with:
        - is_compliant: true if no major violations found, false otherwise
        - score: between 0.0 (many violations) and 1.0 (fully compliant)
        - violations: list of violation descriptions
        - warnings: list of warning descriptions
        """
        
        result = self.agent.structured_output(
            AgentComplianceResult,  # or ComplianceResult if it works
            format_prompt
        )
        
        # Convert to ComplianceResult if using AgentComplianceResult
        return result  # or conversion logic
    
    def check_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"Checking batch of {len(documents)} documents")
        
        # Step 1: Tool invocation
        tool_prompt = f"""
        Use the check_batch_compliance tool to analyze multiple documents.
        
        Number of documents: {len(documents)}
        
        Execute check_batch_compliance tool with the provided documents.
        """
        
        tool_response = self.agent(tool_prompt)
        
        # Step 2: Format results
        format_prompt = f"""
        Based on batch compliance results: {str(tool_response)}
        
        Summarize:
        - Total documents checked
        - Number of compliant documents
        - Overall compliance rate
        - Common violations found
        """
        
        batch_result = self.agent.structured_output(
            BatchComplianceResult,
            format_prompt
        )
        
        return {
            'analysis': batch_result.summary,
            'results': batch_result.dict()
        }
        
    def generate_report(self, compliance_results: List[ComplianceResult]) -> str:
        logger.info("Generating compliance report")
        
        # Step 1: Tool invocation
        tool_prompt = f"""
        Use the generate_compliance_report tool with {len(compliance_results)} compliance results.
        
        Execute generate_compliance_report tool now.
        """
        
        tool_response = self.agent(tool_prompt)
        
        # Step 2: Format as executive report
        format_prompt = f"""
        Based on this compliance report: {str(tool_response)}
        
        Create an executive report with:
        - Executive summary
        - Key risks identified
        - Immediate actions needed
        - Long-term compliance strategy
        """
        
        report = self.agent.structured_output(
            ComplianceReport,
            format_prompt
        )
        
        # Format final report...
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
    
    def check_keyword_compliance_kb(
        self,
        documents: List[Dict[str, Any]],
        required_keywords: List[str],
        match_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Check if required keywords exist in all knowledge base documents.
        
        Args:
            required_keywords: Keywords that must be present for compliance
            match_type: "all" (must have all keywords) or "any" (must have at least one)
            
        Returns:
            Detailed compliance results
        """
        logger.info(f"Checking keyword compliance for: {required_keywords}")
        
        # # Step 1: Retrieve all documents from knowledge base
        # retrieve_prompt = """
        # Use the list_knowledge_base_documents tool to get ALL documents.
        # Set limit to maximum (1000) to get as many documents as possible.
        
        # Execute list_knowledge_base_documents tool now.
        # """
        
        # list_response = self.agent(retrieve_prompt)
        
        # # Step 2: For each document, retrieve full content
        # # Note: You might need to batch this for large KBs
        # retrieve_all_prompt = f"""
        # Retrieve the full content of all documents in the knowledge base.
        # We need to check them for compliance.
        
        # Use retrieve_from_knowledge_base tool multiple times if needed.
        # """
        
        # all_docs_response = self.agent(retrieve_all_prompt)
        
        # Step 3: Check keyword compliance
        check_prompt = f"""
        Use the check_keyword_compliance tool with these parameters:
        - required_keywords: {json.dumps(required_keywords)}
        - match_type: {match_type}
        - documents: {json.dumps(documents)}
        
        Execute check_keyword_compliance tool now.
        """
        
        compliance_response = self.agent(check_prompt)
        
        # Step 4: Format results
        format_prompt = f"""
        Based on the keyword compliance check results: {str(compliance_response)}
        
        Create a detailed report showing:
        1. Which documents are compliant (have required keywords)
        2. Which documents are non-compliant and what keywords they're missing
        3. Overall compliance statistics
        4. Recommendations for non-compliant documents
        """
        
        report = self.agent.structured_output(
            KeywordComplianceReport,
            format_prompt
        )
        
        return report.dict()
    
    def suggest_remediation(
        self,
        violations: List[Any]  # Can be List[str] or List[Dict]
    ) -> str:
        """Suggest remediation steps for violations."""
        logger.info("Generating remediation suggestions")
        
        # Handle both string and dict violations
        if violations and isinstance(violations[0], str):
            # Simple string violations
            violations_summary = "\n".join([
                f"- {v}"
                for v in violations[:20]
            ])
        else:
            # Dictionary violations
            violations_summary = "\n".join([
                f"- {v.get('rule_name', 'Unknown')}: {v.get('description', v) if isinstance(v, dict) else v}"
                for v in violations[:20]
            ])
        
        # Step 1: Get remediation suggestions from agent
        prompt = f"""
        Analyze these compliance violations and provide a remediation plan:
        
        Violations:
        {violations_summary}
        
        Create an actionable remediation plan with:
        1. Specific priority actions with effort estimates
        2. Prevention measures for the future
        3. Realistic timeline for implementation
        """
        
        # Use the prompt to get suggestions
        tool_response = self.agent(prompt)  # Now using the prompt variable!
        
        # Step 2: Try to format with structured output
        try:
            format_prompt = f"""
            Format this remediation advice into a structured plan:
            
            {str(tool_response)}
            
            Structure it with:
            - Priority actions (with priority level and effort)
            - Prevention measures
            - Timeline estimate
            """
            
            plan = self.agent.structured_output(
                RemediationPlan,
                format_prompt
            )
            
            # Format as actionable text
            formatted_plan = "## Remediation Plan\n\n"
            formatted_plan += "### Priority Actions\n"
            for action in plan.priority_actions:
                if isinstance(action, dict):
                    formatted_plan += f"- **{action.get('priority', 'Medium')}**: {action.get('action', '')}\n"
                    formatted_plan += f"  *Effort: {action.get('effort', 'Unknown')}*\n"
                else:
                    formatted_plan += f"- {action}\n"
            
            formatted_plan += "\n### Prevention Measures\n"
            for measure in plan.prevention_measures:
                formatted_plan += f"- {measure}\n"
            
            formatted_plan += f"\n### Timeline\n{plan.estimated_timeline}"
            
            return formatted_plan
            
        except Exception as e:
            logger.warning(f"Structured output failed, using raw response: {e}")
            return str(tool_response)