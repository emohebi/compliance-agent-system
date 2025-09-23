"""Compliance Agent for checking document compliance."""

import logging
from typing import Dict, Any, List, Optional
from strands import Agent
from strands.models import BedrockModel
from config.settings import settings
from tools.compliance_tools import compliance_tools
from models.compliance_rules import ComplianceResult

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
            Your responsibilities include:
            1. Checking documents for compliance violations
            2. Identifying potential risks and issues
            3. Generating detailed compliance reports
            4. Providing remediation recommendations
            
            When checking compliance:
            - Be thorough and systematic
            - Check for PII, security risks, and regulatory requirements
            - Provide clear explanations for violations
            - Suggest specific remediation steps
            
            Always maintain high standards for data protection and security."""
        )
    
    def check_document(
        self,
        document_content: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> ComplianceResult:
        """
        Check a single document for compliance.
        
        Args:
            document_content: Document content to check
            document_metadata: Optional document metadata
            
        Returns:
            Compliance check results
        """
        logger.info("Checking document compliance")
        
        prompt = f"""
        Perform a comprehensive compliance check on the following document:
        
        Document Content: {document_content[:1000]}...
        {f'Metadata: {document_metadata}' if document_metadata else ''}
        
        Check for:
        1. PII (Personal Identifiable Information)
        2. Security vulnerabilities
        3. Regulatory compliance issues
        4. Data protection concerns
        
        Provide detailed findings and recommendations.
        """
        
        response = self.agent(prompt)
        
        # Also run the tool directly for structured results
        tool_result = compliance_tools.check_document_compliance(
            document_content, 
            document_metadata
        )
        
        return tool_result
    
    def check_batch(
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
        logger.info(f"Checking batch of {len(documents)} documents")
        
        prompt = f"""
        Perform batch compliance checking on {len(documents)} documents.
        
        Analyze each document for compliance violations and provide:
        1. Overall compliance statistics
        2. Common violation patterns
        3. Risk assessment
        4. Prioritized remediation plan
        """
        
        response = self.agent(prompt)
        
        # Get structured results from tool
        tool_result = compliance_tools.check_batch_compliance(documents)
        
        return {
            'analysis': response.content,
            'results': tool_result
        }
    
    def generate_report(
        self,
        compliance_results: List[ComplianceResult]
    ) -> str:
        """
        Generate a detailed compliance report.
        
        Args:
            compliance_results: List of compliance check results
            
        Returns:
            Formatted compliance report
        """
        logger.info("Generating compliance report")
        
        # Get basic report from tool
        basic_report = compliance_tools.generate_compliance_report(compliance_results)
        
        # Enhance with agent analysis
        prompt = f"""
        Based on the following compliance check results, generate an executive summary:
        
        {basic_report}
        
        Include:
        1. Executive summary
        2. Key risks and their business impact
        3. Recommended immediate actions
        4. Long-term compliance strategy
        """
        
        response = self.agent(prompt)
        
        # Combine reports
        final_report = f"{response.content}\n\n---\n\n{basic_report}"
        return final_report
    
    def suggest_remediation(
        self,
        violations: List[Dict[str, Any]]
    ) -> str:
        """
        Suggest remediation steps for violations.
        
        Args:
            violations: List of compliance violations
            
        Returns:
            Remediation recommendations
        """
        logger.info("Generating remediation suggestions")
        
        prompt = f"""
        Analyze the following compliance violations and provide remediation steps:
        
        Violations:
        {violations}
        
        For each violation, provide:
        1. Specific remediation steps
        2. Priority level
        3. Estimated effort
        4. Prevention measures
        
        Format as an actionable plan.
        """
        
        response = self.agent(prompt)
        return response.content