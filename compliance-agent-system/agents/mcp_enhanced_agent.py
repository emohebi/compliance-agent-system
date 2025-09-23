"""Enhanced agent using MCP servers for extended capabilities."""

import logging
from typing import Dict, Any, List, Optional
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from config.settings import settings
import json

logger = logging.getLogger(__name__)

class MCPEnhancedAgent:
    """Agent enhanced with MCP server capabilities."""
    
    def __init__(self):
        """Initialize the MCP-enhanced agent."""
        # Setup Bedrock model
        self.model = BedrockModel(**settings.get_bedrock_config())
        
        # Initialize MCP clients
        self.compliance_mcp_client = MCPClient(
            lambda: stdio_client(
                StdioServerParameters(
                    command="python",
                    args=["mcp_servers/compliance_mcp_server.py"]
                )
            )
        )
        
        self.kb_mcp_client = MCPClient(
            lambda: stdio_client(
                StdioServerParameters(
                    command="python",
                    args=["mcp_servers/kb_mcp_server.py"]
                )
            )
        )
        
        # Create agent with MCP tools
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup agent with MCP tools."""
        # Context managers for MCP clients
        with self.compliance_mcp_client, self.kb_mcp_client:
            # Get tools from MCP servers
            compliance_tools = self.compliance_mcp_client.list_tools_sync()
            kb_tools = self.kb_mcp_client.list_tools_sync()
            
            # Combine all tools
            all_tools = compliance_tools + kb_tools
            
            # Create agent with all MCP tools
            self.agent = Agent(
                model=self.model,
                tools=all_tools,
                system_prompt="""You are an advanced compliance and document management agent.
                
                You have access to powerful MCP-based tools:
                
                Knowledge Base Tools:
                - retrieve_documents: Search and retrieve from AWS Bedrock KB
                - search_by_metadata: Filter documents by metadata
                - get_document_summary: Get KB statistics
                
                Compliance Tools:
                - check_compliance: Comprehensive compliance checking
                - scan_pii: Detect personally identifiable information
                - validate_gdpr: Check GDPR compliance
                - generate_compliance_report: Create detailed reports
                
                Use these tools effectively to:
                1. Retrieve relevant documents
                2. Check for compliance violations
                3. Generate comprehensive reports
                4. Provide actionable recommendations
                
                Always be thorough and provide clear, actionable insights."""
            )
    
    def process_document_compliance(
        self,
        query: str,
        check_gdpr: bool = True
    ) -> Dict[str, Any]:
        """
        Process document retrieval and compliance checking.
        
        Args:
            query: Search query
            check_gdpr: Whether to include GDPR validation
            
        Returns:
            Combined results
        """
        with self.compliance_mcp_client, self.kb_mcp_client:
            prompt = f"""
            Execute the following workflow:
            
            1. Retrieve documents matching: "{query}"
            2. Check the retrieved content for compliance violations
            3. Scan for PII with masking enabled
            {"4. Validate GDPR compliance" if check_gdpr else ""}
            5. Generate a comprehensive compliance report
            
            Provide a complete analysis with all findings and recommendations.
            """
            
            response = self.agent(prompt)
            
            return {
                'query': query,
                'analysis': response.content,
                'usage': response.usage
            }
    
    def batch_scan_pii(
        self,
        documents: List[str],
        mask: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan multiple documents for PII.
        
        Args:
            documents: List of document contents
            mask: Whether to mask found PII
            
        Returns:
            List of scan results
        """
        results = []
        
        with self.compliance_mcp_client:
            for i, doc in enumerate(documents):
                prompt = f"""
                Scan the following document for PII.
                Enable masking: {mask}
                
                Document {i+1}:
                {doc[:500]}...
                
                Provide detailed PII findings.
                """
                
                response = self.agent(prompt)
                results.append({
                    'document_index': i,
                    'scan_results': response.content
                })
        
        return results
    
    def validate_regulatory_compliance(
        self,
        content: str,
        regulations: List[str] = ["GDPR", "HIPAA", "PCI-DSS"]
    ) -> Dict[str, Any]:
        """
        Validate content against multiple regulations.
        
        Args:
            content: Content to validate
            regulations: List of regulations to check
            
        Returns:
            Validation results
        """
        with self.compliance_mcp_client:
            prompt = f"""
            Validate the following content against these regulations: {', '.join(regulations)}
            
            Content:
            {content[:1000]}...
            
            For each regulation:
            1. Check specific requirements
            2. Identify gaps
            3. Provide recommendations
            
            Use validate_gdpr for GDPR and check_compliance for others.
            """
            
            response = self.agent(prompt)
            
            return {
                'regulations_checked': regulations,
                'validation_results': response.content
            }
    
    def generate_executive_report(
        self,
        search_queries: List[str]
    ) -> str:
        """
        Generate an executive compliance report for multiple searches.
        
        Args:
            search_queries: List of search queries to process
            
        Returns:
            Executive report
        """
        with self.compliance_mcp_client, self.kb_mcp_client:
            prompt = f"""
            Generate an executive compliance report:
            
            1. For each query, retrieve relevant documents:
               {json.dumps(search_queries, indent=2)}
            
            2. Check compliance for all retrieved content
            
            3. Identify common patterns and systemic issues
            
            4. Generate a comprehensive executive report including:
               - Executive summary
               - Key findings across all documents
               - Risk assessment
               - Prioritized recommendations
               - Compliance metrics
            
            Format the report in Markdown for easy reading.
            """
            
            response = self.agent(prompt)
            return response.content