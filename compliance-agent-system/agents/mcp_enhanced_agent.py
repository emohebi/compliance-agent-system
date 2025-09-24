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
    
    def process_document_compliance(self, query: str, check_gdpr: bool = True) -> Dict[str, Any]:
        with self.compliance_mcp_client, self.kb_mcp_client:
            # Get tools
            all_tools = (self.compliance_mcp_client.list_tools_sync() + 
                        self.kb_mcp_client.list_tools_sync())
            
            # Create agent
            agent = Agent(
                model=self.model,
                tools=all_tools,
                system_prompt="You have MCP tools for compliance and KB access."
            )
            
            # Step 1: Retrieve documents
            retrieve_prompt = f"""
            Use the retrieve_documents tool to find documents matching: "{query}"
            
            Execute retrieve_documents tool now.
            """
            
            retrieval_response = agent(retrieve_prompt)
            
            # Step 2: Check compliance
            compliance_prompt = f"""
            Use the check_compliance tool to analyze the retrieved content.
            
            Content to check: {str(retrieval_response)}
            
            Execute check_compliance tool now.
            """
            
            compliance_response = agent(compliance_prompt)
            
            # Step 3: GDPR check if needed
            if check_gdpr:
                gdpr_prompt = f"""
                Use the validate_gdpr tool on the content.
                
                Execute validate_gdpr tool now.
                """
                
                gdpr_response = agent(gdpr_prompt)
            
            # Step 4: Generate report
            report_prompt = f"""
            Use generate_compliance_report tool with all the findings.
            
            Execute the tool now.
            """
            
            report_response = agent(report_prompt)
            
            return {
                'query': query,
                'analysis': str(report_response),
                'usage': report_response.metrics.accumulated_usage if hasattr(report_response, 'metrics') else {}
            }
    
    def batch_scan_pii(self, documents: List[str], mask: bool = True) -> List[Dict[str, Any]]:
        results = []
        
        with self.compliance_mcp_client:
            # Get tools and create agent
            tools = self.compliance_mcp_client.list_tools_sync()
            agent = Agent(model=self.model, tools=tools)
            
            for i, doc in enumerate(documents):
                # Step 1: Tool invocation
                tool_prompt = f"""
                Use the scan_pii tool on this document.
                
                Parameters:
                - content: {doc[:300]}...
                - mask: {mask}
                
                Execute scan_pii tool now.
                """
                
                tool_response = agent(tool_prompt)
                
                results.append({
                    'document_index': i,
                    'scan_results': str(tool_response)
                })
        
        return results
    
    def validate_regulatory_compliance(self, content: str, regulations: List[str] = ["GDPR", "HIPAA", "PCI-DSS"]) -> Dict[str, Any]:
        with self.compliance_mcp_client:
            tools = self.compliance_mcp_client.list_tools_sync()
            agent = Agent(model=self.model, tools=tools)
            
            validation_results = {}
            
            for regulation in regulations:
                if regulation == "GDPR":
                    tool_prompt = f"""
                    Use the validate_gdpr tool to check GDPR compliance.
                    
                    Content: {content[:500]}...
                    
                    Execute validate_gdpr tool now.
                    """
                else:
                    tool_prompt = f"""
                    Use the check_compliance tool for {regulation} requirements.
                    
                    Content: {content[:500]}...
                    Regulation: {regulation}
                    
                    Execute check_compliance tool now.
                    """
                
                tool_response = agent(tool_prompt)
                validation_results[regulation] = str(tool_response)
            
            return {
                'regulations_checked': regulations,
                'validation_results': validation_results
            }
    
    def generate_executive_report(self, search_queries: List[str]) -> str:
        with self.compliance_mcp_client, self.kb_mcp_client:
            all_tools = (self.compliance_mcp_client.list_tools_sync() + 
                        self.kb_mcp_client.list_tools_sync())
            
            agent = Agent(model=self.model, tools=all_tools)
            
            all_results = []
            
            # Step 1: Retrieve documents for each query
            for query in search_queries:
                retrieve_prompt = f"""
                Use retrieve_documents tool to search for: "{query}"
                
                Execute the tool now.
                """
                
                retrieval = agent(retrieve_prompt)
                
                # Step 2: Check compliance for retrieved content
                compliance_prompt = f"""
                Use check_compliance tool on the retrieved content.
                
                Execute the tool now.
                """
                
                compliance = agent(compliance_prompt)
                
                all_results.append({
                    'query': query,
                    'retrieval': str(retrieval),
                    'compliance': str(compliance)
                })
            
            # Step 3: Generate final report
            report_prompt = f"""
            Use generate_compliance_report tool to create an executive report.
            
            Findings: {json.dumps(all_results, indent=2)}
            
            Execute the tool to create comprehensive report.
            """
            
            report_response = agent(report_prompt)
            
            return str(report_response)