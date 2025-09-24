"""Orchestrator agent managing multiple MCP servers."""

import logging
from typing import Dict, Any, List, Optional
from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from config.settings import settings
import asyncio

logger = logging.getLogger(__name__)

class MultiMCPOrchestrator:
    """Orchestrator managing multiple MCP servers for complex workflows."""
    
    def __init__(self, mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize orchestrator with multiple MCP servers.
        
        Args:
            mcp_servers: Dictionary of MCP server configurations
        """
        # Setup Bedrock model
        self.model = BedrockModel(**settings.get_bedrock_config())
        
        # Default MCP server configurations
        if mcp_servers is None:
            mcp_servers = {
                "compliance": {
                    "command": "python",
                    "args": ["mcp_servers/compliance_mcp_server.py"]
                },
                "knowledge_base": {
                    "command": "python",
                    "args": ["mcp_servers/kb_mcp_server.py"]
                },
                "analytics": {
                    "command": "python",
                    "args": ["mcp_servers/analytics_mcp_server.py"]
                },
                "notifications": {
                    "command": "python",
                    "args": ["mcp_servers/notification_mcp_server.py"]
                }
            }
        
        self.mcp_servers = mcp_servers
        self.mcp_clients = {}
        self._initialize_mcp_clients()
    
    def _initialize_mcp_clients(self):
        """Initialize all MCP clients."""
        for name, config in self.mcp_servers.items():
            try:
                self.mcp_clients[name] = MCPClient(
                    lambda cfg=config: stdio_client(
                        StdioServerParameters(**cfg)
                    )
                )
                logger.info(f"Initialized MCP client: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize MCP client {name}: {e}")
    
    def create_specialized_agent(
        self,
        agent_type: str,
        mcp_servers: List[str]
    ) -> Agent:
        """
        Create a specialized agent with specific MCP servers.
        
        Args:
            agent_type: Type of agent to create
            mcp_servers: List of MCP server names to use
            
        Returns:
            Configured Agent instance
        """
        tools = []
        
        # Collect tools from specified MCP servers
        for server_name in mcp_servers:
            if server_name in self.mcp_clients:
                client = self.mcp_clients[server_name]
                with client:
                    server_tools = client.list_tools_sync()
                    tools.extend(server_tools)
            else:
                logger.warning(f"MCP server '{server_name}' not found")
        
        # Create agent with collected tools
        system_prompts = {
            "compliance": """You are a compliance specialist agent with access to 
                            advanced compliance checking tools. Focus on identifying 
                            violations, generating reports, and providing remediation.""",
            
            "research": """You are a research agent with access to knowledge base 
                          and analytics tools. Focus on retrieving relevant information,
                          analyzing patterns, and providing insights.""",
            
            "audit": """You are an audit agent with comprehensive access to all tools.
                       Perform thorough audits, cross-reference findings, and ensure
                       complete compliance coverage."""
        }
        
        agent = Agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompts.get(agent_type, "You are a helpful assistant.")
        )
        
        return agent
    
    def execute_workflow(
        self,
        workflow_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a predefined workflow using multiple MCP servers.
        
        Args:
            workflow_name: Name of workflow to execute
            parameters: Workflow parameters
            
        Returns:
            Workflow results
        """
        workflows = {
            "full_compliance_audit": self._full_compliance_audit,
            "document_migration": self._document_migration,
            "regulatory_assessment": self._regulatory_assessment,
            "incident_response": self._incident_response
        }
        
        if workflow_name not in workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        return workflows[workflow_name](parameters)
    
    def _full_compliance_audit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full compliance audit workflow."""
        results = {
            'workflow': 'full_compliance_audit',
            'status': 'started',
            'steps': []
        }
        
        # Step 1: Create audit agent with all MCP servers
        audit_agent = self.create_specialized_agent(
            'audit',
            ['compliance', 'knowledge_base', 'analytics']
        )
        
        # Step 2: Retrieve all relevant documents
        with self.mcp_clients['knowledge_base']:
            kb_prompt = f"""
            Use the retrieve_documents tool to get all documents related to:
            - Scope: {params.get('scope', 'all organizational policies')}
            - Max results: 50
            - Min score: 0.3
            
            Execute retrieve_documents tool now.
            """
            kb_response = audit_agent(kb_prompt)
            results['steps'].append({
                'step': 'document_retrieval',
                'status': 'completed',
                'summary': kb_response.content[:200]
            })
        
        # Step 3: Comprehensive compliance check
        with self.mcp_clients['compliance']:
            compliance_prompt = f"""
            Perform these compliance checks in sequence:
            
            1. Use check_compliance tool on all retrieved documents
            2. Use scan_pii tool with masking enabled
            3. Use validate_gdpr tool for GDPR compliance
            
            Execute each tool and compile results.
            """
            compliance_response = audit_agent(compliance_prompt)
            results['steps'].append({
                'step': 'compliance_check',
                'status': 'completed',
                'summary': compliance_response.content[:200]
            })
        
        # Step 4: Analytics and reporting
        if 'analytics' in self.mcp_clients:
            with self.mcp_clients['analytics']:
                analytics_prompt = f"""
                Analyze compliance findings:
                1. Identify patterns and trends
                2. Calculate risk scores
                3. Generate visualizations
                4. Provide predictive insights
                """
                analytics_response = audit_agent(analytics_prompt)
                results['steps'].append({
                    'step': 'analytics',
                    'status': 'completed',
                    'summary': analytics_response.content[:200]
                })
        
        # Step 5: Generate final report
        final_prompt = """
        Generate comprehensive audit report including:
        - Executive summary
        - Detailed findings
        - Risk assessment
        - Recommendations
        - Action items with priorities
        """
        final_response = audit_agent(final_prompt)
        
        results['status'] = 'completed'
        results['final_report'] = final_response.content
        
        return results
    
    def _document_migration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document migration workflow."""
        # Implementation for document migration
        return {
            'workflow': 'document_migration',
            'status': 'not_implemented',
            'message': 'Document migration workflow to be implemented'
        }
    
    def _regulatory_assessment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute regulatory assessment workflow."""
        results = {
            'workflow': 'regulatory_assessment',
            'regulations': params.get('regulations', ['GDPR', 'HIPAA', 'SOX']),
            'assessments': []
        }
        
        compliance_agent = self.create_specialized_agent(
            'compliance',
            ['compliance', 'knowledge_base']
        )
        
        for regulation in results['regulations']:
            with self.mcp_clients['compliance'], self.mcp_clients['knowledge_base']:
                # Step 1: Retrieve relevant docs
                retrieve_prompt = f"""
                Use retrieve_documents tool to find policies related to {regulation}.
                
                Execute the tool now.
                """
                
                docs = compliance_agent(retrieve_prompt)
                
                # Step 2: Check compliance
                check_prompt = f"""
                Use check_compliance tool to assess {regulation} compliance.
                
                Content: {str(docs)}
                
                Execute the tool now.
                """
                
                assessment = compliance_agent(check_prompt)
                
                results['assessments'].append({
                    'regulation': regulation,
                    'assessment': str(assessment)
                })
                
                return results
    
    def _incident_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute incident response workflow."""
        # Implementation for incident response
        return {
            'workflow': 'incident_response',
            'status': 'not_implemented',
            'message': 'Incident response workflow to be implemented'
        }
    
    async def execute_parallel_workflows(
        self,
        workflows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple workflows in parallel.
        
        Args:
            workflows: List of workflow configurations
            
        Returns:
            List of workflow results
        """
        tasks = []
        for workflow in workflows:
            task = asyncio.create_task(
                asyncio.to_thread(
                    self.execute_workflow,
                    workflow['name'],
                    workflow.get('parameters', {})
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results