"""Main orchestrator agent that coordinates KB and compliance agents."""

import logging
from typing import Dict, Any, List, Optional
from strands import Agent
from strands.models import BedrockModel
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.compliance_agent import ComplianceAgent
from config.settings import settings

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """Main orchestrator agent that coordinates sub-agents."""
    
    def __init__(self):
        """Initialize the orchestrator with sub-agents."""
        # Initialize sub-agents
        self.kb_agent = KnowledgeBaseAgent()
        self.compliance_agent = ComplianceAgent()
        
        # Setup Bedrock model for orchestration
        self.model = BedrockModel(**settings.get_bedrock_config())
        
        # Create orchestrator agent
        self.agent = Agent(
            model=self.model,
            system_prompt="""You are the main orchestrator agent.
            You coordinate between:
            1. Knowledge Base Agent - for document retrieval and storage
            2. Compliance Agent - for compliance checking and reporting
            
            Your responsibilities:
            - Understand user requests and delegate to appropriate agents
            - Coordinate multi-step workflows
            - Synthesize results from multiple agents
            - Ensure quality and completeness of responses
            
            Always think step-by-step about the best approach to fulfill requests."""
        )
    
    def process_compliance_workflow(
        self,
        query: str,
        check_compliance: bool = True
    ) -> Dict[str, Any]:
        """
        Process a complete workflow: retrieve documents and check compliance.
        
        Args:
            query: Search query for documents
            check_compliance: Whether to check compliance
            
        Returns:
            Workflow results including documents and compliance status
        """
        logger.info(f"Processing compliance workflow for query: {query}")
        
        workflow_results = {
            'query': query,
            'steps': []
        }
        
        # Step 1: Retrieve documents from knowledge base
        logger.info("Step 1: Retrieving documents from knowledge base")
        kb_results = self.kb_agent.retrieve_documents(query)
        workflow_results['steps'].append({
            'step': 'document_retrieval',
            'status': 'completed',
            'results': kb_results
        })
        
        # Step 2: Check compliance if requested
        if check_compliance and kb_results.get('response'):
            logger.info("Step 2: Checking compliance")
            compliance_results = self.compliance_agent.check_document(
                document_content=kb_results['response'],
                document_metadata={'source': 'knowledge_base', 'query': query}
            )
            workflow_results['steps'].append({
                'step': 'compliance_check',
                'status': 'completed',
                'results': compliance_results.dict()
            })
            
            # Step 3: Generate recommendations if violations found
            if not compliance_results.is_compliant:
                logger.info("Step 3: Generating remediation recommendations")
                remediation = self.compliance_agent.suggest_remediation(
                    compliance_results.violations
                )
                workflow_results['steps'].append({
                    'step': 'remediation_suggestions',
                    'status': 'completed',
                    'results': remediation
                })
        
        # Generate summary
        workflow_results['summary'] = self._generate_workflow_summary(workflow_results)
        
        return workflow_results
    
    def batch_compliance_check(
        self,
        queries: List[str]
    ) -> Dict[str, Any]:
        """
        Perform batch compliance checking on multiple queries.
        
        Args:
            queries: List of search queries
            
        Returns:
            Batch compliance results
        """
        logger.info(f"Processing batch compliance check for {len(queries)} queries")
        
        documents = []
        for query in queries:
            kb_results = self.kb_agent.retrieve_documents(query)
            if kb_results.get('response'):
                documents.append({
                    'content': kb_results['response'],
                    'metadata': {'query': query}
                })
        
        # Check batch compliance
        batch_results = self.compliance_agent.check_batch(documents)
        
        # Generate report
        if batch_results.get('results', {}).get('results'):
            compliance_results = [
                self.compliance_agent.check_document(doc['content'])
                for doc in documents
            ]
            report = self.compliance_agent.generate_report(compliance_results)
            batch_results['report'] = report
        
        return batch_results
    
    def interactive_session(
        self,
        user_request: str
    ) -> str:
        """
        Handle interactive user requests with intelligent routing.
        
        Args:
            user_request: Natural language user request
            
        Returns:
            Response to user request
        """
        logger.info(f"Processing user request: {user_request}")
        
        # Analyze request and determine action plan
        analysis_prompt = f"""
        Analyze the following user request and determine the best approach:
        
        User Request: {user_request}
        
        Available capabilities:
        1. Search and retrieve documents from knowledge base
        2. Store new documents in knowledge base
        3. Check documents for compliance violations
        4. Generate compliance reports
        5. Provide remediation recommendations
        
        Determine:
        - What actions are needed
        - In what order
        - What information to gather
        """
        
        analysis = self.agent(analysis_prompt)
        
        # Execute based on analysis
        # This is a simplified version - in production, you'd parse the analysis
        # and execute specific actions
        
        if "compliance" in user_request.lower():
            if "check" in user_request.lower() or "scan" in user_request.lower():
                # Extract what to check from the request
                results = self.process_compliance_workflow(
                    query=user_request,
                    check_compliance=True
                )
                return self._format_results(results)
        
        elif "retrieve" in user_request.lower() or "search" in user_request.lower():
            results = self.kb_agent.search_and_summarize(user_request)
            return results
        
        else:
            # Default: Use the orchestrator agent to handle the request
            response = self.agent(user_request)
            return response.message
    
    def _generate_workflow_summary(
        self,
        workflow_results: Dict[str, Any]
    ) -> str:
        """Generate a summary of workflow results."""
        import json

        prompt = f"""
        Summarize the following workflow results in a clear, concise manner:

        Workflow Results: {json.dumps(workflow_results, indent=2, default=str)}

        Include:
        1. What was accomplished
        2. Key findings
        3. Any issues or violations found
        4. Recommended next steps
        """
        
        response = self.agent(prompt)
        return response.message
    
    def _format_results(
        self,
        results: Dict[str, Any]
    ) -> str:
        """Format results for presentation."""
        prompt = f"""
        Format the following results for clear presentation to the user:
        
        Results: {results}
        
        Make it:
        - Easy to read
        - Well-structured
        - Highlight important information
        - Include actionable insights
        """
        
        response = self.agent(prompt)
        return response.message