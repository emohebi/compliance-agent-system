"""Main orchestrator agent that coordinates KB and compliance agents."""

import logging
from typing import Dict, Any, List, Optional
from strands import Agent
from strands.models import BedrockModel
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.compliance_agent import ComplianceAgent
from models.compliance_rules import ActionPlan
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
            if not compliance_results.is_compliant and compliance_results.violations:
                logger.info("Step 3: Generating remediation recommendations")
                try:
                    remediation = self.compliance_agent.suggest_remediation(
                        compliance_results.violations
                    )
                    workflow_results['steps'].append({
                        'step': 'remediation_suggestions',
                        'status': 'completed',
                        'results': remediation
                    })
                except Exception as e:
                    logger.error(f"Remediation generation failed: {e}")
                    workflow_results['steps'].append({
                        'step': 'remediation_suggestions',
                        'status': 'failed',
                        'results': str(e)
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
            queries: List of search queries or document contents
            
        Returns:
            Batch compliance results
        """
        logger.info(f"Processing batch compliance check for {len(queries)} items")
        
        # Determine if queries are search queries or documents
        are_documents = any(len(q) > 100 for q in queries)  # Simple heuristic
        
        if are_documents:
            # Treat as documents to check
            documents = [
                {'content': q, 'metadata': {'index': i}}
                for i, q in enumerate(queries)
            ]
        else:
            # Treat as search queries - retrieve documents first
            documents = []
            for query in queries:
                kb_results = self.kb_agent.retrieve_documents(query)
                if kb_results.get('response'):
                    documents.append({
                        'content': kb_results['response'],
                        'metadata': {'query': query}
                    })
        
        if not documents:
            return {
                'results': {
                    'total_documents': 0,
                    'compliant_documents': 0,
                    'non_compliant_documents': 0,
                    'compliance_rate': 0
                },
                'report': 'No documents found to check.'
            }
        
        # Check batch compliance
        batch_results = self.compliance_agent.check_batch(documents)
        
        # Generate report if we have results
        if batch_results.get('results', {}).get('total_documents', 0) > 0:
            try:
                compliance_results = []
                for doc in documents[:5]:  # Limit to avoid token overflow
                    result = self.compliance_agent.check_document(doc['content'])
                    compliance_results.append(result)
                
                report = self.compliance_agent.generate_report(compliance_results)
                batch_results['report'] = report
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                batch_results['report'] = "Report generation failed."
        
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
        
        # Step 1: Analyze request and create action plan
        analysis_prompt = f"""
        Analyze this user request and determine the best action:
        
        User Request: "{user_request}"
        
        Available actions:
        - search_kb: Search knowledge base for documents
        - check_compliance: Check a document for compliance violations
        - run_workflow: Run complete compliance workflow (search + check)
        - add_document: Add new document to knowledge base
        - view_stats: View knowledge base statistics
        - batch_check: Check multiple documents/queries for compliance
        - generate_report: Generate a compliance report
        - keyword_compliance: Search knowledge base for documents and Check if documents contain required keywords so DO NOT need to call search_kb separately
        - unknown: Request doesn't match any specific action
        
        Determine:
        1. Which action_type best matches the request
        2. What parameters to extract (e.g., query text, document content)
        3. Your reasoning for this choice
        4. Any additional inputs needed from the user
        
        Examples:
        - "search for GDPR policies" → action_type: "search_kb", parameters: {{"query": "GDPR policies"}}
        - "check if this text is compliant: [text]" → action_type: "check_compliance", parameters: {{"content": "[text]"}}
        - "how many documents do we have?" → action_type: "view_stats"
        """
        
        try:
            action_plan = self.agent.structured_output(
                ActionPlan,
                analysis_prompt
            )
            
            logger.info(f"Action plan: {action_plan.action_type} with params: {action_plan.parameters}")
            
            # Step 2: Check if we need more input from user
            if action_plan.requires_input:
                return f"I need more information to help you:\n" + \
                    "\n".join(f"- {inp}" for inp in action_plan.requires_input) + \
                    f"\n\nReason: {action_plan.reasoning}"
            
            # Step 3: Execute based on action type
            if action_plan.action_type == "search_kb":
                query = action_plan.parameters.get('query', user_request)
                results = self.kb_agent.retrieve_documents(
                    query=query,
                    max_results=action_plan.parameters.get('max_results', 5),
                    min_score=action_plan.parameters.get('min_score', 0.4)
                )
                
                # Format response
                response = f"**Knowledge Base Search Results for: '{query}'**\n\n"
                if results.get('documents'):
                    response += f"Found {results.get('total', len(results['documents']))} relevant documents.\n\n"
                    response += "**Key Insights:**\n"
                    for insight in results.get('insights', [])[:5]:
                        response += f"• {insight}\n"
                    response += f"\n**Summary:** {results.get('response', 'No summary available')}"
                else:
                    response += "No relevant documents found."
                
                return response
                
            # In the check_compliance action
            elif action_plan.action_type == "check_compliance":
                content = action_plan.parameters.get('content', '')
                if not content:
                    return "Please provide the document content you want to check for compliance."
                
                result = self.compliance_agent.check_document(
                    document_content=content,
                    document_metadata={'source': 'interactive_session'}
                )
                
                response = f"**Compliance Check Results**\n\n"
                response += f"• **Compliant:** {'✅ Yes' if result.is_compliant else '❌ No'}\n"
                response += f"• **Score:** {result.score:.2f}/1.00\n"
                
                if result.violations:
                    response += f"\n**Violations Found ({len(result.violations)}):**\n"
                    for v in result.violations[:5]:
                        # Handle both string and dict violations
                        if isinstance(v, str):
                            response += f"• {v}\n"
                        else:
                            response += f"• {v.get('description', 'Unknown violation')}\n"
                
                if result.warnings:
                    response += f"\n**Warnings ({len(result.warnings)}):**\n"
                    for w in result.warnings[:3]:
                        # Handle both string and dict warnings
                        if isinstance(w, str):
                            response += f"• {w}\n"
                        else:
                            response += f"• {w.get('description', 'Unknown warning')}\n"
                
                if not result.is_compliant and result.violations:
                    try:
                        remediation = self.compliance_agent.suggest_remediation(result.violations)
                        response += f"\n**Remediation Suggestions:**\n{remediation[:500]}"
                    except Exception as e:
                        logger.error(f"Failed to generate remediation: {e}")
                        response += "\n**Remediation:** Please review violations and take appropriate action."
                
                return response
                
            elif action_plan.action_type == "run_workflow":
                query = action_plan.parameters.get('query', user_request)
                check_compliance = action_plan.parameters.get('check_compliance', True)
                
                results = self.process_compliance_workflow(
                    query=query,
                    check_compliance=check_compliance
                )
                
                response = f"**Compliance Workflow Results**\n\n"
                response += f"Query: '{query}'\n"
                response += f"Steps Completed: {len(results.get('steps', []))}\n\n"
                
                for i, step in enumerate(results.get('steps', []), 1):
                    response += f"**Step {i}: {step['step']}**\n"
                    response += f"Status: ✅ {step['status']}\n\n"
                
                if 'summary' in results:
                    response += f"**Summary:**\n{results['summary']}"
                
                return response
            
            elif action_plan.action_type == "keyword_compliance":
                keywords = action_plan.parameters.get('keywords', [])
                match_type = action_plan.parameters.get('match_type', 'all')
                
                if not keywords:
                    # Try to extract from user request
                    import re
                    # Look for quoted words or comma-separated list
                    quoted = re.findall(r'"([^"]+)"', user_request)
                    if quoted:
                        keywords = quoted
                    else:
                        # Ask for clarification
                        return "Please specify the keywords to check. For example:\n" \
                            'Check if all documents contain "privacy policy", "data protection", "consent"'
                
                results = self.check_kb_keyword_compliance(
                    required_keywords=keywords,
                    match_type=match_type
                )
                
                response = f"**Keyword Compliance Check Results**\n\n"
                response += f"Required Keywords: {', '.join(keywords)}\n"
                response += f"Match Type: Must have {match_type} keywords\n\n"
                
                comp_results = results.get('compliance_results', {})
                response += f"📊 **Statistics:**\n"
                response += f"• Total Documents: {comp_results.get('total_documents', 0)}\n"
                response += f"• Compliant: {comp_results.get('compliant_count', 0)}\n"
                response += f"• Non-Compliant: {comp_results.get('non_compliant_count', 0)}\n"
                response += f"• Compliance Rate: {comp_results.get('compliance_rate', 0)*100:.1f}%\n\n"
                
                if comp_results.get('non_compliant_details'):
                    response += "❌ **Non-Compliant Documents:**\n"
                    for doc in comp_results['non_compliant_details'][:10]:  # Show first 10
                        response += f"• Document: {doc.get('document_id', 'Unknown')}\n"
                        response += f"  Missing: {', '.join(doc.get('missing_keywords', []))}\n"
                
                if results.get('summary'):
                    response += f"\n**Summary:**\n{results['summary']}"
                
                return response
                
            elif action_plan.action_type == "add_document":
                if not settings.USE_LOCAL_KNOWLEDGE_BASE:
                    return "Adding documents is only available in local knowledge base mode."
                
                content = action_plan.parameters.get('content', '')
                if not content:
                    return "Please provide the document content you want to add."
                
                metadata = action_plan.parameters.get('metadata', {})
                result = self.kb_agent.store_document(
                    content=content,
                    metadata=metadata
                )
                
                return f"✅ Document stored successfully!\n{result.get('response', '')}"
                
            elif action_plan.action_type == "view_stats":
                stats_prompt = "Provide knowledge base statistics and summary"
                stats_response = self.kb_agent.agent(stats_prompt)
                
                return f"**Knowledge Base Statistics**\n\n{str(stats_response)}"
                
            elif action_plan.action_type == "batch_check":
                queries_or_docs = action_plan.parameters.get('items', [])
                if not queries_or_docs:
                    # Try to extract from user request
                    items = [line.strip() for line in user_request.split('\n') if line.strip()]
                    if len(items) > 1:
                        queries_or_docs = items[1:]  # Skip the first line (instruction)
                
                if not queries_or_docs:
                    return "Please provide multiple queries or documents to check (one per line)."
                
                results = self.batch_compliance_check(queries_or_docs)
                
                response = f"**Batch Compliance Check Results**\n\n"
                response += f"Processed {len(queries_or_docs)} items\n\n"
                
                if 'results' in results:
                    batch_data = results['results']
                    response += f"• Total documents: {batch_data.get('total_documents', 0)}\n"
                    response += f"• Compliant: {batch_data.get('compliant_documents', 0)}\n"
                    response += f"• Non-compliant: {batch_data.get('non_compliant_documents', 0)}\n"
                    response += f"• Compliance rate: {batch_data.get('compliance_rate', 0)*100:.1f}%\n"
                
                if 'report' in results:
                    response += f"\n**Report Preview:**\n{results['report'][:500]}..."
                
                return response
                
            elif action_plan.action_type == "generate_report":
                scope = action_plan.parameters.get('scope', 'all recent compliance checks')
                
                # Generate a report based on recent activity
                report_prompt = f"""
                Generate a compliance report for: {scope}
                
                Include:
                - Executive summary
                - Key findings
                - Recommendations
                """
                
                report_response = self.agent(report_prompt)
                
                return f"**Compliance Report**\n\n{str(report_response)}"
                
            else:  # action_type == "unknown" or unhandled
                # Fall back to general agent response
                general_response = self.agent(user_request)
                return str(general_response)
                
        except Exception as e:
            logger.error(f"Error in interactive session: {e}")
            
            # Fallback to simple keyword matching if structured output fails
            request_lower = user_request.lower()
            
            if "search" in request_lower or "find" in request_lower or "retrieve" in request_lower:
                # Extract search query
                query = user_request.replace("search", "").replace("find", "").replace("retrieve", "").strip()
                if not query:
                    return "What would you like to search for?"
                results = self.kb_agent.search_and_summarize(query)
                return results
                
            elif "check" in request_lower or "compliance" in request_lower or "scan" in request_lower:
                return "Please provide the document content you want to check for compliance."
                
            elif "stat" in request_lower or "count" in request_lower or "how many" in request_lower:
                stats_response = self.kb_agent.agent("Get knowledge base statistics")
                return f"**Knowledge Base Statistics**\n\n{str(stats_response)}"
                
            else:
                # Use general agent response
                response = self.agent(user_request)
                return str(response)
            
    def check_kb_keyword_compliance(
        self,
        required_keywords: List[str],
        match_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Check if all documents in knowledge base contain required keywords.
        
        Args:
            required_keywords: List of required keywords
            match_type: "all" or "any"
            
        Returns:
            Compliance results
        """
        logger.info(f"Checking KB keyword compliance for: {required_keywords}")
        
        workflow_results = {
            'workflow_type': 'keyword_compliance',
            'required_keywords': required_keywords,
            'match_type': match_type,
            'steps': []
        }
        
        try:
            # Step 1: Get all documents from KB
            logger.info("Step 1: Retrieving all documents from knowledge base")
            
            # First, get the list of documents
            all_docs = []
            batch_size = 20
            
            for i in range(0, 100, batch_size):  # Retrieve up to 100 documents
                batch_results = self.kb_agent.retrieve_documents(
                    query="*",  # Or use a broad query
                    max_results=batch_size
                )
                
                if batch_results.get('documents'):
                    all_docs.extend(batch_results['documents'])
                
                if len(batch_results.get('documents', [])) < batch_size:
                    break  # No more documents
            
            workflow_results['steps'].append({
                'step': 'document_retrieval',
                'status': 'completed',
                'documents_retrieved': len(all_docs)
            })
            
            # Step 2: Check keyword compliance
            logger.info("Step 2: Checking keyword compliance")
            
            # Use the compliance agent's keyword check
            compliance_results = self.compliance_agent.check_keyword_compliance_kb(
                documents=all_docs,
                required_keywords=required_keywords,
                match_type=match_type
            )
            
            workflow_results['steps'].append({
                'step': 'keyword_compliance_check',
                'status': 'completed',
                'results': compliance_results
            })
            
            # Step 3: Generate summary report
            logger.info("Step 3: Generating summary report")
            
            summary_prompt = f"""
            Create a summary report for keyword compliance check:
            
            Required Keywords: {', '.join(required_keywords)}
            Match Type: {match_type}
            Total Documents: {compliance_results.get('total_documents', 0)}
            Compliant Documents: {compliance_results.get('compliant_count', 0)}
            Non-Compliant Documents: {compliance_results.get('non_compliant_count', 0)}
            
            Provide:
            1. Executive summary
            2. List of non-compliant documents with missing keywords
            3. Action items to achieve full compliance
            """
            
            summary = self.agent(summary_prompt)
            
            workflow_results['summary'] = str(summary)
            workflow_results['compliance_results'] = compliance_results
            
        except Exception as e:
            logger.error(f"Keyword compliance check failed: {e}")
            workflow_results['error'] = str(e)
            workflow_results['summary'] = f"Workflow failed: {e}"
        
        return workflow_results
    
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