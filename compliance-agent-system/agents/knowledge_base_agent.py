"""Knowledge Base Agent for reading and managing KB documents."""

import logging
from typing import Dict, Any, Optional, List
from strands import Agent
from strands.models import BedrockModel
from config.settings import settings
from tools.knowledge_base_tools import kb_tools
from models.document_models import Document, DocumentBatch, DocumentSearchResult, DocumentSummary

logger = logging.getLogger(__name__)

class KnowledgeBaseAgent:
    """Agent for interacting with AWS Bedrock Knowledge Base."""
    
    def __init__(self):
        """Initialize the Knowledge Base agent."""
        # Setup Bedrock model
        self.model = BedrockModel(**settings.get_bedrock_config())
        
        # Create agent with KB tools
        self.agent = Agent(
            model=self.model,
            tools=[
                kb_tools.retrieve_from_knowledge_base,
                kb_tools.store_in_knowledge_base,
                kb_tools.list_knowledge_base_documents
            ],
            system_prompt="""You are a Knowledge Base specialist.
                IMPORTANT: You have access to these tools:
                - retrieve_from_knowledge_base: Use this to search documents
                - store_in_knowledge_base: Use this to store documents
                - list_knowledge_base_documents: Use this to list documents
                
                Always use these tools when asked to search, store, or list documents.
                """
            )
    
    def retrieve_documents(self, query: str, max_results: int = 10, min_score: float = 0.4) -> Dict[str, Any]:
        logger.info(f"Retrieving documents for query: {query}")
        
        # Step 1: Tool invocation
        tool_prompt = f"""
        Use the retrieve_from_knowledge_base tool with these exact parameters:
        - query: "{query}"
        - max_results: {max_results}
        - min_score: {min_score}
        
        Execute retrieve_from_knowledge_base tool now.
        """
        
        tool_response = self.agent(tool_prompt)
        
        # Step 2: Format results (optional)
        format_prompt = f"""
        Based on these search results: {str(tool_response)}
        
        Provide:
        - List of relevant document contents
        - Total number found
        - Summary of findings
        - Key insights extracted
        """
        
        result = self.agent.structured_output(
            DocumentSearchResult,
            format_prompt
        )
        
        return {
            'query': query,
            'response': result.summary,
            'documents': result.relevant_documents,
            'insights': result.key_insights,
            'total': result.total_found
        }
            
    def store_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Storing document in knowledge base")
        
        # Tool invocation only (no structured output needed)
        tool_prompt = f"""
        Use the store_in_knowledge_base tool to store this document.
        
        Parameters:
        - content: {content[:200]}...
        - metadata: {metadata}
        
        Execute store_in_knowledge_base tool now.
        """
        
        tool_response = self.agent(tool_prompt)
        
        return {
            'status': 'stored',
            'response': str(tool_response)
        }
    
    def search_and_summarize(self, topic: str, context: Optional[str] = None) -> str:
        logger.info(f"Searching and summarizing topic: {topic}")
        
        # Step 1: Tool invocation
        tool_prompt = f"""
        Use retrieve_from_knowledge_base tool to search for: "{topic}"
        {f'Additional context: {context}' if context else ''}
        
        Execute the tool with appropriate parameters.
        """
        
        tool_response = self.agent(tool_prompt)
        
        # Step 2: Create summary
        format_prompt = f"""
        Based on these search results: {str(tool_response)}
        
        Create a comprehensive summary including:
        - Main topics covered
        - Key points found
        - Information gaps
        - Confidence in the summary
        """
        
        summary = self.agent.structured_output(
            DocumentSummary,
            format_prompt
        )
        
        # Format and return...
        
        # Format the summary
        formatted_summary = f"""
            ## Summary: {topic}

            ### Main Topics
            {chr(10).join(f'- {t}' for t in summary.main_topics)}

            ### Key Points
            {chr(10).join(f'- {p}' for p in summary.key_points)}

            ### Information Gaps
            {chr(10).join(f'- {g}' for g in summary.gaps) if summary.gaps else 'None identified'}

            **Confidence Score:** {summary.confidence_score:.1%}
            """
        
        return formatted_summary
    
    def batch_retrieve(
        self,
        queries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of results for each query
        """
        results = []
        for query in queries:
            result = self.retrieve_documents(query)
            results.append(result)
        return results