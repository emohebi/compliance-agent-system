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
            system_prompt="""You are a Knowledge Base specialist agent.
            Your role is to:
            1. Retrieve relevant documents from the knowledge base
            2. Store new documents when needed
            3. Provide accurate information from stored documents
            4. Maintain document organization and metadata
            
            When retrieving documents:
            - Use semantic search to find the most relevant content
            - Apply appropriate score thresholds
            - Summarize findings clearly
            
            Always provide sources and confidence scores for retrieved information."""
        )
    
    def retrieve_documents(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.4
    ) -> Dict[str, Any]:
        """Retrieve documents from the knowledge base."""
        logger.info(f"Retrieving documents for query: {query}")
        
        prompt = f"""
        Search for and retrieve documents related to: {query}
        
        Requirements:
        - Find up to {max_results} most relevant documents
        - Focus on documents with relevance score above {min_score}
        - Provide a comprehensive summary of findings
        - Identify key insights from the documents
        """
        
        result = self.agent.structured_output(
            DocumentSearchResult,
            prompt
        )
        
        return {
            'query': query,
            'response': result.summary,
            'documents': result.relevant_documents,
            'insights': result.key_insights,
            'total': result.total_found
        }
        
    def store_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a document in the knowledge base.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Storage status
        """
        logger.info("Storing document in knowledge base")
        
        prompt = f"""
        Store the following document in the knowledge base:
        
        Content: {content}
        Metadata: {metadata}
        
        Ensure proper indexing and categorization.
        """
        
        response = self.agent(prompt)
        return {
            'status': 'stored',
            'response': response.message
        }
    
    def search_and_summarize(
        self,
        topic: str,
        context: Optional[str] = None
    ) -> str:
        """Search for a topic and provide a comprehensive summary."""
        logger.info(f"Searching and summarizing topic: {topic}")
        
        prompt = f"""
        Search the knowledge base for information about: {topic}
        {f'Additional context: {context}' if context else ''}
        
        Provide a comprehensive analysis including:
        1. Main topics and themes
        2. Key points and findings
        3. Any gaps in available information
        4. Confidence level in the summary
        """
        
        summary = self.agent.structured_output(
            DocumentSummary,
            prompt
        )
        
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