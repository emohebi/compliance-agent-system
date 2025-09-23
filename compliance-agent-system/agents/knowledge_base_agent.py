"""Knowledge Base Agent for reading and managing KB documents."""

import logging
from typing import Dict, Any, Optional, List
from strands import Agent
from strands.models import BedrockModel
from config.settings import settings
from tools.knowledge_base_tools import kb_tools

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
        """
        Retrieve documents from the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            min_score: Minimum relevance score
            
        Returns:
            Retrieved documents and metadata
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        prompt = f"""
        Retrieve relevant documents for the following query:
        Query: {query}
        
        Parameters:
        - Maximum results: {max_results}
        - Minimum score: {min_score}
        
        Provide a summary of the most relevant findings.
        """
        
        response = self.agent(prompt)
        return {
            'query': query,
            'response': response.content,
            'usage': response.usage
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
            'response': response.content
        }
    
    def search_and_summarize(
        self,
        topic: str,
        context: Optional[str] = None
    ) -> str:
        """
        Search for a topic and provide a comprehensive summary.
        
        Args:
            topic: Topic to search for
            context: Additional context for the search
            
        Returns:
            Summarized findings
        """
        logger.info(f"Searching and summarizing topic: {topic}")
        
        prompt = f"""
        Search the knowledge base for information about: {topic}
        
        {f'Additional context: {context}' if context else ''}
        
        Provide a comprehensive summary of all relevant findings, including:
        1. Key information points
        2. Document sources
        3. Confidence scores
        4. Any gaps in information
        """
        
        response = self.agent(prompt)
        return response.content
    
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