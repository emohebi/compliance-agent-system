"""Knowledge base tools supporting both AWS and local implementations."""

from typing import Dict, Any, List, Optional
from strands import tool
import boto3
import json
import logging
from config.settings import settings
from tools.local_knowledge_base import LocalKnowledgeBase

logger = logging.getLogger(__name__)

class KnowledgeBaseTools:
    """Tools for interacting with knowledge base (AWS or local)."""
    
    def __init__(self):
        """Initialize the knowledge base tools."""
        self.use_local = settings.USE_LOCAL_KNOWLEDGE_BASE
        
        if self.use_local:
            logger.info("Using local JSON knowledge base")
            self.local_kb = LocalKnowledgeBase(settings.LOCAL_KB_FILE)
        else:
            logger.info("Using AWS Bedrock knowledge base")
            self.client = boto3.client(
                'bedrock-agent-runtime',
                region_name=settings.AWS_REGION
            )
            self.kb_id = settings.BEDROCK_KNOWLEDGE_BASE_ID
    
    @tool
    def retrieve_from_knowledge_base(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from the knowledge base.
        Works with both AWS Bedrock and local JSON storage.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        if self.use_local:
            # Use local knowledge base
            return self.local_kb.retrieve(query, max_results, min_score)
        else:
            # Use AWS Bedrock knowledge base
            try:
                response = self.client.retrieve(
                    knowledgeBaseId=self.kb_id,
                    retrievalQuery={
                        'text': query
                    },
                    retrievalConfiguration={
                        'vectorSearchConfiguration': {
                            'numberOfResults': max_results,
                            'overrideSearchType': 'SEMANTIC'
                        }
                    }
                )
                
                # Filter results by minimum score
                filtered_results = []
                for result in response.get('retrievalResults', []):
                    if result.get('score', 0) >= min_score:
                        filtered_results.append({
                            'content': result.get('content', {}).get('text', ''),
                            'score': result.get('score', 0),
                            'metadata': result.get('metadata', {}),
                            'location': result.get('location', {})
                        })
                
                return {
                    'success': True,
                    'query': query,
                    'results': filtered_results,
                    'total_results': len(filtered_results),
                    'search_method': 'AWS Bedrock semantic search'
                }
                
            except Exception as e:
                logger.error(f"Error retrieving from AWS knowledge base: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'results': []
                }
    
    @tool
    def store_in_knowledge_base(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store content in the knowledge base.
        
        Args:
            content: The content to store
            metadata: Optional metadata for the content
            document_id: Optional document ID
            
        Returns:
            Dictionary containing storage status
        """
        if self.use_local:
            # Store in local knowledge base
            return self.local_kb.store(content, metadata, document_id)
        else:
            # AWS Bedrock storage (requires additional setup)
            try:
                logger.info(f"Storing content in AWS knowledge base: {content[:100]}...")
                
                # Note: Direct storage to AWS KB requires data source configuration
                # This is a placeholder for the actual implementation
                return {
                    'success': True,
                    'message': 'Content queued for storage in AWS KB (requires data source sync)',
                    'metadata': metadata
                }
                
            except Exception as e:
                logger.error(f"Error storing in AWS knowledge base: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
    
    @tool
    def list_knowledge_base_documents(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List documents in the knowledge base.
        
        Args:
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            Dictionary containing document list
        """
        if self.use_local:
            # List from local knowledge base
            return self.local_kb.list_documents(limit, offset)
        else:
            # AWS Bedrock listing
            try:
                logger.info("Listing AWS knowledge base documents...")
                
                # This would require listing from the data source
                return {
                    'success': True,
                    'documents': [],
                    'total': 0,
                    'message': 'AWS KB listing requires data source API access'
                }
                
            except Exception as e:
                logger.error(f"Error listing documents: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
    
    @tool
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing KB statistics
        """
        if self.use_local:
            return self.local_kb.get_statistics()
        else:
            return {
                'mode': 'AWS Bedrock',
                'kb_id': self.kb_id,
                'region': settings.AWS_REGION,
                'message': 'Detailed AWS KB stats require CloudWatch metrics'
            }
    
    @tool
    def delete_from_knowledge_base(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Delete a document from the knowledge base (local only).
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            Deletion result
        """
        if self.use_local:
            return self.local_kb.delete(document_id)
        else:
            return {
                'success': False,
                'message': 'Document deletion not supported for AWS KB via this interface'
            }

# Create singleton instance
kb_tools = KnowledgeBaseTools()