"""Local JSON-based knowledge base implementation."""

import json
import re
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)

class LocalKnowledgeBase:
    """
    Local knowledge base using JSON file storage with simple search capabilities.
    """
    
    def __init__(self, kb_file: str = "./data/local_knowledge_base.json"):
        """Initialize local knowledge base."""
        self.kb_file = Path(kb_file)
        self.documents = []
        self.document_vectors = {}  # For TF-IDF vectors
        self.idf_scores = {}  # Inverse document frequency scores
        self._ensure_kb_exists()
        self._load_knowledge_base()
        self._build_search_index()
    
    def _ensure_kb_exists(self):
        """Ensure knowledge base file and directory exist."""
        self.kb_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.kb_file.exists():
            # Create initial knowledge base structure
            initial_kb = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "documents": []
            }
            self._save_knowledge_base(initial_kb)
            logger.info(f"Created new knowledge base at: {self.kb_file}")
    
    def _load_knowledge_base(self):
        """Load knowledge base from JSON file."""
        try:
            with open(self.kb_file, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
                self.documents = kb_data.get('documents', [])
                logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.documents = []
    
    def _save_knowledge_base(self, kb_data: Dict[str, Any] = None):
        """Save knowledge base to JSON file."""
        if kb_data is None:
            kb_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "documents": self.documents
            }
        else:
            kb_data["updated_at"] = datetime.now().isoformat()
        
        try:
            with open(self.kb_file, 'w', encoding='utf-8') as f:
                json.dump(kb_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved knowledge base with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for text."""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate term frequency."""
        token_count = len(tokens)
        if token_count == 0:
            return {}
        
        tf = Counter(tokens)
        # Normalize by document length
        return {term: count / token_count for term, count in tf.items()}
    
    def _calculate_idf(self):
        """Calculate inverse document frequency for all terms."""
        total_docs = len(self.documents)
        if total_docs == 0:
            return {}
        
        # Count documents containing each term
        term_doc_count = {}
        for doc in self.documents:
            tokens = set(self._tokenize(doc.get('content', '')))
            for token in tokens:
                term_doc_count[token] = term_doc_count.get(token, 0) + 1
        
        # Calculate IDF
        self.idf_scores = {
            term: math.log(total_docs / count)
            for term, count in term_doc_count.items()
        }
    
    def _build_search_index(self):
        """Build TF-IDF search index for all documents."""
        self._calculate_idf()
        
        for i, doc in enumerate(self.documents):
            content = doc.get('content', '')
            tokens = self._tokenize(content)
            tf = self._calculate_tf(tokens)
            
            # Calculate TF-IDF vector
            tfidf_vector = {}
            for term, tf_score in tf.items():
                idf_score = self.idf_scores.get(term, 0)
                tfidf_vector[term] = tf_score * idf_score
            
            self.document_vectors[i] = tfidf_vector
    
    def _calculate_similarity(self, query_vector: Dict[str, float], doc_vector: Dict[str, float]) -> float:
        """Calculate cosine similarity between query and document vectors."""
        # Get all unique terms
        all_terms = set(query_vector.keys()) | set(doc_vector.keys())
        
        if not all_terms:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in all_terms)
        query_magnitude = math.sqrt(sum(v ** 2 for v in query_vector.values()))
        doc_magnitude = math.sqrt(sum(v ** 2 for v in doc_vector.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
        
        return dot_product / (query_magnitude * doc_magnitude)
    
    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.1
    ) -> Dict[str, Any]:
        """
        Retrieve documents using TF-IDF similarity search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            Dictionary with search results
        """
        if not self.documents:
            return {
                'success': True,
                'query': query,
                'results': [],
                'total_results': 0,
                'message': 'No documents in knowledge base'
            }
        
        # Tokenize and create query vector
        # query_tokens = self._tokenize(query)
        # query_tf = self._calculate_tf(query_tokens)
        
        # # Create query TF-IDF vector
        # query_vector = {}
        # for term, tf_score in query_tf.items():
        #     idf_score = self.idf_scores.get(term, 0)
        #     query_vector[term] = tf_score * idf_score
        
        # # Calculate similarity scores for all documents
        # scores = []
        # for i, doc in enumerate(self.documents):
        #     doc_vector = self.document_vectors.get(i, {})
        #     similarity = self._calculate_similarity(query_vector, doc_vector)
            
        #     if similarity >= min_score:
        #         scores.append((i, similarity))
        
        # # Sort by score (descending) and limit results
        # scores.sort(key=lambda x: x[1], reverse=True)
        # scores = scores[:max_results]
        
        # Build results
        results = []
        for doc_idx, doc in enumerate(self.documents):
            # doc = self.documents[doc_idx]
            results.append({
                'content': doc.get('content', ''),
                'score': round(1, 4),
                'metadata': doc.get('metadata', {}),
                'document_id': doc.get('id', f'doc_{doc_idx}')
            })
        
        return {
            'success': True,
            'query': query,
            'results': results,
            'total_results': len(results),
            'search_method': '*'
        }
    
    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a document in the knowledge base.
        
        Args:
            content: Document content
            metadata: Optional metadata
            document_id: Optional document ID
            
        Returns:
            Storage result
        """
        if not document_id:
            # Generate ID from content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
            document_id = f"doc_{content_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Check if document already exists
        existing_idx = None
        for i, doc in enumerate(self.documents):
            if doc.get('id') == document_id:
                existing_idx = i
                break
        
        # Create document
        document = {
            'id': document_id,
            'content': content,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        if existing_idx is not None:
            # Update existing document
            self.documents[existing_idx] = document
            message = f"Updated document: {document_id}"
        else:
            # Add new document
            self.documents.append(document)
            message = f"Stored new document: {document_id}"
        
        # Rebuild search index
        self._build_search_index()
        
        # Save to file
        self._save_knowledge_base()
        
        return {
            'success': True,
            'document_id': document_id,
            'message': message,
            'total_documents': len(self.documents)
        }
    
    def delete(self, document_id: str) -> Dict[str, Any]:
        """Delete a document from the knowledge base."""
        for i, doc in enumerate(self.documents):
            if doc.get('id') == document_id:
                self.documents.pop(i)
                self._build_search_index()
                self._save_knowledge_base()
                return {
                    'success': True,
                    'message': f"Deleted document: {document_id}",
                    'total_documents': len(self.documents)
                }
        
        return {
            'success': False,
            'message': f"Document not found: {document_id}",
            'total_documents': len(self.documents)
        }
    
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List documents in the knowledge base."""
        total = len(self.documents)
        documents = self.documents[offset:offset + limit]
        
        return {
            'success': True,
            'documents': [
                {
                    'id': doc.get('id'),
                    'metadata': doc.get('metadata', {}),
                    'created_at': doc.get('created_at'),
                    'content_preview': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', '')
                }
                for doc in documents
            ],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        total_docs = len(self.documents)
        total_tokens = sum(len(self._tokenize(doc.get('content', ''))) for doc in self.documents)
        unique_terms = len(self.idf_scores)
        
        # Calculate average document length
        avg_doc_length = total_tokens / total_docs if total_docs > 0 else 0
        
        # Get metadata statistics
        metadata_keys = set()
        for doc in self.documents:
            metadata_keys.update(doc.get('metadata', {}).keys())
        
        return {
            'total_documents': total_docs,
            'total_tokens': total_tokens,
            'unique_terms': unique_terms,
            'average_document_length': round(avg_doc_length, 2),
            'metadata_fields': list(metadata_keys),
            'kb_file': str(self.kb_file),
            'last_updated': self.documents[-1].get('updated_at') if self.documents else None
        }


# Create singleton instance
local_kb = LocalKnowledgeBase()