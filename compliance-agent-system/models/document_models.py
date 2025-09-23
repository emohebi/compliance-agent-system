"""Document data models."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

from pydantic import BaseModel, Field
from typing import List, Dict, Any

class DocumentSearchResult(BaseModel):
    """Model for document search results."""
    relevant_documents: List[str] = Field(description="List of relevant document contents")
    total_found: int = Field(description="Total number of documents found")
    summary: str = Field(description="Summary of findings")
    key_insights: List[str] = Field(description="Key insights from the documents")

class DocumentSummary(BaseModel):
    """Model for document summary."""
    main_topics: List[str] = Field(description="Main topics covered")
    key_points: List[str] = Field(description="Key points from the documents")
    gaps: List[str] = Field(description="Information gaps identified")
    confidence_score: float = Field(description="Confidence in the summary (0.0-1.0)")

class Document(BaseModel):
    """Model for a document."""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    source: Optional[str] = Field(None, description="Document source")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
class DocumentBatch(BaseModel):
    """Model for a batch of documents."""
    documents: List[Document]
    batch_id: str = Field(..., description="Batch ID")
    processing_status: str = Field("pending", description="Processing status")
