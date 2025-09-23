"""Document data models."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

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
