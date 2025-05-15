# Pydantic models for data validation and structured data handling.

from pydantic import BaseModel, Field
from typing import List, Optional

# Example Pydantic model (can be expanded later)
class DocumentMetadata(BaseModel):
    source: str = Field(description="The source identifier of the document, e.g., file path or URL")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    last_modified: Optional[str] = Field(None, description="Last modified timestamp of the document")

class Chunk(BaseModel):
    text: str = Field(description="The text content of the chunk")
    metadata: DocumentMetadata = Field(description="Metadata associated with the chunk")
    embedding: Optional[List[float]] = Field(None, description="The vector embedding of the chunk text")

class Query(BaseModel):
    question: str = Field(description="The user's query")
    chat_history: Optional[List[dict]] = Field(None, description="Previous conversation turns")

class Answer(BaseModel):
    answer: str = Field(description="The LLM generated answer")
    retrieved_contexts: Optional[List[Chunk]] = Field(None, description="List of retrieved context chunks used for the answer")
    source_documents: Optional[List[str]] = Field(None, description="List of source document identifiers relevant to the answer") 