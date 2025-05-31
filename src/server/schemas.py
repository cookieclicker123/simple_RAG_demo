from pydantic import BaseModel
from typing import Optional, List # Optional might be used by IndexingResponse, List for ChatQuery history
from enum import Enum

class ChatQuery(BaseModel):
    query: str
    # session_id: Optional[str] = None # Could add session management later
    # chat_history: Optional[List[dict]] = None # If we want to pass history per request

# Ensure IndexingResponse is defined for the /index/documents endpoint
class IndexingResponse(BaseModel):
    message: str
    details: Optional[str] = None

class IndexStatusResponse(BaseModel):
    exists: bool
    document_count_in_data_folder: int
    message: str

class StreamResponse(BaseModel):
    token: str
    # Could add other fields like event_type (e.g., "token", "end", "error") 

# New schemas for enhanced index management
class IndexStatus(str, Enum):
    EXISTS = "exists"
    MISSING = "missing" 
    EMPTY_DATA_FOLDER = "empty_data_folder"

class IndexCheckResult(BaseModel):
    status: IndexStatus
    document_count: int
    message: str
    needs_indexing: bool
    can_proceed_without_indexing: bool = False

class UserConfirmation(BaseModel):
    confirmed: bool
    action: str  # What action the user is confirming (e.g., "index_documents", "proceed_anyway")
    message: str 