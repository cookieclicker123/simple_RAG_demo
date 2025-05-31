from pydantic import BaseModel, Field
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

# New schemas for indexing completion detection
class IndexingState(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

class IndexingStatusCheck(BaseModel):
    state: IndexingState
    progress_message: str
    files_found: List[str]  # List of required files that were found
    files_missing: List[str]  # List of required files that are missing
    is_complete: bool

class IndexCompletionRequest(BaseModel):
    check_files: bool = True  # Whether to check file structure for completion 

class IndexCleanupResponse(BaseModel):
    """Response model for index cleanup operations."""
    success: bool

class IndexTriggerResponse(BaseModel):
    """Response model for triggering indexing operations."""
    success: bool

class UserQueryRequest(BaseModel):
    """Request model for conversational chat with optional session management."""
    query: str = Field(..., description="User's question or query")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation memory") 