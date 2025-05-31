from pydantic import BaseModel
from typing import Optional, List # Optional might be used by IndexingResponse, List for ChatQuery history

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