from pydantic import BaseModel
class ChatQuery(BaseModel):
    query: str
    # session_id: Optional[str] = None # Could add session management later
    # chat_history: Optional[List[dict]] = None # If we want to pass history per request

class StreamResponse(BaseModel):
    token: str
    # Could add other fields like event_type (e.g., "token", "end", "error") 