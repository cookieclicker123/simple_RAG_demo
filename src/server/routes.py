import asyncio
import logging
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List # Import List for type hinting

from src.server.schemas import ChatQuery # , StreamResponse # StreamResponse might be used if not directly yielding strings
from src.server.services import stream_qa_responses, get_chat_engine
from src.models import DocumentCitation # Import DocumentCitation

logger = logging.getLogger(__name__)
router = APIRouter()

# Health check endpoint for the router
@router.get("/health", tags=["Server Health"])
async def health_check_router():
    return {"status": "Router is healthy"}

@router.post("/chat/stream", tags=["Chat"])
async def stream_chat(query: ChatQuery):
    """
    Endpoint to stream chat responses.
    Accepts a query and returns a streaming response of tokens.
    """
    logger.info(f"Received streaming chat request: {query.query}")
    try:
        # Ensure engine is initialized before starting the stream if it wasn't already
        # This also helps in giving a quicker error if engine init fails right away.
        engine = await get_chat_engine()
        if not engine:
            logger.error("Chat engine failed to initialize for /chat/stream endpoint.")
            raise HTTPException(status_code=503, detail="Chat engine is not available.")

        async def event_generator():
            # stream_qa_responses yields str (tokens) or List[DocumentCitation]
            async for item in stream_qa_responses(query.query):
                payload = {}
                event_type = "token" # Default event type

                if isinstance(item, str): # It's a text token or a string error message
                    if item.startswith("Error:") or item.startswith("Sorry, an error occurred:"):
                        # Log the error server-side
                        logger.error(f"Error token in stream: {item}")
                        payload = {"token": item, "error": True}
                    else:
                        payload = {"token": item}
                elif isinstance(item, list): # Expected to be List[DocumentCitation]
                    # Check if it's actually a list of DocumentCitation (or empty list)
                    if all(isinstance(dc, DocumentCitation) for dc in item) or not item:
                        event_type = "citations"
                        # Convert each DocumentCitation to a dict for JSON serialization
                        payload = {"citations": [dc.model_dump(mode='json') for dc in item]}
                    else:
                        # This case should ideally not happen if services.py is correct
                        logger.warning(f"Received a list in stream that isn't DocumentCitations: {item}")
                        # Fallback: treat as an error or skip
                        event_type = "token" # Or a special error event
                        payload = {"token": f"Internal server error: Unexpected data type in stream.", "error": True}
                else:
                    # This case should not be reached if services.py correctly yields str, List, or None for end.
                    logger.error(f"Unexpected item type from stream_qa_responses: {type(item)}")
                    event_type = "token" # Or a special error event
                    payload = {"token": "Internal server error: Corrupted stream data.", "error": True}

                # Send the event
                # For token events, the payload is like {"token": "..."}
                # For citation events, payload is {"citations": [...]}
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.01) # Small delay to allow other tasks, if necessary
            
            # Signal end of stream after all tokens and citations
            end_event_payload = {"event": "stream_end"}
            yield f"data: {json.dumps(end_event_payload)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    except HTTPException as http_exc: # Re-raise HTTPExceptions to be handled by FastAPI
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in /chat/stream endpoint: {e}", exc_info=True)
        # This will be caught by FastAPI's default error handling and result in a 500 error.
        # For more specific client feedback on such errors, an HTTPException could be raised here.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Example of a non-streaming endpoint (can be removed or adapted)
@router.post("/chat/simple", tags=["Chat"])
async def simple_chat(query: ChatQuery):
    logger.info(f"Received simple chat request: {query.query}")
    # This would require a non-streaming version in services.py
    # For now, let's just acknowledge and return a placeholder
    # response_text = await get_simple_qa_response(query.query)
    # return {"response": response_text}
    return {"message": "Simple chat endpoint placeholder. Implement if needed.", "query_received": query.query} 