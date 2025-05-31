import asyncio
import logging
import json
import uuid # Keep for potential future use, though not for ADK session IDs here
import os # Added for os.listdir
from pathlib import Path # Added for Path operations
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from src.server.schemas import ChatQuery, IndexingResponse, IndexStatusResponse
# Import from original services and models for the pre-ADK RAG flow
from src.server.services import stream_qa_responses, get_chat_engine
from src.models import DocumentCitation
from src.core.indexing_service import run_indexing_pipeline
from src.config import settings # Import settings for paths

logger = logging.getLogger(__name__)
router = APIRouter()

# Health check endpoint for the router
@router.get("/health", tags=["Server Health"])
async def health_check_router():
    return {"status": "Router is healthy"}

@router.get("/index/status", response_model=IndexStatusResponse, tags=["Indexing"])
async def get_index_status():
    """Checks the status of the document index."""
    index_dir_path = Path(settings.vector_store_path)
    # A simple check could be if the main faiss file exists, e.g., inside the vector_store subdir
    # For a more robust check, one might look for multiple essential files.
    # For FAISS, the actual index file is often named like 'index.faiss' or 'default__vector_store.faiss'
    # and could be within a 'vector_store' subdirectory of settings.vector_store_path
    # Let's check for the existence of the directory specified by settings.vector_store_path and a key file.
    # A simpler check: does the root index path directory exist and is not empty?
    index_exists = False
    if index_dir_path.is_dir():
        # Check if there are any files/subdirs, implying it's been populated
        if any(index_dir_path.iterdir()):
            index_exists = True

    doc_count_in_data = 0
    data_dir = Path(settings.documents_dir)
    if data_dir.is_dir():
        try:
            # Count files, excluding hidden files like .DS_Store
            doc_count_in_data = len([f for f in data_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
        except Exception as e:
            logger.error(f"Error counting documents in data folder: {e}")
            doc_count_in_data = -1 # Indicate an error in counting
    
    status_message = ""
    if index_exists:
        status_message = f"An index appears to exist. The '{settings.documents_dir}' folder contains {doc_count_in_data} document(s)."
    else:
        status_message = f"No index found at '{settings.vector_store_path}'. The '{settings.documents_dir}' folder contains {doc_count_in_data} document(s) available for indexing."
        if doc_count_in_data == 0:
            status_message += " Please add documents to the data folder first."
            
    return IndexStatusResponse(
        exists=index_exists, 
        document_count_in_data_folder=doc_count_in_data,
        message=status_message
    )

@router.post("/index/documents", tags=["Indexing"], response_model=IndexingResponse)
async def trigger_document_indexing(background_tasks: BackgroundTasks):
    """
    Triggers the document indexing pipeline to run in the background.
    """
    logger.info("Received request to trigger document indexing.")
    try:
        background_tasks.add_task(run_indexing_pipeline)
        logger.info("Document indexing pipeline has been scheduled to run in the background.")
        return IndexingResponse(message="Document indexing process initiated in the background.")
    except Exception as e:
        logger.error(f"Failed to schedule document indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initiate indexing: {str(e)}")

@router.post("/chat/stream", tags=["Chat"])
async def stream_chat(query: ChatQuery):
    """
    Original endpoint to stream chat responses directly from RAG service.
    """
    request_id = str(uuid.uuid4()) # For logging this specific request
    logger.info(f"Request [{request_id}] Received streaming chat request: {query.query}")
    try:
        engine = await get_chat_engine()
        if not engine:
            logger.error(f"Request [{request_id}] Chat engine failed to initialize for /chat/stream endpoint.")
            raise HTTPException(status_code=503, detail="Chat engine is not available.")

        async def event_generator():
            logger.debug(f"Request [{request_id}] event_generator started.")
            event_count = 0
            try:
                async for item in stream_qa_responses(query.query):
                    event_count += 1
                    payload = {}
                    # event_type = "token" # Not strictly needed if payload structure is consistent

                    if isinstance(item, str):
                        logger.debug(f"Request [{request_id}] Event_gen item #{event_count} (str): '{item[:100]}...'")
                        if item.startswith("Error:") or item.startswith("Sorry, an error occurred:"):
                            logger.error(f"Request [{request_id}] Error token in stream: {item}")
                            payload = {"token": item, "error": True}
                        # elif item == "QA_SERVICE_STREAM_ENDED_SENTINEL":
                            # Let this fall through to be sent as a token, as original behavior showed.
                            # logger.debug(f"Request [{request_id}] Sentinel received, will be sent as token.")
                        #    payload = {"token": item} # Ensure it's packaged as a token
                        else:
                            payload = {"token": item} # This will package the sentinel too
                    elif isinstance(item, list): 
                        logger.debug(f"Request [{request_id}] Event_gen item #{event_count} (list): count {len(item)}")
                        if all(isinstance(dc, DocumentCitation) for dc in item) or not item:
                            # event_type = "citations" # Not strictly needed for client if client checks key
                            payload = {"citations": [dc.model_dump(mode='json') for dc in item]}
                        else:
                            logger.warning(f"Request [{request_id}] Received a list that isn't DocumentCitations: {item}")
                            payload = {"token": f"Internal server error: Unexpected data type in stream.", "error": True}
                    else:
                        logger.error(f"Request [{request_id}] Unexpected item type from stream_qa_responses: {type(item)}")
                        payload = {"token": "Internal server error: Corrupted stream data.", "error": True}
                    
                    sse_data = f"data: {json.dumps(payload)}\n\n"
                    logger.debug(f"Request [{request_id}] Yielding SSE data: {sse_data.strip()}")
                    yield sse_data
                    await asyncio.sleep(0.01)
                
                logger.info(f"Request [{request_id}] Finished iterating stream_qa_responses. Total items processed by event_gen: {event_count}")
            except Exception as e_inner:
                logger.error(f"Request [{request_id}] Exception inside event_generator loop: {e_inner}", exc_info=True)
                try:
                    error_payload = {"token": f"Server error during generation: {str(e_inner)}", "error": True}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                except Exception as e_yield_error:
                    logger.error(f"Request [{request_id}] Failed to yield error to client: {e_yield_error}")
            finally:
                logger.info(f"Request [{request_id}] event_generator finally block. Attempting to send stream_end.")
                end_event_payload = {"event": "stream_end"}
                try:
                    yield f"data: {json.dumps(end_event_payload)}\n\n"
                    logger.info(f"Request [{request_id}] Successfully yielded stream_end.")
                except Exception as e_yield_end:
                    logger.error(f"Request [{request_id}] Failed to yield stream_end: {e_yield_end}", exc_info=True)

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Request [{request_id}] Unexpected error in /chat/stream endpoint: {e}", exc_info=True)
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