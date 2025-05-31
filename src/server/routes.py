import asyncio
import logging
import json
import uuid # Keep for potential future use, though not for ADK session IDs here
import os # Added for os.listdir
from pathlib import Path # Added for Path operations
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict

from src.server.schemas import ChatQuery, IndexingResponse, IndexStatusResponse, UserQueryRequest, IndexTriggerResponse, IndexingStatusCheck, IndexCompletionRequest, IndexCleanupResponse
# Import from original services and models for the pre-ADK RAG flow
from src.server.services import stream_qa_responses, get_chat_engine
from src.models import DocumentCitation, ConversationMemory
from src.core.indexing_service import run_indexing_pipeline
from src.config import settings # Import settings for paths
from src.core.qa_service import initialize_chat_engine, stream_chat_response_with_memory
from src.utils.index_manager import IndexFileStructure, index_manager

# Add new imports for indexing completion detection
from src.server.schemas import IndexingState
# Import utilities for DRY code

logger = logging.getLogger(__name__)
router = APIRouter()

# Global variables for engine and memory management
chat_engine = None
conversation_memories: Dict[str, ConversationMemory] = defaultdict(lambda: ConversationMemory())

@router.on_event("startup")
async def startup_event():
    """Initialize the chat engine on application startup."""
    global chat_engine
    logger.info("Starting up FastAPI application...")
    
    chat_engine = initialize_chat_engine()
    if chat_engine:
        logger.info("Chat engine initialized successfully.")
    else:
        logger.error("Failed to initialize chat engine!")

@router.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Simple RAG Demo API is running"}

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "chat_engine_available": chat_engine is not None,
        "active_conversations": len(conversation_memories)
    }

@router.post("/chat/stream")
async def stream_chat(request: UserQueryRequest):
    """
    Stream chat responses with optional conversation memory.
    
    If session_id is provided, maintains conversation memory across requests.
    If no session_id, treats as a one-off query.
    """
    global chat_engine
    
    if not chat_engine:
        raise HTTPException(status_code=500, detail="Chat engine not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Get or create conversation memory
    session_id = request.session_id or str(uuid.uuid4())
    conversation_memory = conversation_memories[session_id]
    
    logger.info(f"Processing query for session {session_id}: {request.query}")
    
    # Use the new conversational stream function
    request_id = str(uuid.uuid4())
    
    async def generate_response():
        """Generate streaming response with conversation memory."""
        try:
            event_count = 0
            citations_sent = False
            
            for item in stream_chat_response_with_memory(request.query, chat_engine, conversation_memory):
                event_count += 1
                
                if isinstance(item, str):
                    logger.debug(f"Request [{request_id}] Event_gen item #{event_count} (str): '{item[:100]}...'")
                    if item.startswith("Error:") or item.startswith("Sorry, an error occurred:"):
                        logger.error(f"Request [{request_id}] Error token in stream: {item}")
                        payload = {"token": item, "error": True}
                    elif item == "QA_SERVICE_STREAM_ENDED_SENTINEL":
                        # Sentinel received - this signals the end of the QA service stream
                        # Don't send this to the client, just continue to the finally block
                        logger.debug(f"Request [{request_id}] Received QA service stream end sentinel")
                        continue
                    else:
                        # Regular content token
                        payload = {"token": item}
                    
                    event_data = f"data: {json.dumps(payload)}\n\n"
                    yield event_data.encode()
                
                elif isinstance(item, list):
                    # This should be the citations list
                    logger.debug(f"Request [{request_id}] Event_gen item #{event_count} (list): citations with {len(item)} items")
                    citations_data = []
                    for citation in item:
                        if hasattr(citation, '__dict__'):
                            citations_data.append(citation.__dict__)
                        else:
                            citations_data.append(citation)
                    
                    payload = {"citations": citations_data}
                    event_data = f"data: {json.dumps(payload)}\n\n"
                    yield event_data.encode()
                    citations_sent = True
                
                else:
                    # Unexpected type
                    logger.warning(f"Request [{request_id}] Unexpected item type in stream: {type(item)}")
                    continue
                    
            # Send final event to signal completion
            logger.info(f"Request [{request_id}] Stream complete. Events sent: {event_count}, Citations sent: {citations_sent}")
            
        except Exception as e:
            logger.error(f"Request [{request_id}] Error in generate_response: {e}", exc_info=True)
            error_payload = {"token": f"Error: {str(e)}", "error": True}
            error_event = f"data: {json.dumps(error_payload)}\n\n"
            yield error_event.encode()
        
        finally:
            # Send stream end event
            end_payload = {"event": "stream_end"}
            end_event = f"data: {json.dumps(end_payload)}\n\n"
            yield end_event.encode()
            logger.debug(f"Request [{request_id}] Final stream_end event sent")

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id
        }
    )

@router.delete("/chat/memory/{session_id}")
async def clear_conversation_memory(session_id: str):
    """Clear conversation memory for a specific session."""
    if session_id in conversation_memories:
        del conversation_memories[session_id]
        logger.info(f"Cleared conversation memory for session: {session_id}")
        return {"message": f"Conversation memory cleared for session {session_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"No conversation found for session {session_id}")

@router.get("/chat/memory/{session_id}")
async def get_conversation_memory(session_id: str):
    """Get conversation history for a specific session."""
    if session_id in conversation_memories:
        memory = conversation_memories[session_id]
        turns_data = []
        for turn in memory.turns:
            turns_data.append({
                "user_query": turn.user_query,
                "ai_response": turn.ai_response,
                "timestamp": turn.timestamp.isoformat(),
                "query_type": turn.query_type
            })
        
        return {
            "session_id": session_id,
            "total_turns": len(memory.turns),
            "max_turns": memory.max_turns,
            "turns": turns_data
        }
    else:
        raise HTTPException(status_code=404, detail=f"No conversation found for session {session_id}")

@router.get("/chat/sessions")
async def list_active_sessions():
    """List all active conversation sessions."""
    sessions = []
    for session_id, memory in conversation_memories.items():
        last_activity = memory.turns[-1].timestamp if memory.turns else None
        sessions.append({
            "session_id": session_id,
            "total_turns": len(memory.turns),
            "last_activity": last_activity.isoformat() if last_activity else None
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }

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

@router.post("/index/check-completion", response_model=IndexingStatusCheck, tags=["Indexing"])
async def check_indexing_completion(request: IndexCompletionRequest):
    """
    Checks if indexing is complete by verifying the required file structure exists.
    Based on the file structure created by indexing_service.py.
    """
    logger.info("Received request to check indexing completion status.")
    
    # Use utility to get required files and check existence
    index_dir_path = Path(settings.vector_store_path)
    files_found, files_missing = IndexFileStructure.check_files_exist(index_dir_path)
    
    # Determine completion status
    is_complete = len(files_missing) == 0  # All required files must exist
    
    if is_complete:
        state = IndexingState.COMPLETED
        progress_message = f"Indexing completed successfully. Found {len(files_found)} index files."
    elif len(files_found) > 0:
        state = IndexingState.IN_PROGRESS
        progress_message = f"Indexing in progress. Found {len(files_found)} of {len(settings.required_index_files)} required files."
    else:
        # Check if index directory exists at all
        if not index_dir_path.exists():
            state = IndexingState.NOT_STARTED
            progress_message = "Indexing not started. Index directory does not exist."
        else:
            state = IndexingState.IN_PROGRESS
            progress_message = "Indexing may have started but no files detected yet."
    
    logger.info(f"Indexing completion check: {state.value} - {progress_message}")
    
    return IndexingStatusCheck(
        state=state,
        progress_message=progress_message,
        files_found=files_found,
        files_missing=files_missing,
        is_complete=is_complete
    )

@router.delete("/index/cleanup", tags=["Indexing"])
async def cleanup_existing_index():
    """
    Deletes existing index files to prepare for re-indexing.
    This prevents false completion detection when re-indexing.
    """
    logger.info("Received request to clean up existing index files.")
    
    try:
        index_dir_path = Path(settings.vector_store_path)
        
        # Use utility to get required files for cleanup
        required_files = IndexFileStructure.get_required_files(index_dir_path)
        files_deleted = []
        
        # Add vector_store directory for complete cleanup
        cleanup_targets = [
            index_dir_path / "vector_store"  # Entire vector_store directory
        ]
        # Add individual files
        cleanup_targets.extend(required_files.values())
        
        for target in cleanup_targets:
            if target.exists():
                try:
                    if target.is_dir():
                        import shutil
                        shutil.rmtree(target)
                        files_deleted.append(f"directory: {target}")
                    else:
                        target.unlink()
                        files_deleted.append(f"file: {target}")
                    logger.info(f"Deleted: {target}")
                except Exception as e:
                    logger.warning(f"Could not delete {target}: {e}")
        
        if files_deleted:
            message = f"Successfully cleaned up {len(files_deleted)} index file(s)/directory(ies)"
            logger.info(f"Index cleanup completed: {message}")
        else:
            message = "No existing index files found to clean up"
            logger.info(message)
            
        return {"message": message, "files_deleted": files_deleted}
        
    except Exception as e:
        logger.error(f"Error during index cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clean up index files: {str(e)}")

@router.post("/index/trigger", response_model=IndexTriggerResponse)
async def trigger_indexing(background_tasks: BackgroundTasks):
    """Trigger the document indexing process."""
    success = await index_manager.trigger_indexing()
    return IndexTriggerResponse(success=success)

@router.get("/index/status", response_model=IndexingStatusCheck)
async def get_enhanced_index_status():
    """Get enhanced index status with document count and detailed messaging."""
    return await index_manager.get_enhanced_status()

@router.post("/index/check-completion")
async def check_indexing_completion(request: IndexCompletionRequest):
    """Check if indexing process has completed by examining file structure."""
    return await index_manager.check_completion_status()

@router.post("/index/cleanup", response_model=IndexCleanupResponse)
async def cleanup_existing_index():
    """Clean up existing index files before re-indexing."""
    success = await index_manager.cleanup_existing()
    return IndexCleanupResponse(success=success)

# Example of a non-streaming endpoint (can be removed or adapted)
@router.post("/chat/simple", tags=["Chat"])
async def simple_chat(query: ChatQuery):
    logger.info(f"Received simple chat request: {query.query}")
    # This would require a non-streaming version in services.py
    # For now, let's just acknowledge and return a placeholder
    # response_text = await get_simple_qa_response(query.query)
    # return {"response": response_text}
    return {"message": "Simple chat endpoint placeholder. Implement if needed.", "query_received": query.query} 