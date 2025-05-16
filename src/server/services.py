import asyncio
import logging
from typing import AsyncGenerator, List, Union

from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core import Settings as LlamaSettings

from src.core.qa_service import initialize_chat_engine as sync_initialize_chat_engine, stream_chat_response as sync_stream_chat_response
from src.core.indexing_service import configure_llama_index_globals, get_active_settings
from src.models import DocumentCitation

logger = logging.getLogger(__name__)
# logging.basicConfig(level=settings.log_level) # Main app or web_app should configure this

# Global chat engine instance
# This is a simple approach; for robustness, consider FastAPI's lifespan events
# or a more sophisticated dependency injection for managing such resources.
_chat_engine_instance: BaseChatEngine | None = None
_chat_engine_lock = asyncio.Lock() # To prevent race conditions during initialization

async def get_chat_engine() -> BaseChatEngine | None:
    """
    Initializes and returns the chat engine.
    Uses a lock to prevent multiple initializations if called concurrently.
    """
    global _chat_engine_instance
    if _chat_engine_instance is None:
        async with _chat_engine_lock:
            if _chat_engine_instance is None: # Double check after acquiring lock
                logger.info("No active chat engine session. Initializing for FastAPI service...")
                # Ensure LlamaIndex globals are configured
                active_app_settings = get_active_settings()
                # This needs to be synchronous as configure_llama_index_globals is sync
                # and LlamaSettings are global.
                # Running synchronous I/O bound task in a separate thread
                await asyncio.to_thread(configure_llama_index_globals, active_app_settings)

                logger.info(f"FastAPI service using LLM (from LlamaSettings): {LlamaSettings.llm.model if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model') else LlamaSettings.llm}")
                logger.info(f"FastAPI service using Embedding Model (from LlamaSettings): {LlamaSettings.embed_model}")

                # sync_initialize_chat_engine can be blocking (loads models, index)
                # Run it in a thread pool to avoid blocking the event loop
                _chat_engine_instance = await asyncio.to_thread(sync_initialize_chat_engine)
                if _chat_engine_instance:
                    logger.info("Chat engine initialized successfully for FastAPI service.")
                else:
                    logger.error("Failed to initialize chat engine for FastAPI service.")
    return _chat_engine_instance

async def stream_qa_responses(query: str) -> AsyncGenerator[Union[str, List[DocumentCitation]], None]:
    """
    Asynchronously streams responses from the QA service.
    """
    chat_engine = await get_chat_engine()
    if not chat_engine:
        # Yield a string error token first, then an empty list for citations
        yield "Error: Chat engine not initialized."
        yield [] 
        return

    logger.info(f"Streaming QA response for query: {query}")
    try:
        # sync_stream_chat_response now yields strings (tokens) and then a List[DocumentCitation]
        # We'll run the synchronous generator in a separate thread
        # and yield items back to the async generator.
        
        # Create a queue to pass data from the sync thread to the async generator
        queue = asyncio.Queue()

        # Get the current running event loop in the main async context
        main_event_loop = asyncio.get_running_loop()

        def run_sync_generator_in_thread(loop: asyncio.AbstractEventLoop):
            try:
                for item in sync_stream_chat_response(query, chat_engine):
                    # Schedule queue.put in the main event loop
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
                # Signal completion of all items (tokens + citations list)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result() # End of stream signal
            except Exception as e:
                logger.error(f"Error in sync_stream_chat_response thread: {e}", exc_info=True)
                # Send error token, then empty citation list, then end signal
                asyncio.run_coroutine_threadsafe(queue.put(f"Error during streaming: {e}"), loop).result()
                asyncio.run_coroutine_threadsafe(queue.put([]), loop).result() # Empty citations on error
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        # Start the synchronous generator in a separate thread
        import threading
        # Pass the main event loop to the thread function
        thread = threading.Thread(target=run_sync_generator_in_thread, args=(main_event_loop,))
        thread.start()

        # Consume from the queue in the async generator
        while True:
            item = await queue.get()
            if item is None: # End of stream signal
                break
            yield item # Yields strings (tokens) or List[DocumentCitation]
            queue.task_done() # Signal that the item has been processed
        
        thread.join() # Ensure the thread finishes

    except Exception as e:
        logger.error(f"Error in stream_qa_responses: {e}", exc_info=True)
        yield f"Sorry, an error occurred: {e}"
        yield [] # Yield empty list for citations in case of error 