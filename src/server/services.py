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

async def initialize_global_chat_engine():
    """Initializes the global chat engine instance. Can be called at startup or on demand."""
    global _chat_engine_instance
    # Use the lock to ensure only one initialization attempt happens at a time
    async with _chat_engine_lock:
        # Check again inside the lock in case another coroutine was waiting and instance got set
        if _chat_engine_instance is None: 
            logger.info("Attempting to initialize global chat engine...")
            active_app_settings = get_active_settings()
            await asyncio.to_thread(configure_llama_index_globals, active_app_settings)
            logger.info(f"LlamaSettings.llm for engine init: {LlamaSettings.llm.model if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model') else LlamaSettings.llm}")
            logger.info(f"LlamaSettings.embed_model for engine init: {LlamaSettings.embed_model}")

            temp_engine = await asyncio.to_thread(sync_initialize_chat_engine)
            if temp_engine:
                _chat_engine_instance = temp_engine
                logger.info("Global chat engine initialized successfully.")
            else:
                logger.error("Failed to initialize global chat engine during attempt.")
        else:
            logger.info("Global chat engine was already initialized by another coroutine or at startup.")

async def shutdown_global_chat_engine():
    """Placeholder for shutting down/releasing resources from the global chat engine."""
    global _chat_engine_instance
    logger.info("Lifespan: Shutting down global chat engine...")
    if _chat_engine_instance is not None:
        # For most LlamaIndex engines, there isn't an explicit close().
        # Deleting the reference can help with garbage collection.
        # If specific components had close methods (e.g., a db connection), call them here.
        del _chat_engine_instance
        _chat_engine_instance = None
        logger.info("Lifespan: Global chat engine instance released.")
    else:
        logger.info("Lifespan: Global chat engine was not initialized or already shut down.")

async def get_chat_engine() -> BaseChatEngine | None:
    """
    Returns the globally initialized chat engine.
    If not initialized, attempts to initialize it.
    """
    if _chat_engine_instance is None:
        logger.warning("get_chat_engine: Global engine is None. Attempting on-demand initialization.")
        await initialize_global_chat_engine() # This will attempt to set _chat_engine_instance
        if _chat_engine_instance is None:
            logger.error("get_chat_engine: Engine is STILL None after on-demand initialization attempt.")
    return _chat_engine_instance

async def stream_qa_responses(query: str) -> AsyncGenerator[Union[str, List[DocumentCitation]], None]:
    """
    Asynchronously streams responses from the QA service using the global engine.
    """
    chat_engine = await get_chat_engine()
    if not chat_engine:
        yield "Error: Chat engine not available. Failed to initialize."
        yield [] 
        return

    logger.info(f"Streaming QA response for query: {query}")
    try:
        queue = asyncio.Queue()
        main_event_loop = asyncio.get_running_loop()

        def run_sync_generator_in_thread(loop: asyncio.AbstractEventLoop):
            try:
                logger.info(f"SERVICES.PY: run_sync_generator_in_thread starting for query: {query}")
                # Pass the obtained chat_engine to the synchronous function
                for item_count, item in enumerate(sync_stream_chat_response(query, chat_engine)):
                    logger.info(f"SERVICES.PY: run_sync_generator_in_thread received item #{item_count} from qa_service: '{str(item)[:100]}...'")
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
                logger.info(f"SERVICES.PY: run_sync_generator_in_thread loop FINISHED for query: {query}. Sending None to queue.")
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
            except Exception as e:
                logger.error(f"Error in sync_stream_chat_response thread: {e}", exc_info=True)
                asyncio.run_coroutine_threadsafe(queue.put(f"Error during streaming: {e}"), loop).result()
                asyncio.run_coroutine_threadsafe(queue.put([]), loop).result()
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        import threading
        thread = threading.Thread(target=run_sync_generator_in_thread, args=(main_event_loop,))
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                logger.debug("stream_qa_responses: Received None from queue, breaking.")
                break
            if isinstance(item, str):
                logger.debug(f"stream_qa_responses: Yielding token: '{item[:30]}...'")
            elif isinstance(item, list):
                logger.debug(f"stream_qa_responses: Yielding citations list (count: {len(item)})")
            else:
                logger.debug(f"stream_qa_responses: Yielding item of type {type(item)}")
            yield item
            queue.task_done()
        
        logger.debug(f"stream_qa_responses: Exited queue consumption loop for query '{query}'.")
        thread.join()
        logger.debug(f"stream_qa_responses: Thread joined for query '{query}'. Async generator finishing.")

    except Exception as e:
        logger.error(f"Error in stream_qa_responses: {e}", exc_info=True)
        yield f"Sorry, an error occurred: {e}"
        yield [] 