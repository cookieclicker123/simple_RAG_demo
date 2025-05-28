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
    """Initializes the global chat engine instance. To be called at application startup."""
    global _chat_engine_instance
    async with _chat_engine_lock:
        if _chat_engine_instance is None:
            logger.info("Lifespan: Initializing global chat engine for FastAPI service at startup...")
            active_app_settings = get_active_settings()
            # Configure LlamaIndex globals. This is synchronous.
            # Running synchronous I/O bound task in a separate thread is good practice if it were slow, 
            # but configure_llama_index_globals is likely CPU-bound and fast.
            # For lifespan, it might be okay to run it directly if it's quick, or use to_thread.
            await asyncio.to_thread(configure_llama_index_globals, active_app_settings)
            logger.info(f"Lifespan: LlamaSettings.llm: {LlamaSettings.llm.model if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model') else LlamaSettings.llm}")
            logger.info(f"Lifespan: LlamaSettings.embed_model: {LlamaSettings.embed_model}")

            # sync_initialize_chat_engine can be blocking (loads models, index)
            _chat_engine_instance = await asyncio.to_thread(sync_initialize_chat_engine)
            if _chat_engine_instance:
                logger.info("Lifespan: Global chat engine initialized successfully.")
            else:
                logger.error("Lifespan: Failed to initialize global chat engine.")
        else:
            logger.info("Lifespan: Global chat engine already initialized.")

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
        logger.info("Lifespan: Global chat engine was not initialized.")

async def get_chat_engine() -> BaseChatEngine | None:
    """
    Returns the globally initialized chat engine.
    Relies on the startup event to have initialized the engine.
    """
    if _chat_engine_instance is None:
        logger.warning("get_chat_engine called but global engine is not initialized. This might happen if accessed before startup or after shutdown.")
        # Optionally, could try to initialize it here as a fallback, but ideally startup handles it.
        # await initialize_global_chat_engine() # Fallback initialization, uncomment if needed but makes lifespan less distinct
    return _chat_engine_instance

async def stream_qa_responses(query: str) -> AsyncGenerator[Union[str, List[DocumentCitation]], None]:
    """
    Asynchronously streams responses from the QA service using the global engine.
    """
    chat_engine = await get_chat_engine()
    if not chat_engine:
        yield "Error: Chat engine not available. It may not have initialized correctly at startup."
        yield [] 
        return

    logger.info(f"Streaming QA response for query: {query}")
    try:
        queue = asyncio.Queue()
        main_event_loop = asyncio.get_running_loop()

        def run_sync_generator_in_thread(loop: asyncio.AbstractEventLoop):
            try:
                logger.info(f"SERVICES.PY: run_sync_generator_in_thread starting for query: {query}")
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