import asyncio
import logging
import uuid
from typing import AsyncGenerator, List, Union, Dict, Any
from datetime import datetime, timezone

from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core import Settings as LlamaSettings

from src.core.qa_service import initialize_chat_engine as sync_initialize_chat_engine, stream_chat_response as sync_stream_chat_response
from src.core.indexing_service import configure_llama_index_globals, get_active_settings
from src.models import DocumentCitation, ConversationMemory

# Import agent framework components
from agents.agent_models import (
    AgentInput,
    AgentOutput, 
    AgentResult,
    AgentContext,
    AgentMetadata,
    AgentRole
)
from agents.meta_agent import meta_agent
from tools.tool_registry import tool_registry
from tools.rag_tool import set_global_chat_engine

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
                # Set the chat engine reference in the RAG tool
                set_global_chat_engine(temp_engine)
                logger.info("Global chat engine initialized successfully.")
                logger.info("Agent framework initialized - meta-agent and tools ready.")
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

# NEW AGENT-BASED SERVICES

async def stream_agent_responses(
    query: str, 
    session_id: str,
    conversation_memory: ConversationMemory
) -> AsyncGenerator[Union[str, List[DocumentCitation]], None]:
    """
    Stream responses using the agent framework.
    
    This is the new primary interface that uses the meta-agent to orchestrate
    tool usage and provide intelligent responses with conversation memory.
    
    Args:
        query: User's query
        session_id: Session identifier for tracking
        conversation_memory: Current conversation state
        
    Yields:
        Either response tokens (str) or citations (List[DocumentCitation])
    """
    logger.info(f"Agent framework: Processing query via meta-agent: {query}")
    logger.info(f"Session: {session_id}, Memory turns: {len(conversation_memory.turns)}")
    
    try:
        # Create agent input
        agent_input = AgentInput(
            query=query,
            context=AgentContext(
                conversation_memory=conversation_memory,
                user_query=query,
                available_tools=tool_registry.list_tools(),
                execution_config={}
            ),
            metadata=AgentMetadata(
                agent_id=str(uuid.uuid4()),
                agent_role=AgentRole.META,
                session_id=session_id,
                execution_id=str(uuid.uuid4())
            )
        )
        
        # Execute meta-agent
        agent_result = await meta_agent(agent_input)
        
        if agent_result.success:
            # Stream the response text as tokens (simulating real-time output)
            response_text = agent_result.output.response_text
            
            # Simple token-like streaming by splitting on words
            # This maintains the streaming interface while using the agent framework
            words = response_text.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word
                
                # Small delay to simulate streaming (can be removed in production)
                await asyncio.sleep(0.01)
            
            # Yield citations at the end
            if agent_result.output.citations:
                yield agent_result.output.citations
            
            logger.info(f"Agent framework: Successfully processed query with {len(agent_result.output.tool_calls_made)} tools")
            logger.info(f"Tools used: {agent_result.output.tool_calls_made}")
        
        else:
            # Handle agent execution failure
            error_message = agent_result.error_details or "Agent execution failed"
            logger.error(f"Agent framework: Execution failed: {error_message}")
            yield f"I encountered an error while processing your request: {error_message}"
            yield []
            
    except Exception as e:
        logger.error(f"Agent framework: Unexpected error: {e}", exc_info=True)
        yield f"I apologize, but I encountered an unexpected error: {str(e)}"
        yield []


async def execute_agent_query(
    query: str,
    session_id: str, 
    conversation_memory: ConversationMemory
) -> AgentResult:
    """
    Execute a single query through the agent framework without streaming.
    
    This function provides a non-streaming interface to the agent framework,
    useful for API endpoints that need the complete result.
    
    Args:
        query: User's query
        session_id: Session identifier
        conversation_memory: Current conversation state
        
    Returns:
        Complete AgentResult with response and metadata
    """
    logger.info(f"Agent framework: Executing non-streaming query: {query}")
    
    try:
        # Create agent input
        agent_input = AgentInput(
            query=query,
            context=AgentContext(
                conversation_memory=conversation_memory,
                user_query=query,
                available_tools=tool_registry.list_tools(),
                execution_config={}
            ),
            metadata=AgentMetadata(
                agent_id=str(uuid.uuid4()),
                agent_role=AgentRole.META,
                session_id=session_id,
                execution_id=str(uuid.uuid4())
            )
        )
        
        # Execute meta-agent
        agent_result = await meta_agent(agent_input)
        
        logger.info(f"Agent framework: Query executed successfully")
        logger.info(f"Tools used: {agent_result.output.tool_calls_made}")
        logger.info(f"Citations: {len(agent_result.output.citations)}")
        
        return agent_result
        
    except Exception as e:
        logger.error(f"Agent framework: Query execution failed: {e}", exc_info=True)
        
        # Return error result
        error_output = AgentOutput(
            response_text=f"I encountered an error while processing your request: {str(e)}",
            tool_calls_made=[],
            tool_results=[],
            citations=[],
            sub_agent_calls=[],
            confidence_score=0.0,
            execution_metadata={"error": str(e)},
            next_suggested_actions=[]
        )
        
        return AgentResult(
            input_metadata=AgentMetadata(
                agent_id=str(uuid.uuid4()),
                agent_role=AgentRole.META,
                session_id=session_id,
                execution_id=str(uuid.uuid4())
            ),
            output=error_output,
            execution_duration_ms=0.0,
            success=False,
            error_details=str(e),
            memory_updated=False
        )


def get_agent_framework_info() -> Dict[str, Any]:
    """
    Get information about the agent framework capabilities.
    
    Returns:
        Dictionary with framework information and available tools
    """
    return {
        "framework_version": "1.0.0",
        "meta_agent_available": True,
        "available_tools": tool_registry.list_tools(),
        "tool_registry_stats": tool_registry.get_registry_stats(),
        "capabilities": [
            "Intelligent tool selection",
            "Conversation memory integration", 
            "Multi-step reasoning",
            "Document retrieval and analysis",
            "Extensible tool framework"
        ]
    } 

async def stream_qa_responses(query: str) -> AsyncGenerator[Union[str, List[DocumentCitation]], None]:
    """
    Asynchronously streams responses from the QA service using the global engine.
    
    LEGACY FUNCTION: Maintained for backward compatibility.
    New implementations should use stream_agent_responses instead.
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