# Service for Question Answering: conversational retrieval, memory, and LLM interaction.

import logging
from typing import  Generator

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings as LlamaSettings
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI # LlamaIndex wrapper for OpenAI

# LangChain imports for memory (though LlamaIndex chat engine can manage its own)
# from langchain.memory import ConversationBufferMemory # Option 1: LangChain memory
from llama_index.core.memory import ChatMemoryBuffer # Option 2: LlamaIndex native memory (preferred with LlamaIndex chat engine)

# Configuration and utility imports (using src. prefix as per your current setup)
from src.config import settings
from src.utils.vector_store_handlers import load_faiss_index_from_storage

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

# Explicitly configure/re-configure LlamaSettings.llm based on this module's view of settings.
# This ensures that even if another module (like indexing_service) set it first with a different
# value (e.g., from an old .env or different default), qa_service uses its configured LLM.
current_llm_model_in_settings = settings.llm_model_name
llm_needs_configuration = True
if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model'):
    if LlamaSettings.llm.model == current_llm_model_in_settings:
        llm_needs_configuration = False
        logger.info(f"LlamaSettings.llm already configured correctly with: {LlamaSettings.llm.model}")
    else:
        logger.info(f"LlamaSettings.llm is {LlamaSettings.llm.model}, but config wants {current_llm_model_in_settings}. Re-configuring.")
elif not LlamaSettings.llm:
    logger.info("LlamaSettings.llm not yet configured.")
else: # LLM is set but has no model attribute or is not an OpenAI instance we expect
    logger.info(f"LlamaSettings.llm is an unexpected type ({type(LlamaSettings.llm)}), re-configuring.")

if llm_needs_configuration:
    logger.info(f"Configuring LlamaSettings.llm with OpenAI model: {current_llm_model_in_settings}")
    try:
        LlamaSettings.llm = LlamaIndexOpenAI(
            model=current_llm_model_in_settings,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
            max_tokens=settings.max_tokens
        )
    except Exception as e:
        logger.error(f"Failed to configure LlamaSettings.llm with {current_llm_model_in_settings}: {e}", exc_info=True)
        # Potentially fall back to a default or raise a more specific error if LLM is critical
        if not LlamaSettings.llm: # If it failed and LlamaSettings.llm is still None
            logger.warning("Falling back to a default gpt-3.5-turbo due to configuration error.")
            LlamaSettings.llm = LlamaIndexOpenAI(model="gpt-3.5-turbo", temperature=settings.temperature, api_key=settings.openai_api_key)

def initialize_chat_engine() -> BaseChatEngine | None:
    """Initializes and returns a LlamaIndex chat engine.

    This involves loading the vector index, configuring the LLM, and setting up memory.
    Returns:
        An initialized LlamaIndex BaseChatEngine instance, or None if setup fails.
    """
    logger.info("Initializing chat engine...")
    logger.info(f"Attempting to use LLM: {LlamaSettings.llm.model if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model') else 'Not configured or unknown type'}")

    # 1. Load the persisted vector index
    logger.info(f"Loading vector index from: {settings.vector_store_path}")
    vector_index: VectorStoreIndex | None = load_faiss_index_from_storage(
        vector_store_path_str=settings.vector_store_path
    )

    if not vector_index:
        logger.error("Failed to load vector index. Cannot initialize chat engine.")
        return None
    logger.info(f"Vector index loaded successfully. Index ID: {vector_index.index_id}")

    # 2. Setup Chat Memory (Using LlamaIndex native memory)
    # This memory buffer will be managed by the LlamaIndex chat engine.
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000) # Adjust token_limit as needed
    logger.info(f"Chat memory (ChatMemoryBuffer) initialized.")

    # 4. Create Chat Engine from the index
    try:
        # LlamaSettings.llm will be used by default if `llm` argument is not provided to as_chat_engine
        chat_engine = vector_index.as_chat_engine(
            chat_mode="condense_plus_context", # A good general-purpose mode
            memory=memory,
            # llm=llm, # Can override global LlamaSettings.llm here if needed
            verbose=True, # For more detailed logging from the chat engine
            system_prompt=(
                "You are a helpful and knowledgeable AI assistant. "
                "Answer the user's questions based on the provided context. "
                "If the context doesn't contain the answer, say that you don't know. "
                "Be concise and informative."
            ),
            streaming=True # Enable streaming for the chat engine if it supports it directly
        )
        logger.info("Chat engine initialized successfully.")
        return chat_engine
    except Exception as e:
        logger.error(f"Error initializing chat engine: {e}", exc_info=True)
        return None

def stream_chat_response(query: str, chat_engine: BaseChatEngine) -> Generator[str, None, None]:
    """Gets a streaming response from the chat engine for the given query.

    Args:
        query: The user's question.
        chat_engine: The initialized LlamaIndex chat engine.

    Yields:
        Response tokens from the AI.
    """
    logger.info(f"User query (for streaming): {query}")
    try:
        streaming_response = chat_engine.stream_chat(query)
        # streaming_response is an AgentChatResponse or StreamingAgentChatResponse object
        # It should have a `response_gen` attribute that is the generator of tokens.
        if hasattr(streaming_response, 'response_gen'):
            for token in streaming_response.response_gen:
                yield token
        else:
            # Fallback if response_gen is not available (e.g., error or non-streaming response type)
            logger.warning("Streaming response object does not have 'response_gen'. Returning full response if available.")
            if hasattr(streaming_response, 'response'):
                 yield str(streaming_response.response) # Yield the whole response as one chunk
            else:
                yield "Error: Could not get a streamable response."

    except Exception as e:
        logger.error(f"Error getting streaming chat response: {e}", exc_info=True)
        yield "Sorry, I encountered an error while processing your request."

if __name__ == "__main__":
    # This test block requires an existing index and a configured OpenAI API key.
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # Re-trigger LlamaIndex global config to ensure it's set based on current .env
    # (especially if .env was just created or changed for testing OpenAI key)
    from src.core.indexing_service import  get_active_settings as resolve_embed_model
    from src.config import AppSettings # Import AppSettings for potential reload

    logger.info("--- QA Service Test (Streaming) --- ")
    # Ensure .env is loaded for OPENAI_API_KEY
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here_if_not_in_env":
        logger.warning("OPENAI_API_KEY not found or is default in .env. QA service test might fail or use dummy responses.")
        print("Please set your OPENAI_API_KEY in the .env file to test the QA service properly.")
        # Optionally, create a dummy .env for skeleton to run without real calls
        # exit(1) # Or allow to proceed if LlamaSettings.llm can handle no key (e.g. if it's a local model)
    
    # Reload AppSettings for the test context to pick up fresh .env, then reconfigure LlamaIndex globals
    _test_settings = AppSettings()
    # configure_llama_index_globals handles both embed_model and LLM based on passed settings.
    # However, the module-level LLM config above will run first based on initial settings import.
    # For the test, we ensure the LLM setting specifically uses the potentially reloaded _test_settings.
    if hasattr(LlamaSettings, 'llm') and hasattr(LlamaSettings.llm, 'model') and LlamaSettings.llm.model != _test_settings.llm_model_name:
        logger.info(f"Re-configuring LlamaSettings.llm for test from {_test_settings.llm_model_name}")
        LlamaSettings.llm = LlamaIndexOpenAI(
            model=_test_settings.llm_model_name,
            temperature=_test_settings.temperature,
            api_key=_test_settings.openai_api_key,
            max_tokens=_test_settings.max_tokens
        )
    elif not LlamaSettings.llm:
         LlamaSettings.llm = LlamaIndexOpenAI(
            model=_test_settings.llm_model_name,
            temperature=_test_settings.temperature,
            api_key=_test_settings.openai_api_key,
            max_tokens=_test_settings.max_tokens
        )

    # We also need to ensure embed_model is set for index loading for the test
    if not hasattr(LlamaSettings, 'embed_model') or (hasattr(LlamaSettings.embed_model, 'model_name') and LlamaSettings.embed_model.model_name != _test_settings.embedding_model_name):
         LlamaSettings.embed_model = resolve_embed_model(f"local:{_test_settings.embedding_model_name}")

    logger.info(f"Test - LlamaSettings.llm: {LlamaSettings.llm}")
    logger.info(f"Test - LlamaSettings.embed_model: {LlamaSettings.embed_model}")

    test_chat_engine = initialize_chat_engine()
    if test_chat_engine:
        print("Streaming chat engine initialized for testing. Enter queries or type 'exit'.")
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                break
            print("AI: ", end="")
            for token in stream_chat_response(user_query, test_chat_engine):
                print(token, end="", flush=True)
            print() # Newline after full response
    else:
        print("Failed to initialize chat engine for testing.")
    logger.info("--- QA Service Test (Streaming) Complete ---") 