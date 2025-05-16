# Service for Question Answering: conversational retrieval, memory, and LLM interaction.

import logging
from typing import Generator, List, Union, Dict
from pathlib import Path

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings as LlamaSettings
from llama_index.core.chat_engine.types import BaseChatEngine, StreamingAgentChatResponse
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.llms import ChatMessage, MessageRole

# LangChain imports for memory (though LlamaIndex chat engine can manage its own)
# from langchain.memory import ConversationBufferMemory # Option 1: LangChain memory
from llama_index.core.memory import ChatMemoryBuffer

# Configuration and utility imports (using src. prefix as per your current setup)
from src.config import settings
from src.utils.vector_store_handlers import load_faiss_index_from_storage
from src.models import DocumentCitation

logger = logging.getLogger(__name__)
# logging.basicConfig(level=settings.log_level) # BasicConfig should ideally be called once

# Explicitly configure/re-configure LlamaSettings.llm based on this module's view of settings.
# This ensures that even if another module (like indexing_service) set it first with a different
# value (e.g., from an old .env or different default), qa_service uses its configured LLM.
current_llm_model_in_settings = settings.llm_model_name
llm_needs_configuration = True
if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model'):
    if LlamaSettings.llm.model == current_llm_model_in_settings:
        llm_needs_configuration = False
    else:
        logger.info(f"LlamaSettings.llm is {LlamaSettings.llm.model}, but config wants {current_llm_model_in_settings}. Re-configuring.")
elif not LlamaSettings.llm:
    logger.info("LlamaSettings.llm not yet configured.")
else:
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
        if not LlamaSettings.llm:
            logger.warning("Falling back to a default gpt-3.5-turbo due to configuration error.")
            LlamaSettings.llm = LlamaIndexOpenAI(model="gpt-3.5-turbo", temperature=settings.temperature, api_key=settings.openai_api_key)

def initialize_chat_engine() -> BaseChatEngine | None:
    logger.info("Initializing chat engine...")
    # logger.info(f"Attempting to use LLM: {LlamaSettings.llm.model if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model') else 'Not configured or unknown type'}") # Reduced noise

    logger.info(f"Loading vector index from: {settings.vector_store_path}")
    vector_index: VectorStoreIndex | None = load_faiss_index_from_storage(
        vector_store_path_str=settings.vector_store_path
    )

    if not vector_index:
        logger.error("Failed to load vector index. Cannot initialize chat engine.")
        return None
    # logger.info(f"Vector index loaded successfully. Index ID: {vector_index.index_id}") # Reduced noise

    memory = ChatMemoryBuffer.from_defaults(token_limit=settings.chat_memory_token_limit if hasattr(settings, 'chat_memory_token_limit') else 3000)
    logger.info(f"Chat memory (ChatMemoryBuffer) initialized with token limit: {memory.token_limit}")

    try:
        chat_engine = vector_index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            verbose=settings.chat_engine_verbose if hasattr(settings, 'chat_engine_verbose') else True,
            system_prompt=(
                "You are a helpful and knowledgeable AI assistant. "
                "Answer the user's questions based on the provided context. "
                "If the context doesn't contain the answer, say that you don't know. "
                "Be concise and informative."
            ),
            streaming=True
        )
        logger.info("Chat engine initialized successfully.")
        return chat_engine
    except Exception as e:
        logger.error(f"Error initializing chat engine: {e}", exc_info=True)
        return None

def _get_title_for_chunk_from_llm(text_chunk: str, llm: LlamaIndexOpenAI, filename_stem: str) -> str | None:
    """Uses the provided LLM to generate a title for a text chunk."""
    if not text_chunk:
        return filename_stem # Fallback to filename stem if chunk is empty
    
    # Using a more specific model for this if needed, or the globally configured one.
    # For simplicity, using the globally configured LlamaSettings.llm.
    # Ensure LlamaSettings.llm is an instance of LlamaIndexOpenAI or compatible.
    if not llm or not hasattr(llm, 'chat'):
        logger.warning("LLM for title extraction is not properly configured or not an OpenAI model. Falling back to filename stem.")
        return filename_stem

    prompt = (
        f"Based on the following text chunk from a document (filename stem: '{filename_stem}'), "
        f"provide a concise and descriptive title for the document this chunk likely belongs to. "
        f"Focus on the primary subject or product name if evident. If a clear title is present within the text, prefer that. "
        f"Return only the title itself, and nothing else. Example: 'Speed Measurement System speedMATE'\n\n"
        f"Text Chunk (first 1500 characters):\n---\n{text_chunk[:1500]}\n---\nTitle:"
    )
    try:
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        # Using llm.chat for a more standard chat completion call if available
        response = llm.chat(messages)
        title = response.message.content.strip()
        # Basic cleaning: remove potential quotes if LLM adds them
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        if title.startswith(''') and title.endswith('''):
            title = title[1:-1]
        logger.info(f"LLM-generated title for chunk from '{filename_stem}': '{title}'")
        return title if title else filename_stem # Fallback if LLM returns empty string
    except Exception as e:
        logger.error(f"Error getting title from LLM for '{filename_stem}': {e}", exc_info=True)
        return filename_stem # Fallback to filename stem

TITLE_CACHE: Dict[str, str] = {} # Simple in-memory cache for titles (file_path -> title)

def _create_document_citations(source_nodes: List[NodeWithScore], llm_for_titles: LlamaIndexOpenAI) -> List[DocumentCitation]:
    citations = []
    if not source_nodes:
        return citations

    # Clear cache for each new query's citations to ensure fresh titles if content changes or for different queries.
    # For a more persistent cache, it would need to be managed outside this function.
    # TITLE_CACHE.clear() # Or manage lifetime more carefully if needed across requests.
    # For this function scope, we can cache per file_path processed in this call.
    processed_files_titles = {}

    for node in source_nodes:
        metadata = node.node.metadata or {}
        file_path_str = metadata.get("file_path")
        
        document_name = Path(file_path_str).stem if file_path_str else "Unknown Document"
        page_label = metadata.get("page_label", "N/A")
        snippet_text = node.node.get_content()[:200] + "..." if node.node.get_content() else "N/A"
        
        doc_title = document_name # Default title
        if file_path_str:
            if file_path_str in processed_files_titles:
                doc_title = processed_files_titles[file_path_str]
                logger.info(f"Using cached title for {file_path_str}: '{doc_title}'")
            else:
                node_text_for_title = node.node.get_content()
                # Use the passed LLM instance for title generation
                extracted_title = _get_title_for_chunk_from_llm(node_text_for_title, llm_for_titles, document_name)
                if extracted_title:
                    doc_title = extracted_title
                    processed_files_titles[file_path_str] = doc_title # Cache it
        
        citations.append(
            DocumentCitation(
                document_name=document_name,
                document_title=doc_title,
                file_path=file_path_str,
                page_label=str(page_label),
                snippet=snippet_text
            )
        )
    return citations

def stream_chat_response(query: str, chat_engine: BaseChatEngine) -> Generator[Union[str, List[DocumentCitation]], None, None]:
    logger.info(f"User query (for streaming with citations): {query}")
    try:
        streaming_response: StreamingAgentChatResponse = chat_engine.stream_chat(query)
        
        # Yield tokens from the generator
        if hasattr(streaming_response, 'response_gen'):
            for token in streaming_response.response_gen:
                yield token
        else:
            logger.warning("Streaming response object does not have 'response_gen'.")
            if hasattr(streaming_response, 'response') and streaming_response.response:
                 yield str(streaming_response.response) # Yield the whole response as one chunk
            else:
                yield "Error: Could not get a streamable or complete response."
                # Early exit if no response parts can be obtained.
                # Process source nodes even if response_gen is missing but source_nodes exist.

        # After all tokens, process source nodes for citations
        citations = []
        if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
            logger.info(f"Processing {len(streaming_response.source_nodes)} source nodes for citations.")
            # Ensure LlamaSettings.llm is appropriate for title extraction (OpenAI compatible for .chat)
            llm_for_titles = LlamaSettings.llm 
            if not isinstance(llm_for_titles, LlamaIndexOpenAI):
                 logger.warning(f"LlamaSettings.llm (type: {type(llm_for_titles)}) is not LlamaIndexOpenAI. Title extraction might fail or use fallback.")
                 # Optionally, create a specific OpenAI instance here if LlamaSettings.llm is not suitable.
                 # For now, we pass it and _get_title_for_chunk_from_llm will handle it or fallback.
            citations = _create_document_citations(streaming_response.source_nodes, llm_for_titles)
        else:
            logger.info("No source nodes found in streaming response for citations.")
        
        yield citations # Yield the list of citation objects

    except Exception as e:
        logger.error(f"Error getting streaming chat response with citations: {e}", exc_info=True)
        yield "Sorry, I encountered an error while processing your request."
        yield [] # Yield empty list for citations in case of error

if __name__ == "__main__":
    # This test block requires an existing index and a configured OpenAI API key.
    import sys
    #from pathlib import Path # Already imported
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.core.indexing_service import get_active_settings as resolve_embed_model_settings, configure_llama_index_globals
    from src.config import AppSettings

    logger.info("--- QA Service Test (Streaming with LLM Citations) --- ")
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here_if_not_in_env":
        logger.warning("OPENAI_API_KEY not found. Test might fail.")
        # exit(1)
    
    _test_settings = AppSettings()
    # Must configure LlamaIndex globals, including embed_model for index loading
    # and the LLM for the chat engine, using current settings.
    configure_llama_index_globals(_test_settings) 
    # The module-level LLM config runs on import. If _test_settings differs (e.g., .env changed),
    # we might need to re-apply LlamaSettings.llm specifically if configure_llama_index_globals doesn't cover it for qa_service's needs.
    # However, configure_llama_index_globals *should* set LlamaSettings.embed_model AND LlamaSettings.chunk_size/overlap.
    # The LLM for qa_service is also set at the module level when this file is imported.
    # To be absolutely sure the test uses the LLM from the reloaded _test_settings:
    if not LlamaSettings.llm or (hasattr(LlamaSettings.llm, 'model') and LlamaSettings.llm.model != _test_settings.llm_model_name):
        logger.info(f"Re-configuring LlamaSettings.llm for test with: {_test_settings.llm_model_name}")
        LlamaSettings.llm = LlamaIndexOpenAI(
            model=_test_settings.llm_model_name,
            temperature=_test_settings.temperature,
            api_key=_test_settings.openai_api_key,
            max_tokens=_test_settings.max_tokens
        )

    logger.info(f"Test - LlamaSettings.llm for QA: {LlamaSettings.llm.model if LlamaSettings.llm else 'None'}")
    logger.info(f"Test - LlamaSettings.embed_model: {LlamaSettings.embed_model}")

    test_chat_engine = initialize_chat_engine()
    if test_chat_engine:
        print("Streaming chat engine initialized. Enter queries or type 'exit'.")
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                break
            print("AI: ", end="", flush=True)
            final_citations = []
            for response_part in stream_chat_response(user_query, test_chat_engine):
                if isinstance(response_part, str):
                    print(response_part, end="", flush=True)
                elif isinstance(response_part, list): # It's the citations list
                    final_citations = response_part
            print() # Newline after full response
            if final_citations:
                print("\n--- Citations (Title from LLM) ---")
                for cit in final_citations:
                    print(f"  ID: {cit.document_id}")
                    print(f"  Name: {cit.document_name}")
                    print(f"  Title: {cit.document_title}")
                    print(f"  Path: {cit.file_path}")
                    print(f"  Page: {cit.page_label}")
                    print(f"  Snippet: {cit.snippet}\n")
            else:
                print("(No citations found or error in processing them)")
    else:
        print("Failed to initialize chat engine for testing.")
    logger.info("--- QA Service Test (Streaming with LLM Citations) Complete ---") 