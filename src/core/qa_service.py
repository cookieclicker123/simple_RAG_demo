# Service for Question Answering: conversational retrieval, memory, and LLM interaction.

import logging
from typing import Generator, List, Union, Optional
from pathlib import Path
from datetime import datetime, timezone
import pickle

# LlamaIndex imports
from llama_index.core import Settings as LlamaSettings
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core import get_response_synthesizer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# Configuration and utility imports
from src.config import settings as app_settings
from src.utils.vector_store_handlers import load_faiss_index_from_storage
from src.utils.llm_handlers import llm_manager
from src.utils.citation_handlers import citation_processor
from src.utils.rag_handlers import rag_processor
from src.models import DocumentCitation

logger = logging.getLogger(__name__)

def initialize_chat_engine() -> BaseChatEngine | None:
    """Initialize chat engine with reranking and hybrid search."""
    logger.info("Initializing chat engine with reranking and hybrid search...")

    # Configure LLM first
    if not llm_manager.configure_llama_settings():
        logger.error("Failed to configure LLM settings. Cannot initialize chat engine.")
        return None

    # Setup debug handler
    llama_debug_handler = LlamaDebugHandler(print_trace_on_end=False)
    logger.info("LlamaDebugHandler activated. Detailed LlamaIndex logs will NOT auto-print on end.")

    # Load vector index
    logger.info(f"Loading vector index and storage context from: {app_settings.vector_store_path}")
    vector_index, storage_context = load_faiss_index_from_storage(
        vector_store_path_str=app_settings.vector_store_path
    )

    if not vector_index or not storage_context or not storage_context.docstore:
        logger.error("Failed to load vector index, storage context, or docstore. Cannot initialize chat engine.")
        return None
    logger.info("Vector index and storage context (with docstore) loaded successfully.")

    # Create retrievers
    vector_retriever = vector_index.as_retriever(similarity_top_k=app_settings.retriever_similarity_top_k)
    logger.info(f"Vector retriever created with similarity_top_k={app_settings.retriever_similarity_top_k}.")

    bm25_retriever = _create_bm25_retriever(storage_context)
    if not bm25_retriever:
        return None

    # Create hybrid retriever
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=app_settings.retriever_similarity_top_k,
        num_queries=1,
        mode="reciprocal_rerank", 
        use_async=False,
        verbose=True,
        callback_manager=CallbackManager([llama_debug_handler])
    )
    logger.info("QueryFusionRetriever (hybrid) created.")

    # Initialize reranker
    reranker = SentenceTransformerRerank(
        model=app_settings.reranker_model_name, 
        top_n=app_settings.reranker_top_n
    )
    logger.info(f"SentenceTransformerRerank initialized with model='{app_settings.reranker_model_name}', top_n={app_settings.reranker_top_n}.")

    # Create query engine
    try:
        logger.info("Initializing RetrieverQueryEngine with hybrid retriever and reranker...")
        
        response_synthesizer = get_response_synthesizer(
            llm=LlamaSettings.llm,
            streaming=True,
            callback_manager=CallbackManager([llama_debug_handler])
        )
        logger.info("Streaming ResponseSynthesizer created.")

        query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[reranker]
        )
        logger.info("RetrieverQueryEngine initialized successfully with hybrid retriever, reranker, and streaming synthesizer.")
        return query_engine # type: ignore 

    except Exception as e:
        logger.error(f"Error initializing RetrieverQueryEngine: {e}", exc_info=True)
        return None

def _create_bm25_retriever(storage_context) -> Optional[BM25Retriever]:
    """Create BM25 retriever from storage context."""
    logger.info("Initializing BM25Retriever...")
    all_nodes_from_docstore = list(storage_context.docstore.docs.values())
    if not all_nodes_from_docstore:
        logger.error("Docstore is empty, cannot initialize BM25Retriever.")
        return None

    bm25_engine_path = Path(app_settings.vector_store_path) / "bm25_engine.pkl"
    
    # Try to load persisted BM25 engine
    if bm25_engine_path.exists():
        logger.info(f"Found persisted BM25 engine at: {bm25_engine_path}. Attempting to load.")
        try:
            with open(bm25_engine_path, "rb") as f:
                loaded_bm25_engine = pickle.load(f)
            
            if loaded_bm25_engine:
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=all_nodes_from_docstore, 
                    similarity_top_k=app_settings.bm25_similarity_top_k,
                    tokenizer=None 
                )
                bm25_retriever.bm25 = loaded_bm25_engine
                logger.info("Successfully loaded and configured BM25Retriever with persisted BM25 engine.")
                return bm25_retriever
            else:
                logger.warning(f"Loaded object from {bm25_engine_path} was None or empty. Falling back to on-the-fly creation.")
        except Exception as e:
            logger.error(f"Error loading persisted BM25 engine from {bm25_engine_path}: {e}. Falling back to on-the-fly creation.", exc_info=True)

    # Create BM25 retriever on-the-fly
    logger.info("Creating BM25Retriever on-the-fly...")
    try:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes_from_docstore,
            similarity_top_k=app_settings.bm25_similarity_top_k,
            tokenizer=None
        )
        logger.info(f"BM25Retriever created successfully on-the-fly from {len(all_nodes_from_docstore)} nodes.")
        return bm25_retriever
    except Exception as e:
        logger.error(f"Failed to create BM25Retriever on-the-fly: {e}", exc_info=True)
        return None

def stream_chat_response(query: str, chat_engine: BaseChatEngine) -> Generator[Union[str, List[DocumentCitation]], None, None]:
    """Stream chat response with citations using utilities."""
    logger.info(f"User query (for streaming with citations): {query}")
    try:
        if not isinstance(chat_engine, RetrieverQueryEngine):
            logger.error(f"Engine is not a RetrieverQueryEngine. Type: {type(chat_engine)}. Cannot process.")
            yield "Error: Chat engine is not correctly configured."
            yield []
            return

        logger.info("Using RetrieverQueryEngine directly, with response_gen for streaming...")
        query_start_time = datetime.now(timezone.utc)
        original_query = query
        response = chat_engine.query(query)
        
        # Process streaming response
        full_response_text_parts = []
        if hasattr(response, 'response_gen') and response.response_gen:
            for token in response.response_gen:
                full_response_text_parts.append(token)
                yield token
        elif response.response:
            logger.warning("RetrieverQueryEngine response does not have response_gen, yielding full response text.")
            full_response_text_parts.append(str(response.response))
            yield str(response.response)
        else:
            logger.warning("RetrieverQueryEngine response has no response_gen and no response text.")
            err_msg = "Error: Could not get a streamable or complete response from query engine."
            full_response_text_parts.append(err_msg)
            yield err_msg
        
        final_answer_text = "".join(full_response_text_parts)

        # Extract BM25 snippets for observability
        bm25_raw_snippets = rag_processor.extract_bm25_snippets(chat_engine, original_query)

        # Process source nodes and create citations
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        citations_generated: List[DocumentCitation] = []
        source_node_details = []

        llm_for_titles_instance = LlamaSettings.llm
        if not isinstance(llm_for_titles_instance, LlamaIndexOpenAI):
            logger.warning(f"LlamaSettings.llm (type: {type(llm_for_titles_instance)}) is not LlamaIndexOpenAI. Title extraction may use fallback.")

        if source_nodes:
            # Create citations using utility
            citations_generated = citation_processor.create_document_citations(
                source_nodes, llm_for_titles_instance # type: ignore
            )
            
            # Create source node details using utility
            source_node_details = citation_processor.create_rag_source_details(
                source_nodes, citations_generated, llm_for_titles_instance # type: ignore
            )
        
        # Create and save RAG response using utility
        rag_response_data = rag_processor.create_rag_response(
            query_start_time=query_start_time,
            original_query=original_query,
            final_answer_text=final_answer_text,
            source_nodes=source_nodes,
            source_node_details=source_node_details,
            citations_generated=citations_generated,
            bm25_raw_snippets=bm25_raw_snippets
        )
        
        rag_processor.save_rag_response(rag_response_data)
        
        yield "QA_SERVICE_STREAM_ENDED_SENTINEL"
        return 
        
    except Exception as e:
        logger.error(f"Error getting streaming chat response with citations: {e}", exc_info=True)
        yield "Sorry, I encountered an error while processing your request."
        yield [] 
    finally:
        pass