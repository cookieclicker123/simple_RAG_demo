# Service for Question Answering: conversational retrieval, memory, and LLM interaction.

import logging
from typing import Generator, List, Union, Dict, Optional
from pathlib import Path
from datetime import datetime, timezone
import json
import math # Added for sigmoid function
import pickle # Re-adding for BM25 persistence

# LlamaIndex imports
from llama_index.core import  Settings as LlamaSettings
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core import get_response_synthesizer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from rank_bm25 import BM25Okapi # For type hinting the loaded object

# Configuration and utility imports (using src. prefix as per your current setup)
from src.config import settings as app_settings
from src.utils.vector_store_handlers import load_faiss_index_from_storage
from src.models import DocumentCitation, RAGResponse, RAGSourceNodeDetail
from src.prompts.investor_prompt import INVESTOR_SYSTEM_PROMPT # Added import

logger = logging.getLogger(__name__)
# logging.basicConfig(level=settings.log_level) # BasicConfig should ideally be called once

# Define project root and tmp directory path
# Assuming this file (qa_service.py) is in src/core/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"

# Ensure TMP_DIR exists when module is loaded
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Explicitly configure/re-configure LlamaSettings.llm based on this module's view of settings.
# This ensures that even if another module (like indexing_service) set it first with a different
# value (e.g., from an old .env or different default), qa_service uses its configured LLM.
current_llm_model_in_settings = app_settings.llm_model_name
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
            temperature=app_settings.temperature,
            api_key=app_settings.openai_api_key,
            max_tokens=app_settings.max_tokens,
            system_prompt=INVESTOR_SYSTEM_PROMPT
        )
    except Exception as e:
        logger.error(f"Failed to configure LlamaSettings.llm with {current_llm_model_in_settings}: {e}", exc_info=True)
        if not LlamaSettings.llm:
            logger.warning("Falling back to a default gpt-3.5-turbo due to configuration error during specific setup.")
            LlamaSettings.llm = LlamaIndexOpenAI(
                model="gpt-3.5-turbo", 
                temperature=app_settings.temperature, 
                api_key=app_settings.openai_api_key,
                system_prompt=INVESTOR_SYSTEM_PROMPT
            )
        # If LlamaSettings.llm was already set to something else before this specific configuration attempt and failed,
        # it would retain its previous value. This block ensures a fallback ONLY if it's None after the try-except.

def initialize_chat_engine() -> BaseChatEngine | None:
    logger.info("Initializing chat engine with reranking and hybrid search...")

    # Setup LlamaDebugHandler for detailed logging
    llama_debug_handler = LlamaDebugHandler(print_trace_on_end=False)
    logger.info(f"LlamaDebugHandler activated. Detailed LlamaIndex logs will NOT auto-print on end.")

    logger.info(f"Loading vector index and storage context from: {app_settings.vector_store_path}")
    vector_index, storage_context = load_faiss_index_from_storage(
        vector_store_path_str=app_settings.vector_store_path
    )

    if not vector_index or not storage_context or not storage_context.docstore:
        logger.error("Failed to load vector index, storage context, or docstore. Cannot initialize chat engine.")
        return None
    logger.info("Vector index and storage context (with docstore) loaded successfully.")

    # 1.a Create a dense retriever (from existing FAISS index)
    vector_retriever = vector_index.as_retriever(similarity_top_k=app_settings.retriever_similarity_top_k)
    logger.info(f"Vector retriever created with similarity_top_k={app_settings.retriever_similarity_top_k}.")

    # 1.b Create BM25 sparse retriever on-the-fly from the loaded docstore
    logger.info("Initializing BM25Retriever...")
    all_nodes_from_docstore = list(storage_context.docstore.docs.values())
    if not all_nodes_from_docstore:
        logger.error("Docstore is empty, cannot initialize BM25Retriever.")
        return None

    bm25_retriever: Optional[BM25Retriever] = None
    bm25_engine_path = Path(app_settings.vector_store_path) / "bm25_engine.pkl"
    
    if bm25_engine_path.exists():
        logger.info(f"Found persisted BM25 engine at: {bm25_engine_path}. Attempting to load.")
        try:
            with open(bm25_engine_path, "rb") as f:
                loaded_bm25_engine = pickle.load(f)
            
            # We expect loaded_bm25_engine to be a BM25Okapi (or similar rank_bm25) instance.
            # We need to check its type if we want to be very specific, but often direct assignment works if the pickled object is correct.
            if loaded_bm25_engine: # Basic check that something was loaded
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=all_nodes_from_docstore, 
                    similarity_top_k=app_settings.bm25_similarity_top_k,
                    tokenizer=None 
                )
                bm25_retriever.bm25 = loaded_bm25_engine # Assign loaded engine to .bm25
                logger.info("Successfully loaded and configured BM25Retriever with persisted BM25 engine (using .bm25).")
            else:
                logger.warning(f"Loaded object from {bm25_engine_path} was None or empty. Falling back to on-the-fly creation.")
                bm25_retriever = None # Ensure fallback
        except Exception as e_bm25_load:
            logger.error(f"Error loading persisted BM25 engine from {bm25_engine_path}: {e_bm25_load}. Falling back to on-the-fly creation.", exc_info=True)
            bm25_retriever = None # Ensure fallback

    if bm25_retriever is None:
        logger.info("Persisted BM25 engine not loaded/found. Creating BM25Retriever on-the-fly...")
        try:
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=all_nodes_from_docstore,
                similarity_top_k=app_settings.bm25_similarity_top_k,
                tokenizer=None
            )
            logger.info(f"BM25Retriever created successfully on-the-fly from {len(all_nodes_from_docstore)} nodes.")
        except Exception as e_bm25_create:
            logger.error(f"Failed to create BM25Retriever on-the-fly: {e_bm25_create}", exc_info=True)
            return None
    

    # 1.c Create a QueryFusionRetriever for hybrid search
    # The QueryFusionRetriever will combine results before they go to the reranker.
    # The similarity_top_k for QueryFusionRetriever is how many results it passes ON, 
    # so it should be >= the individual retriever top_k values if you want to give the fusion process enough candidates.
    # Let's make it sum of both, or a fixed larger number like our main retriever_similarity_top_k.
    # For now, we'll set it to the main dense retriever's top_k, assuming it's the desired number of candidates for reranking.
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=app_settings.retriever_similarity_top_k, # Number of results after fusion
        num_queries=1,  # Set to > 1 for query expansion by the fusion retriever
        mode="reciprocal_rerank", 
        use_async=False, # Keep it sync for now with our current setup
        verbose=True, # Enable verbose logging for the fusion retriever
        callback_manager=CallbackManager([llama_debug_handler]) # For LlamaDebugHandler
    )
    logger.info("QueryFusionRetriever (hybrid) created.")

    # 2. Initialize the reranker (cross-encoder)
    reranker = SentenceTransformerRerank(
        model=app_settings.reranker_model_name, 
        top_n=app_settings.reranker_top_n # This is the final top_n after reranking the hybrid results
    )
    logger.info(f"SentenceTransformerRerank (cross-encoder) initialized with model='{app_settings.reranker_model_name}', top_n={app_settings.reranker_top_n}.")

    try:
        logger.info("Initializing RetrieverQueryEngine with hybrid retriever and reranker...")
        
        response_synthesizer = get_response_synthesizer(
            llm=LlamaSettings.llm,
            streaming=True,
            callback_manager=CallbackManager([llama_debug_handler])
        )
        logger.info("Streaming ResponseSynthesizer created.")

        query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever, # Use the hybrid retriever
            response_synthesizer=response_synthesizer,
            node_postprocessors=[reranker] # Reranker applied after hybrid retrieval + fusion
        )
        logger.info("RetrieverQueryEngine initialized successfully with hybrid retriever, reranker, and streaming synthesizer.")
        return query_engine # type: ignore 

    except Exception as e:
        logger.error(f"Error initializing RetrieverQueryEngine: {e}", exc_info=True)
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
    # Accessing and printing debug information would ideally happen after the call.
    # For now, the LlamaDebugHandler is set to print_trace_on_end=True, which should output to console.
    # We can add more specific logging of its contents later if needed.
    try:
        # Ensure this path is for RetrieverQueryEngine
        if isinstance(chat_engine, RetrieverQueryEngine):
            logger.info("Using RetrieverQueryEngine directly, with response_gen for streaming...")
            
            query_start_time = datetime.now(timezone.utc)
            original_query = query # Capture original query
            condensed_query_text = None # Placeholder, RQE doesn't readily expose this

            response = chat_engine.query(query) # Call query(), expect Response object
            
            full_response_text_parts = []
            # Yield tokens from the response_gen attribute
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
                full_response_text_parts.append("Error: Could not get a streamable or complete response from query engine.")
                yield "Error: Could not get a streamable or complete response from query engine."
            
            final_answer_text = "".join(full_response_text_parts)

            # --- Observe raw BM25 results (for logging/debug) ---
            bm25_raw_snippets_for_log: Optional[List[str]] = None
            # Attempt to get the bm25_retriever from the hybrid retriever
            if isinstance(chat_engine.retriever, QueryFusionRetriever):
                actual_bm25_retriever = None
                # Try accessing via _retrievers (common private attribute name)
                if hasattr(chat_engine.retriever, '_retrievers'):
                    for r in chat_engine.retriever._retrievers:
                        if isinstance(r, BM25Retriever):
                            actual_bm25_retriever = r
                            break
                else:
                    logger.warning("QA_SERVICE: QueryFusionRetriever does not have '_retrievers' attribute.")
                
                if actual_bm25_retriever:
                    logger.info("QA_SERVICE: Retrieving raw BM25 results for observability...")
                    try:
                        # Retrieve top N (e.g., 3) raw results from BM25
                        # Use a different similarity_top_k for this observability query if needed,
                        # or rely on its configured one.
                        # For simplicity, let's assume its default similarity_top_k is reasonable for a peek.
                        # To be more explicit, we could set it: actual_bm25_retriever.similarity_top_k = 3
                        raw_bm25_nodes = actual_bm25_retriever.retrieve(original_query)
                        if raw_bm25_nodes:
                            bm25_raw_snippets_for_log = []
                            logger.info(f"QA_SERVICE: Raw BM25 nodes found: {len(raw_bm25_nodes)}")
                            for i, node_ws in enumerate(raw_bm25_nodes[:3]): # Log top 3 snippets
                                # Include full content for the snippet in the JSON
                                snippet = f"BM25 Raw {i+1} (Score: {node_ws.score:.4f}): {node_ws.node.get_content()}" 
                                bm25_raw_snippets_for_log.append(snippet)
                                # For console logging, still truncate to avoid flooding the console
                                logger.info(f"  Snippet for RAGResponse (console log truncated): BM25 Raw {i+1} (Score: {node_ws.score:.4f}): {node_ws.node.get_content()[:200]}...") 
                        else:
                            logger.info("QA_SERVICE: No raw results from BM25 retriever for this query.")
                    except Exception as e_bm25_raw:
                        logger.error(f"QA_SERVICE: Error retrieving raw BM25 results: {e_bm25_raw}", exc_info=True)
                else:
                    logger.warning("QA_SERVICE: BM25Retriever not found within the QueryFusionRetriever.")
            else:
                logger.warning("QA_SERVICE: Chat engine's retriever is not a QueryFusionRetriever. Cannot get raw BM25 results.")
            # --- End Observe raw BM25 results ---

            # Log and process source nodes after streaming is complete (or response is obtained)
            source_nodes = response.source_nodes
            source_node_details_for_rag: List[RAGSourceNodeDetail] = []

            # Process source nodes for RAGResponse details AND for citations (titles are linked)
            citations_generated: List[DocumentCitation] = [] # Initialize here
            llm_for_titles_instance = LlamaSettings.llm
            if not isinstance(llm_for_titles_instance, LlamaIndexOpenAI):
                 logger.warning(f"LlamaSettings.llm (type: {type(llm_for_titles_instance)}) is not LlamaIndexOpenAI. Title extraction for RAGSourceNodeDetail might fail or use fallback.")

            if source_nodes:
                # First, create citations which also generates/extracts titles
                citations_generated = _create_document_citations(source_nodes, llm_for_titles_instance) # type: ignore
                
                # Now, populate RAGSourceNodeDetail using titles from citations where possible
                for i, node_ws in enumerate(source_nodes):
                    node_content = node_ws.node.get_content()
                    node_title = "N/A"
                    # Try to find the corresponding citation for this node to reuse its title
                    if i < len(citations_generated) and \
                       citations_generated[i].file_path == node_ws.node.metadata.get("file_path") and \
                       citations_generated[i].page_label == node_ws.node.metadata.get("page_label"):
                       node_title = citations_generated[i].document_title or "N/A"
                    else: # Fallback if no match or out of bounds, try to generate title for the current node
                        logger.warning(f"Citation not matched or out of bounds for node {i}, attempting direct title generation.")
                        if isinstance(llm_for_titles_instance, LlamaIndexOpenAI):
                            generated_title_for_node = _get_title_for_chunk_from_llm(node_content, llm_for_titles_instance, Path(node_ws.node.metadata.get("file_path", "Unknown Document")).stem)
                            if generated_title_for_node: 
                                node_title = generated_title_for_node 
                    
                    normalized_node_score = None
                    if node_ws.score is not None:
                        try:
                            normalized_node_score = 1 / (1 + math.exp(-node_ws.score)) # Sigmoid normalization
                        except OverflowError:
                            # Handle extreme scores that might cause overflow in math.exp()
                            # If -node_ws.score is very large (node_ws.score very negative), exp result is huge -> normalized_score is near 0
                            # If -node_ws.score is very small (node_ws.score very positive), exp result is near 0 -> normalized_score is near 1
                            normalized_node_score = 0.0 if node_ws.score < 0 else 1.0 
                            logger.warning(f"OverflowError during sigmoid normalization for score {node_ws.score}. Assigned {normalized_node_score}")

                    source_node_details_for_rag.append(
                        RAGSourceNodeDetail(
                            node_id=node_ws.node.node_id,
                            file_path=node_ws.node.metadata.get("file_path"),
                            page_label=node_ws.node.metadata.get("page_label"),
                            score=normalized_node_score, # Store normalized score
                            node_title=node_title, 
                            full_text_content=node_content # Store full content
                        )
                    )
            # Do NOT yield citations_generated here anymore to keep terminal clean
            # yield citations_generated 

            query_end_time = datetime.now(timezone.utc)
            processing_duration = (query_end_time - query_start_time).total_seconds()

            rag_response_data = RAGResponse(
                query_time_utc=query_start_time,
                response_time_utc=query_end_time,
                processing_duration_seconds=processing_duration,
                original_query=original_query,
                llm_model_used=LlamaSettings.llm.model if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model') else str(LlamaSettings.llm),
                embedding_model_name_config=app_settings.embedding_model_name,
                retriever_similarity_top_k_config=app_settings.retriever_similarity_top_k,
                reranker_model_name_config=app_settings.reranker_model_name,
                reranker_top_n_config=app_settings.reranker_top_n,
                source_nodes_count_after_reranking=len(source_nodes) if source_nodes else 0,
                source_nodes_details=source_node_details_for_rag,
                final_answer_text=final_answer_text,
                citations_generated=citations_generated,
                bm25_raw_retrieved_snippets=bm25_raw_snippets_for_log
            )

            output_file_path = TMP_DIR / "latest_rag_response.json"
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(rag_response_data.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
                logger.info(f"RAG response details saved to: {output_file_path}")
            except Exception as e:
                logger.error(f"Failed to write RAG response to JSON: {e}", exc_info=True)
            
            logger.info("QA_SERVICE: stream_chat_response is ABOUT TO YIELD FINAL SENTINEL BEFORE RETURN")
            yield "QA_SERVICE_STREAM_ENDED_SENTINEL"
            logger.info("QA_SERVICE: stream_chat_response is ABOUT TO RETURN")
            return 
        else: # Should not happen if initialize_chat_engine only returns RetrieverQueryEngine or None
            logger.error(f"Engine is not a RetrieverQueryEngine. Type: {type(chat_engine)}. Cannot process.")
            yield "Error: Chat engine is not correctly configured."
            yield []
            return

        # Code for StreamingAgentChatResponse (CondensePlusContextChatEngine) is now effectively bypassed
        # streaming_response: StreamingAgentChatResponse = chat_engine.stream_chat(query)
        
        # # Log the content of source nodes after reranking
        # if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
        #     logger.info("--- CONTENT OF RERANKED SOURCE NODES ---")
        #     for i, node_with_score in enumerate(streaming_response.source_nodes):
        #         logger.info(f"Source Node {i+1} (Score: {node_with_score.score:.4f}):\n"
        #                     f"  ID: {node_with_score.node.node_id}\n"
        #                     f"  File: {node_with_score.node.metadata.get('file_path', 'N/A')}\n"
        #                     f"  Page: {node_with_score.node.metadata.get('page_label', 'N/A')}\n"
        #                     f"  First 200 chars: {node_with_score.node.get_content()[:200]}..."
        #         )
        #     logger.info("--- END CONTENT OF RERANKED SOURCE NODES ---")
        # else:
        #     logger.info("No source nodes found in streaming_response to log content.")

        # Yield tokens from the generator
        # if hasattr(streaming_response, 'response_gen'):
        #     for token in streaming_response.response_gen:
        #         yield token
        # else:
        #     logger.warning("Streaming response object does not have 'response_gen'.")
        #     if hasattr(streaming_response, 'response') and streaming_response.response:
        #          yield str(streaming_response.response) # Yield the whole response as one chunk
        #     else:
        #         yield "Error: Could not get a streamable or complete response."
        #         # Early exit if no response parts can be obtained.
        #         # Process source nodes even if response_gen is missing but source_nodes exist.

        # After all tokens, process source nodes for citations
        # citations = []
        # if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
        #     logger.info("--- RERANKING CONFIRMATION ---")
        #     logger.info(f"Initial retrieval configured for top_k={settings.retriever_similarity_top_k} candidates.")
        #     logger.info(f"Reranker model='{settings.reranker_model_name}' configured for top_n={settings.reranker_top_n}.")
        #     logger.info(f"After reranking, {len(streaming_response.source_nodes)} nodes remain for citation processing.")
        #     logger.info("--- END RERANKING CONFIRMATION ---")
        #     # Ensure LlamaSettings.llm is appropriate for title extraction (OpenAI compatible for .chat)
        #     llm_for_titles = LlamaSettings.llm 
        #     if not isinstance(llm_for_titles, LlamaIndexOpenAI):
        #          logger.warning(f"LlamaSettings.llm (type: {type(llm_for_titles)}) is not LlamaIndexOpenAI. Title extraction might fail or use fallback.")
        #          # Optionally, create a specific OpenAI instance here if LlamaSettings.llm is not suitable.
        #          # For now, we pass it and _get_title_for_chunk_from_llm will handle it or fallback.
        #     citations = _create_document_citations(streaming_response.source_nodes, llm_for_titles)
        # else:
        #     logger.info("No source nodes found in streaming response for citations.")
        
        # yield citations # Yield the list of citation objects

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
    if not app_settings.openai_api_key or app_settings.openai_api_key == "your_openai_api_key_here_if_not_in_env":
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
            max_tokens=_test_settings.max_tokens,
            system_prompt=INVESTOR_SYSTEM_PROMPT
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