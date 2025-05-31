"""RAG response processing and logging utilities."""

import logging
import json
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import Settings as LlamaSettings
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from src.models import RAGResponse, DocumentCitation, RAGSourceNodeDetail
from src.config import settings

logger = logging.getLogger(__name__)

class RAGResponseProcessor:
    """Utilities for processing and logging RAG responses."""
    
    def __init__(self, tmp_dir: Optional[Path] = None):
        self.tmp_dir = tmp_dir or Path(__file__).resolve().parent.parent.parent / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_rag_response(
        self,
        query_start_time: datetime,
        original_query: str,
        final_answer_text: str,
        source_nodes: List[NodeWithScore],
        source_node_details: List[RAGSourceNodeDetail],
        citations_generated: List[DocumentCitation],
        bm25_raw_snippets: Optional[List[str]] = None
    ) -> RAGResponse:
        """Create a complete RAG response object."""
        query_end_time = datetime.now(timezone.utc)
        
        return RAGResponse(
            query_time_utc=query_start_time,
            response_time_utc=query_end_time,
            processing_duration_seconds=(query_end_time - query_start_time).total_seconds(),
            original_query=original_query,
            llm_model_used=self._get_llm_model_name(),
            embedding_model_name_config=settings.embedding_model_name,
            retriever_similarity_top_k_config=settings.retriever_similarity_top_k,
            reranker_model_name_config=settings.reranker_model_name,
            reranker_top_n_config=settings.reranker_top_n,
            source_nodes_count_after_reranking=len(source_nodes) if source_nodes else 0,
            source_nodes_details=source_node_details,
            final_answer_text=final_answer_text,
            citations_generated=citations_generated,
            bm25_raw_retrieved_snippets=bm25_raw_snippets,
            expanded_queries=None  # Explicitly None as query expansion is not active
        )
    
    def save_rag_response(self, rag_response: RAGResponse) -> bool:
        """Save RAG response to JSON file."""
        output_file_path = self.tmp_dir / "latest_rag_response.json"
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(rag_response.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
            logger.info(f"RAG response details saved to: {output_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write RAG response to JSON: {e}", exc_info=True)
            return False
    
    def extract_bm25_snippets(self, chat_engine, original_query: str) -> Optional[List[str]]:
        """Extract BM25 raw snippets for observability."""
        if not isinstance(chat_engine.retriever, QueryFusionRetriever):
            logger.warning("Chat engine's retriever is not a QueryFusionRetriever. Cannot get raw BM25 results.")
            return None
        
        actual_bm25_retriever = None
        if hasattr(chat_engine.retriever, '_retrievers'):
            for r in chat_engine.retriever._retrievers:
                if isinstance(r, BM25Retriever):
                    actual_bm25_retriever = r
                    break
        
        if not actual_bm25_retriever:
            logger.warning("BM25Retriever not found within the QueryFusionRetriever.")
            return None
        
        logger.info("Retrieving raw BM25 results for observability...")
        try:
            raw_bm25_nodes = actual_bm25_retriever.retrieve(original_query)
            if raw_bm25_nodes:
                bm25_raw_snippets = []
                logger.info(f"Raw BM25 nodes found: {len(raw_bm25_nodes)}")
                for i, node_ws in enumerate(raw_bm25_nodes[:3]):
                    snippet = f"BM25 Raw {i+1} (Score: {node_ws.score:.4f}): {node_ws.node.get_content()}"
                    bm25_raw_snippets.append(snippet)
                    logger.info(f"Snippet for RAGResponse (console log truncated): BM25 Raw {i+1} (Score: {node_ws.score:.4f}): {node_ws.node.get_content()[:200]}...")
                return bm25_raw_snippets
            else:
                logger.info("No raw results from BM25 retriever for this query.")
                return None
        except Exception as e:
            logger.error(f"Error retrieving raw BM25 results: {e}", exc_info=True)
            return None
    
    def _get_llm_model_name(self) -> str:
        """Get the current LLM model name."""
        if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model'):
            return LlamaSettings.llm.model
        return str(LlamaSettings.llm) if LlamaSettings.llm else "unknown"

class ResponseStreamProcessor:
    """Utilities for processing streaming responses."""
    
    @staticmethod
    def process_streaming_response(response):
        """
        Process a streaming response and yield tokens, return final text.
        Generator that yields individual response tokens as strings.
        """
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
        
        # Note: final_answer_text would be available via "".join(full_response_text_parts)
        # but since this is a generator, the caller should collect the tokens themselves

# Convenience instances
rag_processor = RAGResponseProcessor()
stream_processor = ResponseStreamProcessor() 