"""Document citation generation and processing utilities."""

import logging
import math
from typing import List, Dict, Optional
from pathlib import Path

from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from src.models import DocumentCitation, RAGSourceNodeDetail
from src.utils.llm_handlers import title_generator

logger = logging.getLogger(__name__)

class CitationProcessor:
    """Utilities for creating and processing document citations."""
    
    @staticmethod
    def create_document_citations(
        source_nodes: List[NodeWithScore], 
        llm_for_titles: LlamaIndexOpenAI
    ) -> List[DocumentCitation]:
        """Create DocumentCitation objects from source nodes with title generation."""
        citations = []
        if not source_nodes:
            return citations

        # Cache titles per file_path for this request to avoid duplicate LLM calls
        processed_files_titles: Dict[str, str] = {}

        for node in source_nodes:
            metadata = node.node.metadata or {}
            file_path_str = metadata.get("file_path")
            
            document_name = Path(file_path_str).stem if file_path_str else "Unknown Document"
            page_label = metadata.get("page_label", "N/A")
            snippet_text = CitationProcessor._create_snippet(node.node.get_content())
            
            # Generate or retrieve cached title
            doc_title = CitationProcessor._get_document_title(
                file_path_str, 
                node.node.get_content(),
                document_name,
                llm_for_titles,
                processed_files_titles
            )
            
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
    
    @staticmethod
    def create_rag_source_details(
        source_nodes: List[NodeWithScore],
        citations: List[DocumentCitation],
        llm_for_titles: LlamaIndexOpenAI
    ) -> List[RAGSourceNodeDetail]:
        """Create RAGSourceNodeDetail objects from source nodes."""
        source_node_details = []
        
        for i, node_ws in enumerate(source_nodes):
            node_content = node_ws.node.get_content()
            node_metadata = node_ws.node.metadata or {}
            
            # Try to get title from corresponding citation
            node_title = CitationProcessor._get_node_title(
                i, citations, node_metadata, node_content, llm_for_titles
            )
            
            # Normalize score using sigmoid
            normalized_score = CitationProcessor.normalize_score(node_ws.score)
            
            source_node_details.append(
                RAGSourceNodeDetail(
                    node_id=node_ws.node.node_id,
                    file_path=node_metadata.get("file_path"),
                    page_label=str(node_metadata.get("page_label", "N/A")), 
                    score=normalized_score,
                    node_title=node_title, 
                    full_text_content=node_content
                )
            )
        
        return source_node_details
    
    @staticmethod
    def normalize_score(score: Optional[float]) -> Optional[float]:
        """Normalize a score using sigmoid function."""
        if score is None:
            return None
            
        try: 
            return 1 / (1 + math.exp(-score))  # Sigmoid normalization
        except OverflowError:
            normalized = 0.0 if score < 0 else 1.0 
            logger.warning(f"OverflowError during sigmoid normalization for score {score}. Assigned {normalized}")
            return normalized
    
    @staticmethod
    def _create_snippet(content: str) -> str:
        """Create a snippet from node content."""
        if not content:
            return "N/A"
        return content[:200] + "..." if len(content) > 200 else content
    
    @staticmethod
    def _get_document_title(
        file_path_str: Optional[str],
        node_content: str,
        document_name: str,
        llm_for_titles: LlamaIndexOpenAI,
        processed_files_titles: Dict[str, str]
    ) -> str:
        """Get or generate document title with caching."""
        doc_title = document_name  # Default title
        
        if file_path_str:
            if file_path_str in processed_files_titles:
                doc_title = processed_files_titles[file_path_str]
                logger.info(f"Using cached title for {file_path_str}: '{doc_title}'")
            else:
                extracted_title = title_generator.generate_title_from_chunk(
                    node_content, llm_for_titles, document_name
                )
                if extracted_title:
                    doc_title = extracted_title
                    processed_files_titles[file_path_str] = doc_title  # Cache it
        
        return doc_title
    
    @staticmethod
    def _get_node_title(
        index: int,
        citations: List[DocumentCitation],
        node_metadata: Dict,
        node_content: str,
        llm_for_titles: LlamaIndexOpenAI
    ) -> str:
        """Get title for a node, either from citations or generate new one."""
        # Try to match with corresponding citation
        if (index < len(citations) and 
            citations[index].file_path == node_metadata.get("file_path") and 
            str(citations[index].page_label) == str(node_metadata.get("page_label"))):
            return citations[index].document_title or "N/A"
        
        # Generate new title if no match
        if isinstance(llm_for_titles, LlamaIndexOpenAI):
            filename_stem = Path(node_metadata.get("file_path", "Unknown Document")).stem
            generated_title = title_generator.generate_title_from_chunk(
                node_content, llm_for_titles, filename_stem
            )
            if generated_title:
                return generated_title
        
        return "N/A"

# Convenience instance
citation_processor = CitationProcessor() 