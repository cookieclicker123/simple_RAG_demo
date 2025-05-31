"""
RAG Tool - Retrieval-Augmented Generation Tool

This module wraps our existing RAG functionality in the standardized tool interface,
allowing it to be seamlessly used by agents in the multi-agent framework.

The RAG tool provides:
- Document retrieval using hybrid search (vector + BM25)
- Context-aware query enhancement
- LLM-powered answer generation
- Document citations and metadata
- Integration with conversation memory

Key Design Principles:
- Reuse existing RAG infrastructure
- Stateless operation for thread safety
- Rich metadata and observability
- Type-safe interfaces
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Union, Optional

from llama_index.core.chat_engine.types import BaseChatEngine

from agents.agent_models import ToolResult, ToolExecutionStatus
from tools.tool_models import (
    ToolSchema, 
    ToolParameter, 
    ToolParameterType,
    ToolMetadata,
    RAGToolInput,
    RAGToolOutput
)
from src.models import ConversationMemory, DocumentCitation
from src.core.qa_service import stream_chat_response_with_memory

logger = logging.getLogger(__name__)

# Global chat engine reference (set by services during initialization)
_global_chat_engine: Optional[BaseChatEngine] = None

def set_global_chat_engine(chat_engine: BaseChatEngine) -> None:
    """
    Set the global chat engine for the RAG tool.
    
    This function should be called by the services module during initialization
    to provide the RAG tool with access to the chat engine.
    
    Args:
        chat_engine: Initialized chat engine instance
    """
    global _global_chat_engine
    _global_chat_engine = chat_engine
    logger.info("RAG tool: Global chat engine reference set")

def get_global_chat_engine() -> Optional[BaseChatEngine]:
    """
    Get the global chat engine for RAG operations.
    
    Returns:
        Global chat engine instance or None if not set
    """
    return _global_chat_engine


# Tool schema definition for RAG capabilities
RAG_TOOL_SCHEMA = ToolSchema(
    tool_name="rag_tool",
    display_name="RAG Document Search and Answer Generation", 
    description="""
    Retrieval-Augmented Generation tool that searches through indexed documents
    to provide accurate, context-aware answers with proper citations.
    
    Capabilities:
    - Hybrid search using vector similarity and BM25 ranking
    - Conversation context integration for follow-up questions
    - LLM-powered answer synthesis from retrieved documents
    - Automatic document citation and source tracking
    - Query enhancement for improved retrieval accuracy
    
    Best used for: Factual questions, document analysis, information lookup,
    research queries, and any task requiring grounded responses from your 
    document collection.
    """,
    version="1.0.0",
    category="rag",
    parameters=[
        ToolParameter(
            name="query",
            type=ToolParameterType.STRING,
            description="The search query or question to answer using documents",
            required=True
        ),
        ToolParameter(
            name="max_documents",
            type=ToolParameterType.INTEGER,
            description="Maximum number of documents to retrieve for context",
            required=False,
            default_value=10,
            min_value=1,
            max_value=50
        ),
        ToolParameter(
            name="similarity_threshold", 
            type=ToolParameterType.FLOAT,
            description="Minimum similarity score for document inclusion",
            required=False,
            default_value=0.7,
            min_value=0.0,
            max_value=1.0
        ),
        ToolParameter(
            name="include_citations",
            type=ToolParameterType.BOOLEAN,
            description="Whether to include detailed document citations",
            required=False,
            default_value=True
        ),
        ToolParameter(
            name="conversation_context",
            type=ToolParameterType.STRING,
            description="Previous conversation context for enhanced understanding",
            required=False
        )
    ],
    tags=["retrieval", "qa", "search", "documents", "citations"],
    examples=[
        {
            "description": "Basic document search",
            "input": {
                "query": "What was the company's revenue last quarter?",
                "max_documents": 5
            },
            "expected_output": "Answer with financial data and document citations"
        },
        {
            "description": "Contextual follow-up question",
            "input": {
                "query": "How does that compare to the previous year?",
                "conversation_context": "Previous question was about Q3 2024 revenue"
            },
            "expected_output": "Comparative analysis with appropriate context"
        }
    ]
)


async def rag_tool(query: str, parameters: Dict[str, Any]) -> ToolResult:
    """
    Execute RAG tool for document retrieval and answer generation.
    
    This function serves as the standardized interface for RAG operations,
    wrapping our existing RAG infrastructure in the tool framework.
    
    Args:
        query: The user's question or search query
        parameters: Tool-specific parameters including:
            - max_documents: Maximum documents to retrieve (default: 10)
            - similarity_threshold: Minimum similarity score (default: 0.7) 
            - include_citations: Whether to include citations (default: True)
            - conversation_context: Previous conversation context (optional)
            
    Returns:
        ToolResult containing the RAG response, citations, and metadata
        
    Raises:
        Exception: If RAG execution fails due to system issues
    """
    start_time = datetime.now(timezone.utc)
    execution_id = str(uuid.uuid4())
    
    logger.info(f"RAG Tool execution started: {execution_id}")
    logger.info(f"Query: {query}")
    logger.debug(f"Parameters: {parameters}")
    
    try:
        # Extract and validate parameters
        max_documents = parameters.get("max_documents", 10)
        similarity_threshold = parameters.get("similarity_threshold", 0.7)
        include_citations = parameters.get("include_citations", True)
        conversation_context = parameters.get("conversation_context")
        
        # Get global chat engine
        chat_engine = get_global_chat_engine()
        if not chat_engine:
            raise RuntimeError("Chat engine not available - ensure system is properly initialized")
        
        # Create conversation memory for this execution
        # If we have conversation context, we could parse it, but for now start fresh
        conversation_memory = ConversationMemory()
        
        # Collect the response from our existing streaming function
        response_tokens = []
        citations_collected = []
        
        for item in stream_chat_response_with_memory(query, chat_engine, conversation_memory):
            if isinstance(item, str):
                if item == "QA_SERVICE_STREAM_ENDED_SENTINEL":
                    break
                response_tokens.append(item)
            elif isinstance(item, list):
                # This should be citations
                citations_collected = item
        
        # Combine response tokens
        full_response = "".join(response_tokens)
        
        if not full_response:
            raise RuntimeError("Empty response from RAG system")
        
        # Calculate execution time
        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Create RAG-specific output
        rag_output = RAGToolOutput(
            answer=full_response,
            retrieved_documents=[],  # Could be enhanced to include document metadata
            citations=citations_collected,
            confidence_score=0.8,  # Could be calculated based on retrieval scores
            retrieval_metadata={
                "execution_id": execution_id,
                "max_documents_requested": max_documents,
                "similarity_threshold": similarity_threshold,
                "documents_found": len(citations_collected),
                "query_enhanced": conversation_context is not None
            }
        )
        
        # Create standardized tool result
        tool_result = ToolResult(
            tool_name="rag_tool",
            status=ToolExecutionStatus.COMPLETED,
            result_data=rag_output,
            metadata={
                "execution_id": execution_id,
                "query_length": len(query),
                "response_length": len(full_response),
                "citations_count": len(citations_collected),
                "parameters_used": parameters
            },
            execution_time_ms=execution_time_ms,
            citations=citations_collected
        )
        
        logger.info(f"RAG Tool execution completed successfully: {execution_id}")
        logger.info(f"Response length: {len(full_response)} chars, Citations: {len(citations_collected)}")
        
        return tool_result
        
    except Exception as e:
        # Calculate execution time even for failures
        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        error_message = f"RAG tool execution failed: {str(e)}"
        logger.error(f"RAG Tool execution failed: {execution_id} - {error_message}", exc_info=True)
        
        # Return failed tool result
        tool_result = ToolResult(
            tool_name="rag_tool",
            status=ToolExecutionStatus.FAILED,
            result_data=None,
            metadata={
                "execution_id": execution_id,
                "error_type": type(e).__name__,
                "parameters_attempted": parameters
            },
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            citations=[]
        )
        
        return tool_result


def get_rag_tool_schema() -> ToolSchema:
    """
    Get the schema definition for the RAG tool.
    
    Returns:
        ToolSchema containing complete tool definition and parameter specifications
    """
    return RAG_TOOL_SCHEMA


# Utility functions for tool integration
def validate_rag_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize RAG tool parameters.
    
    Args:
        parameters: Raw parameters from agent or user input
        
    Returns:
        Validated and normalized parameters with defaults applied
        
    Raises:
        ValueError: If parameters are invalid
    """
    validated = {}
    
    # Validate max_documents
    max_docs = parameters.get("max_documents", 10)
    if not isinstance(max_docs, int) or max_docs < 1 or max_docs > 50:
        raise ValueError("max_documents must be an integer between 1 and 50")
    validated["max_documents"] = max_docs
    
    # Validate similarity_threshold
    sim_threshold = parameters.get("similarity_threshold", 0.7)
    if not isinstance(sim_threshold, (int, float)) or sim_threshold < 0.0 or sim_threshold > 1.0:
        raise ValueError("similarity_threshold must be a number between 0.0 and 1.0")
    validated["similarity_threshold"] = float(sim_threshold)
    
    # Validate include_citations
    include_cits = parameters.get("include_citations", True)
    if not isinstance(include_cits, bool):
        raise ValueError("include_citations must be a boolean")
    validated["include_citations"] = include_cits
    
    # Validate conversation_context (optional)
    context = parameters.get("conversation_context")
    if context is not None and not isinstance(context, str):
        raise ValueError("conversation_context must be a string if provided")
    if context:
        validated["conversation_context"] = context
    
    return validated


async def test_rag_tool() -> bool:
    """
    Test function to verify RAG tool functionality.
    
    Returns:
        True if test passes, False otherwise
    """
    try:
        test_result = await rag_tool(
            "What is this document about?",
            {"max_documents": 3, "include_citations": True}
        )
        return test_result.status == ToolExecutionStatus.COMPLETED
    except Exception as e:
        logger.error(f"RAG tool test failed: {e}")
        return False 