"""Query enhancement utilities for conversational RAG."""

import logging
import json
from typing import Optional
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.llms import ChatMessage, MessageRole

from src.models import (
    QueryEnhancementRequest, 
    QueryEnhancementResult, 
    QueryEnhancementAssessment
)
from src.config import settings
from src.prompts.query_enhancement_prompt import (
    QUERY_ENHANCEMENT_SYSTEM_PROMPT,
    QUERY_ENHANCEMENT_USER_PROMPT
)

logger = logging.getLogger(__name__)

class QueryEnhancer:
    """
    LLM-powered query enhancement for conversational RAG systems.
    
    This utility provides intelligent query enhancement capabilities that determine
    whether user queries need contextual enrichment based on conversation history.
    It leverages language models to understand implicit references, pronouns, and
    contextual dependencies that would otherwise result in poor retrieval performance.
    
    The enhancer operates on every query (no pre-filtering) and uses structured
    JSON responses with Pydantic validation to ensure consistent, type-safe results.
    Enhanced queries are optimized for semantic search and embedding models by
    incorporating relevant context to make them self-contained and unambiguous.
    
    Key Features:
        - LLM-based intent recognition (standalone vs contextual)
        - Automatic context integration from conversation history
        - Schema-grounded prompts with Pydantic model validation
        - Structured reasoning and confidence scoring
        - Robust error handling with fallback mechanisms
        
    Usage:
        enhancer = QueryEnhancer()
        request = QueryEnhancementRequest(
            current_query="What about the performance?",
            conversation_history="User: Tell me about Apple.\nAssistant: Apple had strong Q3 results..."
        )
        result = enhancer.enhance_query_with_context(request)
        # result.enhanced_query: "What about Apple's financial performance?"
        
    Architecture:
        - Uses OpenAI LLM with configurable model and temperature
        - Employs schema-grounded prompts for consistent JSON responses
        - Validates responses against QueryEnhancementAssessment Pydantic model
        - Provides detailed reasoning and confidence metrics for observability
    """
    
    def __init__(self):
        # Create a separate LLM instance for query enhancement to avoid interfering with main streaming LLM
        self.llm = LlamaIndexOpenAI(
            model=settings.query_enhancement_model,
            temperature=settings.query_enhancement_temperature,
            # Ensure this doesn't interfere with streaming by using different instance
            max_retries=1,
            request_timeout=10.0  # Quick timeout for query enhancement
        )
    
    def enhance_query_with_context(self, request: QueryEnhancementRequest) -> QueryEnhancementResult:
        """
        Use LLM to assess and enhance queries with conversation context.
        
        This is the primary method for query enhancement that analyzes whether a user's
        query needs contextual enrichment and produces an enhanced version optimized
        for document retrieval. The method always runs (no pre-filtering) to ensure
        consistent processing and to catch subtle contextual dependencies.
        
        The enhancement process:
        1. Formats conversation history and current query for LLM analysis
        2. Uses schema-grounded prompts with Pydantic model specifications
        3. Requests structured JSON response with reasoning and confidence
        4. Validates response against QueryEnhancementAssessment model
        5. Returns structured result with original/enhanced queries and metadata
        
        Args:
            request (QueryEnhancementRequest): Contains the current query and 
                conversation history context for enhancement analysis.
                
        Returns:
            QueryEnhancementResult: Structured result containing:
                - original_query: The unmodified user query
                - enhanced_query: Context-enriched query or original if standalone
                - is_contextual: Whether the query was identified as needing context
                - reasoning: LLM's explanation of its enhancement decision
                - confidence: LLM's confidence score (0.0-1.0) in the assessment
                
        Raises:
            No exceptions raised - errors are handled gracefully with fallback
            to original query and confidence=0.0 to indicate enhancement failure.
            
        Example:
            >>> request = QueryEnhancementRequest(
            ...     current_query="How does it compare?",
            ...     conversation_history="User: Tesla's Q3 revenue?\nAssistant: $23.4B..."
            ... )
            >>> result = enhancer.enhance_query_with_context(request)
            >>> result.enhanced_query
            "How does Tesla's Q3 revenue of $23.4B compare to previous quarters?"
            >>> result.is_contextual
            True
        """
        logger.debug(f"Processing query for enhancement: '{request.current_query}'")
        
        # Prepare conversation history (handle empty case)
        conversation_context = request.conversation_history.strip() if request.conversation_history else "No previous conversation."
        
        # Build the prompt
        user_prompt = QUERY_ENHANCEMENT_USER_PROMPT.format(
            current_query=request.current_query,
            conversation_history=conversation_context
        )
        
        try:
            # Create messages for LLM
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=QUERY_ENHANCEMENT_SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=user_prompt)
            ]
            
            # Request structured JSON response
            response = self.llm.chat(
                messages,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            response_content = response.message.content.strip()
            logger.debug(f"LLM response for query enhancement: {response_content}")
            
            try:
                response_data = json.loads(response_content)
                assessment = QueryEnhancementAssessment(**response_data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response as structured data: {e}")
                logger.error(f"Response content: {response_content}")
                # Fallback to original query
                return self._create_fallback_result(request.current_query)
            
            # Create result from assessment
            result = QueryEnhancementResult(
                original_query=request.current_query,
                enhanced_query=assessment.enhanced_query,
                is_contextual=(assessment.query_type == "contextual"),
                reasoning=assessment.reasoning,
                confidence=assessment.confidence
            )
            
            # Log the enhancement decision
            if assessment.needs_enhancement:
                logger.info(f"Enhanced query: '{request.current_query}' → '{assessment.enhanced_query}'")
                logger.info(f"Reasoning: {assessment.reasoning}")
            else:
                logger.debug(f"Query assessed as standalone, no enhancement needed: '{request.current_query}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM-based query enhancement: {e}", exc_info=True)
            return self._create_fallback_result(request.current_query)
    
    def _create_fallback_result(self, original_query: str) -> QueryEnhancementResult:
        """Create fallback result when LLM enhancement fails."""
        logger.warning(f"Using fallback for query: '{original_query}'")
        return QueryEnhancementResult(
            original_query=original_query,
            enhanced_query=original_query,
            is_contextual=False,
            reasoning="Fallback: LLM enhancement failed, using original query",
            confidence=0.0
        )

# Global instance following our utility pattern
query_enhancer = QueryEnhancer() 