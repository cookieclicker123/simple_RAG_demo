"""
Meta Agent - Primary Orchestrator for Multi-Agent Framework

The meta-agent serves as the primary orchestrator in our agent framework,
responsible for:
- Tool selection and coordination
- Query analysis and routing
- Response synthesis and formatting
- Future sub-agent coordination (when implemented)

This implementation follows the higher-order function pattern as requested,
using type-safe interfaces and leveraging our existing infrastructure.

Key Design Principles:
- Stateless operation for thread safety
- Intelligent tool selection via LLM reasoning
- Integration with existing investor prompt
- Preparation for future multi-agent workflows
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core import Settings as LlamaSettings

from agents.agent_models import (
    AgentInput, 
    AgentOutput, 
    AgentResult, 
    AgentRole,
    AgentMetadata
)
from agents.agent_types import MetaAgentFunction
from tools.tool_registry import tool_registry
from src.models import DocumentCitation, QueryEnhancementAssessment
from src.config import settings as app_settings
from src.prompts.investor_prompt import INVESTOR_SYSTEM_PROMPT
from src.prompts.query_enhancement_prompt import get_query_enhancement_system_prompt, QUERY_ENHANCEMENT_USER_PROMPT
from src.utils.language_detection import language_detector

logger = logging.getLogger(__name__)


# Enhanced meta-agent system prompt that builds on the investor prompt
def get_meta_agent_system_prompt() -> str:
    """
    Get the meta-agent system prompt that combines the investor prompt with agent orchestration capabilities.
    
    Returns:
        Complete system prompt for meta-agent operation
    """
    return f"""{INVESTOR_SYSTEM_PROMPT}

---AGENT ORCHESTRATION CAPABILITIES---

In addition to your financial analysis expertise, you are now the primary orchestrator in a multi-agent system with access to specialized tools. Your role includes:

AVAILABLE TOOLS:
- rag_tool: Document search and retrieval-augmented generation
  * Best for: Factual questions, document analysis, research queries
  * Provides: Contextual answers with citations from indexed documents
  * When to use: User asks about specific information that might be in documents

AGENT DECISION FRAMEWORK:
1. Does the query require information from financial documents? → Use rag_tool
2. Is this a general conversation or greeting? → Respond directly without tools  
3. Does the query need multiple capabilities? → Plan multi-step approach
4. Is the query unclear? → Ask for clarification while maintaining financial expertise

TOOL USAGE PRINCIPLES:
- Always prioritize accuracy and cite sources when using tools
- Use tools when they enhance the quality and accuracy of your financial analysis
- Be transparent about tool usage while maintaining your expert persona
- If tools fail, acknowledge limitations and provide what you can from your expertise
- Maintain the high standards of financial analysis outlined above

Remember: You are still primarily an expert financial analyst, but now with enhanced capabilities through tool orchestration."""


async def _enhance_query_with_context(query: str, conversation_context: str) -> tuple[str, bool, str]:
    """
    Enhance query with conversation context using the existing query enhancement system.
    
    This function uses the established query enhancement prompt to determine if a query
    needs contextual enhancement and provides the enhanced version.
    
    Args:
        query: Original user query
        conversation_context: Recent conversation history
        
    Returns:
        Tuple of (enhanced_query, is_contextual, reasoning)
    """
    try:
        # Get LLM instance for enhancement
        llm = LlamaSettings.llm
        if not isinstance(llm, LlamaIndexOpenAI):
            # Fallback: return original query
            return query, False, "LLM not available for enhancement"
        
        # Use the existing query enhancement system prompt
        enhancement_system_prompt = get_query_enhancement_system_prompt()
        
        # Format the user prompt with current data
        enhancement_user_prompt = QUERY_ENHANCEMENT_USER_PROMPT.format(
            current_query=query,
            conversation_history=conversation_context or "No previous conversation context"
        )
        
        # Combine system and user prompts
        full_prompt = f"{enhancement_system_prompt}\n\n{enhancement_user_prompt}"
        
        response = await llm.acomplete(full_prompt)
        
        # Parse the JSON response
        import json
        try:
            enhancement_data = json.loads(response.text.strip())
            
            # Validate the response matches expected structure
            assessment = QueryEnhancementAssessment(**enhancement_data)
            
            return (
                assessment.enhanced_query,
                assessment.query_type == "contextual",
                assessment.reasoning
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse query enhancement response: {e}")
            return query, False, f"Enhancement parsing failed: {str(e)}"
            
    except Exception as e:
        logger.warning(f"Query enhancement failed: {e}")
        return query, False, f"Enhancement error: {str(e)}"


async def meta_agent(agent_input: AgentInput) -> AgentResult:
    """
    Meta-agent function for orchestrating tool usage and response generation.
    
    This function serves as the primary entry point for the agent framework,
    analyzing queries, selecting tools, and coordinating responses.
    
    Args:
        agent_input: Standardized agent input containing query, context, and metadata
        
    Returns:
        AgentResult containing the response, tool results, and execution metadata
        
    Architecture Notes:
        - Follows higher-order function pattern as requested
        - Stateless design for thread safety
        - Type-safe interfaces throughout
        - Extensible for future sub-agents and complex workflows
    """
    start_time = datetime.now(timezone.utc)
    execution_id = str(uuid.uuid4())
    
    logger.info(f"Meta-agent execution started: {execution_id}")
    logger.info(f"Query: {agent_input.query}")
    logger.debug(f"Context: {agent_input.context}")
    
    try:
        # Initialize response tracking
        response_parts = []
        tools_used = []
        tool_results = []
        citations_collected = []
        confidence_score = 0.8
        
        # Step 1: Enhance query with conversation context
        enhanced_query, is_contextual, reasoning = await _enhance_query_with_context(
            agent_input.query, 
            agent_input.context.conversation_memory.get_conversation_history_text(5)  # Get more context
        )
        
        logger.info(f"Enhanced query: {enhanced_query}")
        logger.info(f"Query type: {is_contextual}, Reasoning: {reasoning}")
        
        # Step 2: Language detection and translation workflow
        original_language = "en"  # Default assumption
        translated_query = enhanced_query  # Default to enhanced query
        needs_response_translation = False
        
        # Detect language of the original query (not enhanced, as we want to detect user's language)
        language_result = language_detector.detect_language(agent_input.query)
        logger.info(f"Language detection: {language_result.detected_language} (confidence: {language_result.confidence:.2f})")
        logger.info(f"Translation needed: {language_result.needs_translation}")
        
        if language_result.needs_translation and language_result.detected_language == "zh":
            logger.info("Chinese input detected - initiating translation workflow")
            original_language = "zh"
            needs_response_translation = True
            
            # Translate the enhanced query to English for processing
            try:
                translation_result = await tool_registry.execute_tool(
                    "translator_tool",
                    f"Translate query for processing: {enhanced_query}",
                    {
                        "text": enhanced_query,
                        "source_language": "zh",
                        "target_language": "en"
                    }
                )
                
                if translation_result.status.value == "completed" and translation_result.result_data:
                    translated_query = translation_result.result_data.translated_text
                    logger.info(f"Query translated successfully: '{enhanced_query}' → '{translated_query}'")
                else:
                    logger.warning(f"Query translation failed: {translation_result.error_message}")
                    # Continue with enhanced query (might be mixed language)
                    translated_query = enhanced_query
                    
            except Exception as e:
                logger.error(f"Error translating query: {e}", exc_info=True)
                # Continue with enhanced query as fallback
                translated_query = enhanced_query
        
        # Step 3: Analyze query and determine tool requirements (using translated query)
        tool_decision = await _analyze_query_for_tools(
            translated_query,  # Use translated query for tool analysis
            agent_input.context.conversation_memory.get_conversation_history_text(5),
            is_contextual
        )
        
        logger.info(f"Tool decision: {tool_decision}")
        
        # Step 4: Execute tools based on analysis
        if tool_decision.get("use_rag", False):
            logger.info("Executing RAG tool for document-based financial query")
            
            # Prepare enhanced RAG parameters based on conversation context
            conversation_summary = agent_input.context.conversation_memory.get_conversation_history_text(3)
            rag_params = {
                "query": translated_query,
                "max_documents": tool_decision.get("max_documents", 10),
                "include_citations": True,
                "conversation_context": conversation_summary
            }
            
            # If this is a follow-up query, adjust parameters
            if is_contextual:
                rag_params["max_documents"] = min(15, rag_params["max_documents"] + 3)  # Get more context for contextual queries
            
            # Execute RAG tool
            try:
                rag_result = await tool_registry.execute_tool(
                    "rag_tool", 
                    translated_query, 
                    rag_params
                )
                
                tools_used.append("rag_tool")
                tool_results.append(rag_result)
                
                if rag_result.status.value == "completed" and rag_result.result_data:
                    # Extract response from RAG result
                    rag_output = rag_result.result_data
                    response_parts.append(rag_output.answer)
                    citations_collected.extend(rag_result.citations)
                    confidence_score = rag_output.confidence_score
                    
                    logger.info(f"RAG tool completed successfully with {len(rag_result.citations)} citations")
                else:
                    # RAG failed, provide fallback response
                    error_msg = rag_result.error_message or "RAG tool execution failed"
                    response_parts.append(
                        f"I encountered an issue accessing the document database: {error_msg}. "
                        "I'm unable to provide information from your documents at this time."
                    )
                    confidence_score = 0.3
                    
                    logger.warning(f"RAG tool failed: {error_msg}")
                    
            except Exception as e:
                logger.error(f"RAG tool execution error: {e}", exc_info=True)
                response_parts.append(
                    "I encountered an error while searching through your documents. "
                    "Please ensure the system is properly configured and try again."
                )
                confidence_score = 0.2
        
        else:
            # Handle non-RAG queries with direct LLM response using financial expertise
            logger.info("Handling query without tools - direct financial expertise response")
            
            # Get more conversation context for better continuity
            conversation_summary = agent_input.context.conversation_memory.get_conversation_history_text(5)
            
            llm_response = await _generate_direct_response(
                translated_query,
                conversation_summary
            )
            
            response_parts.append(llm_response)
            confidence_score = 0.7  # Lower confidence for non-grounded responses
        
        # Step 5: Synthesize final response
        final_response = "\n\n".join(response_parts).strip()
        
        if not final_response:
            final_response = "I apologize, but I wasn't able to generate a response to your query. Please try rephrasing your question or check if the system is properly configured."
            confidence_score = 0.1
        
        # Step 5.5: Translate response back to original language if needed
        if needs_response_translation and original_language == "zh":
            logger.info("Translating response back to Chinese")
            try:
                response_translation_result = await tool_registry.execute_tool(
                    "translator_tool",
                    f"Translate response back to Chinese: {final_response[:100]}...",
                    {
                        "text": final_response,
                        "source_language": "en", 
                        "target_language": "zh"
                    }
                )
                
                if response_translation_result.status.value == "completed" and response_translation_result.result_data:
                    final_response = response_translation_result.result_data.translated_text
                    logger.info("Response translated back to Chinese successfully")
                    
                    # Add translation info to tools used for transparency
                    if "translator_tool" not in tools_used:
                        tools_used.append("translator_tool")
                else:
                    logger.warning(f"Response translation failed: {response_translation_result.error_message}")
                    # Keep English response as fallback
                    
            except Exception as e:
                logger.error(f"Error translating response back to Chinese: {e}", exc_info=True)
                # Keep English response as fallback
        
        # Step 6: Update conversation memory
        if hasattr(agent_input.context.conversation_memory, 'add_turn'):
            query_type = "contextual" if tools_used else "standalone"
            agent_input.context.conversation_memory.add_turn(
                user_query=agent_input.query,
                ai_response=final_response,
                query_type=query_type
            )
            memory_updated = True
        else:
            memory_updated = False
        
        # Step 7: Create agent output
        agent_output = AgentOutput(
            response_text=final_response,
            tool_calls_made=tools_used,
            tool_results=tool_results,
            citations=citations_collected,
            sub_agent_calls=[],  # Will be used for future sub-agents
            confidence_score=confidence_score,
            execution_metadata={
                "execution_id": execution_id,
                "tools_considered": tool_decision,
                "query_length": len(agent_input.query),
                "response_length": len(final_response),
                "memory_turns": len(agent_input.context.conversation_memory.turns),
                "query_enhancement": {
                    "original_query": agent_input.query,
                    "enhanced_query": enhanced_query,
                    "is_contextual": is_contextual,
                    "enhancement_reasoning": reasoning
                },
                "language_workflow": {
                    "original_language": original_language,
                    "detected_language": language_result.detected_language,
                    "detection_confidence": language_result.confidence,
                    "translation_needed": needs_response_translation,
                    "translated_query": translated_query if needs_response_translation else None,
                    "detection_reasoning": language_result.reasoning
                }
            },
            next_suggested_actions=_generate_suggested_actions(translated_query, tools_used)
        )
        
        # Step 7: Calculate execution metrics
        end_time = datetime.now(timezone.utc)
        execution_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Step 8: Create agent result
        agent_result = AgentResult(
            input_metadata=agent_input.metadata,
            output=agent_output,
            execution_duration_ms=execution_duration_ms,
            success=True,
            error_details=None,
            memory_updated=memory_updated
        )
        
        logger.info(f"Meta-agent execution completed successfully: {execution_id}")
        logger.info(f"Tools used: {tools_used}, Citations: {len(citations_collected)}")
        
        return agent_result
        
    except Exception as e:
        # Handle execution failures
        end_time = datetime.now(timezone.utc)
        execution_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        error_message = f"Meta-agent execution failed: {str(e)}"
        logger.error(f"Meta-agent execution failed: {execution_id} - {error_message}", exc_info=True)
        
        # Create error response
        error_output = AgentOutput(
            response_text="I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            tool_calls_made=[],
            tool_results=[],
            citations=[],
            sub_agent_calls=[],
            confidence_score=0.0,
            execution_metadata={
                "execution_id": execution_id,
                "error_type": type(e).__name__,
                "error_location": "meta_agent"
            },
            next_suggested_actions=[]
        )
        
        agent_result = AgentResult(
            input_metadata=agent_input.metadata,
            output=error_output,
            execution_duration_ms=execution_duration_ms,
            success=False,
            error_details=error_message,
            memory_updated=False
        )
        
        return agent_result


async def _analyze_query_for_tools(query: str, conversation_context: str, is_contextual: bool = False) -> Dict[str, Any]:
    """
    Analyze enhanced query to determine which tools should be used.
    
    This function uses LLM reasoning with financial expertise to decide on tool usage strategy.
    The query has already been enhanced for context, so this focuses on tool selection.
    
    Args:
        query: Enhanced user query (already processed for context)
        conversation_context: Recent conversation history
        is_contextual: Whether the original query was contextual (from enhancement step)
        
    Returns:
        Dictionary with tool usage decisions and parameters
    """
    try:
        # Get LLM instance for analysis
        llm = LlamaSettings.llm
        if not isinstance(llm, LlamaIndexOpenAI):
            # Fallback to default tool usage for financial queries
            return {"use_rag": True, "reasoning": "Default RAG usage for financial analysis"}
        
        # Use the investor-focused analysis prompt
        analysis_prompt = f"""
        As an expert financial analyst, analyze the following enhanced user query and determine if it requires document search (RAG tool) or can be answered directly with general financial knowledge.

        Enhanced Query: "{query}"
        Query Type: {"Contextual (enhanced from conversation)" if is_contextual else "Standalone"}
        
        Recent Conversation Context: 
        {conversation_context or "No previous context"}

        Consider these factors:
        1. Does the query ask for specific financial data, metrics, or numbers that would be in company documents?
        2. Is it asking about specific company performance, earnings, financial statements, or business metrics?
        3. Does it reference specific time periods, quarters, or financial reporting periods?
        4. Is it a general financial concept question that doesn't require specific company data?
        5. Would accessing company documents significantly improve the accuracy and specificity of the answer?

        Guidelines:
        - Use RAG for: Specific financial data, company performance metrics, earnings information, specific document references
        - Use direct response for: General financial concepts, definitions, market analysis without specific company data
        - For contextual queries: More likely to need RAG since they often refer to previously discussed specific data

        Respond with ONLY a JSON object:
        {{
            "use_rag": true/false,
            "reasoning": "Brief explanation focusing on financial analysis needs",
            "max_documents": 5-15 (if using RAG),
            "confidence": 0.0-1.0
        }}
        """
        
        response = await llm.acomplete(analysis_prompt)
        
        # Parse LLM response
        import json
        try:
            decision = json.loads(response.text.strip())
            return decision
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM tool analysis, defaulting to RAG for financial query")
            return {"use_rag": True, "reasoning": "JSON parse failed, defaulting to RAG for financial analysis"}
            
    except Exception as e:
        logger.warning(f"Tool analysis failed: {e}, defaulting to RAG for financial analysis")
        return {"use_rag": True, "reasoning": f"Analysis error: {str(e)}, using RAG for financial safety"}


async def _generate_direct_response(query: str, conversation_context: str) -> str:
    """
    Generate direct LLM response for queries that don't require tools.
    
    Uses the investor prompt to maintain financial expertise even without document access.
    
    Args:
        query: User's query
        conversation_context: Recent conversation history
        
    Returns:
        Direct LLM response with financial expertise
    """
    try:
        llm = LlamaSettings.llm
        
        # Use the investor prompt as the system context for direct responses
        direct_prompt = f"""{get_meta_agent_system_prompt()}

Recent Conversation Context:
{conversation_context or "No previous context"}

User Query: {query}

Since this query doesn't require searching through specific company documents, provide a helpful response based on your financial expertise. Be clear about what information would require accessing specific company documents versus what you can address with general financial knowledge.

Response:"""
        
        response = await llm.acomplete(direct_prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Direct response generation failed: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again, and if you're asking about specific financial data, I may need to search through your documents to provide accurate information."


def _generate_suggested_actions(query: str, tools_used: List[str]) -> List[str]:
    """
    Generate suggested follow-up actions based on the query and tools used.
    
    Focuses on financial analysis and investor-relevant suggestions.
    
    Args:
        query: Original user query
        tools_used: List of tools that were used
        
    Returns:
        List of suggested actions focused on financial analysis
    """
    suggestions = []
    
    if "rag_tool" in tools_used:
        suggestions.extend([
            "Ask for trends analysis or year-over-year comparisons",
            "Request breakdowns by business segments or geographic regions", 
            "Inquire about management commentary or explanations for these figures",
            "Explore related financial metrics or KPIs",
            "Ask about the methodology or assumptions behind these numbers"
        ])
    else:
        suggestions.extend([
            "Request specific financial data from your documents",
            "Ask for comparative analysis with industry benchmarks",
            "Inquire about historical trends for this metric",
            "Explore related financial concepts or ratios"
        ])
    
    return suggestions[:3]  # Limit to 3 suggestions


# Factory function for creating agent instances (future extensibility)
def create_meta_agent() -> MetaAgentFunction:
    """
    Factory function to create a meta-agent instance.
    
    Returns:
        Meta-agent function conforming to MetaAgentFunction type
    """
    return meta_agent


# Utility functions for meta-agent operations
def get_meta_agent_info() -> Dict[str, Any]:
    """
    Get information about the meta-agent capabilities.
    
    Returns:
        Dictionary with meta-agent information
    """
    return {
        "agent_type": "meta_agent",
        "role": AgentRole.META.value,
        "capabilities": [
            "Query analysis and routing",
            "Tool selection and coordination", 
            "Response synthesis",
            "Conversation memory integration",
            "Future sub-agent orchestration"
        ],
        "available_tools": tool_registry.list_tools(),
        "version": "1.0.0"
    } 