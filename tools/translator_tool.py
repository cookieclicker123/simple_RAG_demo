"""
Translator Tool for Multi-Language Agent Support

This module provides translation capabilities as a standardized tool that can be
used by agents in the framework. It integrates Google Translate API with the
existing tool architecture.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from tools.tool_models import (
    ToolSchema, 
    ToolParameter, 
    ToolParameterType,
    TranslatorToolInput,
    TranslatorToolOutput
)
from agents.agent_models import ToolResult, ToolExecutionStatus
from src.utils.google_translate import google_translator
from src.models import TranslationRequest

logger = logging.getLogger(__name__)

def get_translator_tool_schema() -> ToolSchema:
    """
    Get the schema definition for the translator tool.
    
    Returns:
        ToolSchema with parameter definitions and documentation
    """
    return ToolSchema(
        tool_name="translator_tool",
        display_name="Language Translator",
        description="Translates text between English and Chinese using Google Translate API. Supports bidirectional translation with high accuracy for financial and technical content.",
        version="1.0.0",
        category="translation",
        parameters=[
            ToolParameter(
                name="text",
                type=ToolParameterType.STRING,
                description="Text to translate (required)",
                required=True,
                min_length=1,
                max_length=5000
            ),
            ToolParameter(
                name="source_language", 
                type=ToolParameterType.STRING,
                description="Source language code: 'en' for English, 'zh' for Chinese",
                required=True,
                allowed_values=["en", "zh"]
            ),
            ToolParameter(
                name="target_language",
                type=ToolParameterType.STRING, 
                description="Target language code: 'en' for English, 'zh' for Chinese",
                required=True,
                allowed_values=["en", "zh"]
            ),
            ToolParameter(
                name="context",
                type=ToolParameterType.STRING,
                description="Optional context to improve translation accuracy",
                required=False,
                max_length=1000
            )
        ],
        output_schema={
            "type": "object",
            "properties": {
                "translated_text": {"type": "string", "description": "The translated text"},
                "source_language_detected": {"type": "string", "description": "Confirmed source language"},
                "confidence_score": {"type": "number", "description": "Translation confidence (0.0-1.0)"},
                "alternative_translations": {"type": "array", "description": "Alternative translation options"}
            },
            "required": ["translated_text", "source_language_detected", "confidence_score"]
        },
        examples=[
            {
                "description": "Translate Chinese to English",
                "input": {
                    "text": "公司的收入是多少？",
                    "source_language": "zh",
                    "target_language": "en"
                },
                "output": {
                    "translated_text": "What is the company's revenue?",
                    "source_language_detected": "zh",
                    "confidence_score": 0.95
                }
            },
            {
                "description": "Translate English to Chinese", 
                "input": {
                    "text": "The quarterly financial results show strong growth.",
                    "source_language": "en",
                    "target_language": "zh"
                },
                "output": {
                    "translated_text": "季度财务结果显示强劲增长。",
                    "source_language_detected": "en", 
                    "confidence_score": 0.92
                }
            }
        ],
        tags=["translation", "chinese", "english", "multilingual", "communication"]
    )

async def translator_tool(query: str, parameters: Dict[str, Any]) -> ToolResult:
    """
    Execute translation using Google Translate API.
    
    This function follows the standard tool interface pattern established by the
    existing RAG tool, providing consistent integration with the agent framework.
    
    Args:
        query: The translation query/description (used for logging)
        parameters: Tool parameters containing text and language codes
        
    Returns:
        ToolResult containing translation output and metadata
        
    Tool Parameters:
        - text: Text to translate (required)
        - source_language: Source language code (required) 
        - target_language: Target language code (required)
        - context: Optional context for better translation (optional)
    """
    start_time = datetime.now(timezone.utc)
    execution_id = str(uuid.uuid4())
    
    logger.info(f"Translator tool execution started: {execution_id}")
    logger.info(f"Query: {query}")
    logger.debug(f"Parameters: {parameters}")
    
    try:
        # Validate and extract parameters
        text = parameters.get("text")
        source_language = parameters.get("source_language")
        target_language = parameters.get("target_language")
        context = parameters.get("context")
        
        if not text:
            raise ValueError("'text' parameter is required")
        if not source_language:
            raise ValueError("'source_language' parameter is required")
        if not target_language:
            raise ValueError("'target_language' parameter is required")
        
        # Validate language codes
        if source_language not in ["en", "zh"]:
            raise ValueError("'source_language' must be 'en' or 'zh'")
        if target_language not in ["en", "zh"]:
            raise ValueError("'target_language' must be 'en' or 'zh'")
        
        logger.info(f"Translating from {source_language} to {target_language}")
        logger.debug(f"Text to translate: '{text[:100]}...' (length: {len(text)})")
        
        # Check if Google Translate is configured
        if not google_translator.is_configured():
            error_msg = "Google Translate API key not configured"
            logger.error(error_msg)
            
            return ToolResult(
                tool_name="translator_tool",
                status=ToolExecutionStatus.FAILED,
                result_data=None,
                error_message=error_msg,
                execution_time_ms=0.0,
                metadata={
                    "execution_id": execution_id,
                    "source_language": source_language,
                    "target_language": target_language,
                    "error_type": "configuration_error"
                }
            )
        
        # Perform translation
        translated_text = await google_translator.translate(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context
        )
        
        end_time = datetime.now(timezone.utc)
        execution_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        if translated_text is None:
            error_msg = "Translation failed - Google Translate API returned no result"
            logger.error(error_msg)
            
            return ToolResult(
                tool_name="translator_tool",
                status=ToolExecutionStatus.FAILED,
                result_data=None,
                error_message=error_msg,
                execution_time_ms=execution_duration_ms,
                metadata={
                    "execution_id": execution_id,
                    "source_language": source_language,
                    "target_language": target_language,
                    "text_length": len(text),
                    "error_type": "translation_failed"
                }
            )
        
        # Create structured output
        translator_output = TranslatorToolOutput(
            translated_text=translated_text,
            source_language_detected=source_language,
            confidence_score=0.9,  # Google Translate doesn't provide confidence, using high default
            alternative_translations=[]  # Google Translate API v2 doesn't provide alternatives
        )
        
        logger.info(f"Translation completed successfully in {execution_duration_ms:.1f}ms")
        logger.debug(f"Translated text: '{translated_text[:100]}...'")
        
        return ToolResult(
            tool_name="translator_tool",
            status=ToolExecutionStatus.COMPLETED,
            result_data=translator_output,
            error_message=None,
            execution_time_ms=execution_duration_ms,
            metadata={
                "execution_id": execution_id,
                "source_language": source_language,
                "target_language": target_language,
                "original_text_length": len(text),
                "translated_text_length": len(translated_text),
                "translation_ratio": len(translated_text) / len(text) if text else 0,
                "google_translate_used": True
            },
            citations=[]  # No citations for translation
        )
        
    except ValueError as e:
        error_msg = f"Parameter validation error: {str(e)}"
        logger.error(error_msg)
        
        return ToolResult(
            tool_name="translator_tool", 
            status=ToolExecutionStatus.FAILED,
            result_data=None,
            error_message=error_msg,
            execution_time_ms=0.0,
            metadata={
                "execution_id": execution_id,
                "error_type": "validation_error",
                "parameters_received": parameters
            }
        )
        
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        execution_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        error_msg = f"Unexpected error in translator tool: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return ToolResult(
            tool_name="translator_tool",
            status=ToolExecutionStatus.FAILED,
            result_data=None,
            error_message=error_msg,
            execution_time_ms=execution_duration_ms,
            metadata={
                "execution_id": execution_id,
                "error_type": "unexpected_error",
                "exception_type": type(e).__name__
            }
        ) 