"""Language detection utilities for multi-language agent support."""

import logging
import json
from typing import Optional
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.llms import ChatMessage, MessageRole

from src.models import LanguageDetectionResult, LanguageDetectionAssessment
from src.config import settings
from src.prompts.language_recognition_prompt import (
    get_language_recognition_system_prompt,
    LANGUAGE_RECOGNITION_USER_PROMPT
)

logger = logging.getLogger(__name__)

class LanguageDetector:
    """
    LLM-powered language detection for multi-language agent workflows.
    
    This utility determines whether user input is in English (primary processing
    language) or Chinese (requires translation workflow). It uses language models
    to understand context, mixed-language inputs, and edge cases that simple
    character-based detection might miss.
    
    The detector provides structured responses with confidence scoring and
    reasoning, enabling intelligent routing in the agent framework.
    
    Key Features:
        - LLM-based language understanding beyond character detection
        - Handles mixed-language inputs and technical terms
        - Schema-grounded prompts with Pydantic model validation
        - Confidence scoring for decision quality assessment
        - Robust error handling with fallback mechanisms
        
    Usage:
        detector = LanguageDetector()
        result = detector.detect_language("你好，请问公司的revenue如何？")
        # result.detected_language: "zh"
        # result.needs_translation: True
        
    Architecture:
        - Uses OpenAI LLM with configurable model and temperature
        - Employs schema-grounded prompts for consistent JSON responses
        - Validates responses against LanguageDetectionAssessment Pydantic model
        - Provides detailed reasoning and confidence metrics for observability
    """
    
    def __init__(self):
        # Create a separate LLM instance for language detection
        self.llm = LlamaIndexOpenAI(
            model=settings.language_detection_model,
            temperature=settings.language_detection_temperature,
            max_retries=1,
            request_timeout=5.0  # Quick timeout for language detection
        )
    
    def detect_language(self, user_input: str) -> LanguageDetectionResult:
        """
        Detect the primary language of user input for translation workflow routing.
        
        This method analyzes user input to determine if it's English (no translation
        needed) or Chinese (requires translation workflow). It handles mixed-language
        inputs, technical terms, and provides confidence scoring.
        
        Args:
            user_input (str): The user's query text to analyze for language detection.
                
        Returns:
            LanguageDetectionResult: Structured result containing:
                - detected_language: "en" or "zh"
                - confidence: LLM confidence score (0.0-1.0)
                - needs_translation: Whether translation workflow is needed
                - reasoning: LLM's explanation of its detection decision
                - translation_direction: "zh_to_en" if Chinese detected, None if English
                
        Example:
            >>> detector = LanguageDetector()
            >>> result = detector.detect_language("公司的财务报告如何？")
            >>> result.detected_language
            "zh"
            >>> result.needs_translation
            True
            >>> result.translation_direction
            "zh_to_en"
        """
        logger.debug(f"Detecting language for input: '{user_input[:50]}...'")
        
        # Quick fallback for obviously English input (performance optimization)
        if self._is_likely_english(user_input):
            logger.debug("Quick detection: likely English input")
            return LanguageDetectionResult(
                detected_language="en",
                confidence=0.9,
                needs_translation=False,
                reasoning="Quick detection: input appears to be English based on character patterns",
                translation_direction=None
            )
        
        # Build the prompt using existing pattern
        system_prompt = get_language_recognition_system_prompt()
        user_prompt = LANGUAGE_RECOGNITION_USER_PROMPT.format(user_input=user_input)
        
        try:
            # Create messages for LLM
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt)
            ]
            
            # Request structured JSON response
            response = self.llm.chat(
                messages,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            response_content = response.message.content.strip()
            logger.debug(f"LLM response for language detection: {response_content}")
            
            try:
                response_data = json.loads(response_content)
                assessment = LanguageDetectionAssessment(**response_data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response as structured data: {e}")
                logger.error(f"Response content: {response_content}")
                # Fallback to English assumption
                return self._create_fallback_result(user_input, "en")
            
            # Create result from assessment
            result = LanguageDetectionResult(
                detected_language=assessment.detected_language,
                confidence=assessment.confidence,
                needs_translation=assessment.needs_translation,
                reasoning=assessment.reasoning,
                translation_direction=assessment.translation_direction
            )
            
            # Log the detection decision
            logger.info(f"Language detected: {assessment.detected_language} (confidence: {assessment.confidence:.2f})")
            logger.info(f"Translation needed: {assessment.needs_translation}")
            logger.debug(f"Reasoning: {assessment.reasoning}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM-based language detection: {e}", exc_info=True)
            # Fallback based on character analysis
            fallback_lang = "zh" if self._contains_chinese_chars(user_input) else "en"
            return self._create_fallback_result(user_input, fallback_lang)
    
    def _is_likely_english(self, text: str) -> bool:
        """Quick heuristic to identify likely English text for performance optimization."""
        if not text:
            return True
        
        # Count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len([c for c in text if c.isalpha()])
        
        # If less than 10% Chinese characters and some English letters, likely English
        if total_chars > 0:
            chinese_ratio = chinese_chars / total_chars
            return chinese_ratio < 0.1 and any(c.isascii() and c.isalpha() for c in text)
        
        return not self._contains_chinese_chars(text)
    
    def _contains_chinese_chars(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    def _create_fallback_result(self, user_input: str, detected_lang: str) -> LanguageDetectionResult:
        """Create fallback result when LLM detection fails."""
        needs_translation = detected_lang == "zh"
        translation_direction = "zh_to_en" if needs_translation else None
        
        logger.warning(f"Using fallback language detection: {detected_lang} for input: '{user_input[:50]}...'")
        
        return LanguageDetectionResult(
            detected_language=detected_lang,
            confidence=0.7,  # Medium confidence for fallback
            needs_translation=needs_translation,
            reasoning=f"Fallback: LLM detection failed, using character-based analysis -> {detected_lang}",
            translation_direction=translation_direction
        )

# Global instance following our utility pattern
language_detector = LanguageDetector() 