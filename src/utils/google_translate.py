"""Google Translate integration for multi-language agent support."""

import logging
import asyncio
from typing import Optional, Dict, Any
import httpx
import json

from src.models import TranslationRequest
from src.config import settings

logger = logging.getLogger(__name__)

class GoogleTranslator:
    """
    Google Translate API integration for text translation.
    
    This utility provides robust translation capabilities with error handling,
    retry logic, and proper API key management. It follows the same patterns
    as other utilities in the system for consistency.
    
    Key Features:
        - Google Translate API v2 integration
        - Async operations for non-blocking translation
        - Retry logic with exponential backoff
        - Comprehensive error handling and logging
        - Rate limiting awareness
        - Cost optimization through caching potential
        
    Usage:
        translator = GoogleTranslator()
        result = await translator.translate("你好世界", "zh", "en")
        # result: "Hello world"
        
    Architecture:
        - Uses httpx for async HTTP requests
        - Configurable timeout and retry settings
        - Proper error classification and handling
        - Integration with existing configuration system
    """
    
    def __init__(self):
        self.api_key = settings.google_translate_api_key
        self.base_url = "https://translation.googleapis.com/language/translate/v2"
        self.timeout = settings.translation_timeout_seconds
        self.retry_attempts = settings.translation_retry_attempts
        
        # Validate API key configuration
        if self.api_key == "your_google_translate_api_key_here_if_not_in_env":
            logger.warning("Google Translate API key not configured - translation will fail")
    
    async def translate(
        self, 
        text: str, 
        source_language: str, 
        target_language: str,
        context: Optional[str] = None
    ) -> Optional[str]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_language: Source language code (e.g., 'zh', 'en')
            target_language: Target language code (e.g., 'zh', 'en')
            context: Optional context for better translation (not used by Google Translate API v2)
            
        Returns:
            Translated text if successful, None if translation failed
        """
        if not text.strip():
            logger.warning("Empty text provided for translation")
            return text
        
        if source_language == target_language:
            logger.debug("Source and target languages are the same, returning original text")
            return text
        
        logger.info(f"Translating text from {source_language} to {target_language}")
        logger.debug(f"Text to translate: '{text[:100]}...' (length: {len(text)})")
        
        # Prepare request parameters
        params = {
            "key": self.api_key,
            "q": text,
            "source": source_language,
            "target": target_language,
            "format": "text"
        }
        
        # Attempt translation with retry logic
        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.base_url, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        translated_text = data["data"]["translations"][0]["translatedText"]
                        
                        logger.info(f"Translation successful on attempt {attempt + 1}")
                        logger.debug(f"Translated text: '{translated_text[:100]}...'")
                        
                        return translated_text
                    
                    elif response.status_code == 400:
                        logger.error(f"Bad request to Google Translate API: {response.text}")
                        return None  # Don't retry for bad requests
                    
                    elif response.status_code == 403:
                        logger.error("Google Translate API key invalid or quota exceeded")
                        return None  # Don't retry for auth issues
                    
                    else:
                        logger.warning(f"Translation attempt {attempt + 1} failed with status {response.status_code}: {response.text}")
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
            except httpx.TimeoutException:
                logger.warning(f"Translation attempt {attempt + 1} timed out")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except httpx.RequestError as e:
                logger.warning(f"Translation attempt {attempt + 1} failed with request error: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error during translation attempt {attempt + 1}: {e}", exc_info=True)
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
        
        logger.error(f"All {self.retry_attempts} translation attempts failed")
        return None
    
    async def translate_request(self, request: TranslationRequest) -> Optional[str]:
        """
        Translate text using a TranslationRequest object.
        
        Args:
            request: TranslationRequest containing text and language codes
            
        Returns:
            Translated text if successful, None if translation failed
        """
        return await self.translate(
            request.text,
            request.source_language,
            request.target_language,
            request.context
        )
    
    def is_configured(self) -> bool:
        """
        Check if Google Translate API is properly configured.
        
        Returns:
            True if API key is configured, False otherwise
        """
        return (self.api_key and 
                self.api_key != "your_google_translate_api_key_here_if_not_in_env")
    
    async def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of given text using Google Translate API.
        
        Note: This is mainly for fallback - we primarily use LLM-based detection.
        
        Args:
            text: Text to analyze for language detection
            
        Returns:
            Language code if detected, None if detection failed
        """
        if not text.strip():
            return None
        
        detect_url = "https://translation.googleapis.com/language/translate/v2/detect"
        params = {
            "key": self.api_key,
            "q": text
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(detect_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    detected_language = data["data"]["detections"][0][0]["language"]
                    confidence = data["data"]["detections"][0][0]["confidence"]
                    
                    logger.debug(f"Google detected language: {detected_language} (confidence: {confidence})")
                    return detected_language
                else:
                    logger.warning(f"Language detection failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in Google language detection: {e}")
            return None

# Global instance following our utility pattern
google_translator = GoogleTranslator() 