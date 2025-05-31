"""LLM configuration and interaction utilities."""

import logging
from typing import Optional
from pathlib import Path

from llama_index.core import Settings as LlamaSettings
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.llms import ChatMessage, MessageRole

from src.config import settings
from src.prompts.investor_prompt import INVESTOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class LLMManager:
    """Centralized LLM configuration and interaction."""
    
    @staticmethod
    def configure_llama_settings() -> bool:
        """
        Configure or re-configure LlamaSettings.llm based on current settings.
        Returns True if configuration was successful.
        """
        current_llm_model = settings.llm_model_name
        llm_needs_configuration = True
        
        if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model'):
            if LlamaSettings.llm.model == current_llm_model:
                llm_needs_configuration = False
            else:
                logger.info(f"LlamaSettings.llm is {LlamaSettings.llm.model}, but config wants {current_llm_model}. Re-configuring.")
        elif not LlamaSettings.llm:
            logger.info("LlamaSettings.llm not yet configured.")
        else:
            logger.info(f"LlamaSettings.llm is an unexpected type ({type(LlamaSettings.llm)}), re-configuring.")

        if llm_needs_configuration:
            logger.info(f"Configuring LlamaSettings.llm with OpenAI model: {current_llm_model}")
            try:
                LlamaSettings.llm = LlamaIndexOpenAI(
                    model=current_llm_model,
                    temperature=settings.temperature,
                    api_key=settings.openai_api_key,
                    max_tokens=settings.max_tokens,
                    system_prompt=INVESTOR_SYSTEM_PROMPT
                )
                return True
            except Exception as e:
                logger.error(f"Failed to configure LlamaSettings.llm with {current_llm_model}: {e}", exc_info=True)
                if not LlamaSettings.llm:
                    logger.warning("Falling back to a default gpt-3.5-turbo due to configuration error.")
                    try:
                        LlamaSettings.llm = LlamaIndexOpenAI(
                            model="gpt-3.5-turbo", 
                            temperature=settings.temperature, 
                            api_key=settings.openai_api_key,
                            system_prompt=INVESTOR_SYSTEM_PROMPT
                        )
                        return True
                    except Exception as e2:
                        logger.error(f"Even fallback LLM configuration failed: {e2}", exc_info=True)
                        return False
        return True

class DocumentTitleGenerator:
    """Utilities for generating document titles from content."""
    
    @staticmethod
    def generate_title_from_chunk(
        text_chunk: str, 
        llm: LlamaIndexOpenAI, 
        filename_stem: str
    ) -> Optional[str]:
        """Generate a document title from a text chunk using LLM."""
        if not text_chunk:
            return filename_stem
        
        if not llm or not hasattr(llm, 'chat'):
            logger.warning("LLM for title extraction is not properly configured. Falling back to filename stem.")
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
            response = llm.chat(messages)
            title = response.message.content.strip()
            
            # Clean up potential quotes
            title = DocumentTitleGenerator._clean_title(title)
            
            logger.info(f"LLM-generated title for chunk from '{filename_stem}': '{title}'")
            return title if title else filename_stem
        except Exception as e:
            logger.error(f"Error getting title from LLM for '{filename_stem}': {e}", exc_info=True)
            return filename_stem
    
    @staticmethod
    def _clean_title(title: str) -> str:
        """Clean up LLM-generated title by removing quotes."""
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        if title.startswith(''') and title.endswith('''):
            title = title[1:-1]
        return title

# Convenience instances
llm_manager = LLMManager()
title_generator = DocumentTitleGenerator() 