"""HTTP client utilities for API communication."""

import logging
from typing import Optional, Dict, Any
import httpx
import json

from src.config import settings

logger = logging.getLogger(__name__)

class APIClient:
    """Centralized HTTP client for API communication."""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.api_base_url
        self.endpoints = {
            'index_status': f"{self.base_url}/index/status",
            'trigger_index': f"{self.base_url}/index/documents", 
            'check_completion': f"{self.base_url}/index/check-completion",
            'cleanup_index': f"{self.base_url}/index/cleanup",
            'stream_chat': f"{self.base_url}/chat/stream",
            # Agent framework endpoints
            'agent_stream_chat': f"{self.base_url}/agent/chat/stream",
            'agent_info': f"{self.base_url}/agent/info",
            'agent_tools': f"{self.base_url}/agent/tools"
        }
    
    async def get_json(self, endpoint_key: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Make a GET request and return JSON response."""
        endpoint = self.endpoints.get(endpoint_key)
        if not endpoint:
            logger.error(f"Unknown endpoint key: {endpoint_key}")
            return None
            
        try:
            timeout = timeout or settings.api_timeout_default
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} error for {endpoint}: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {e}", exc_info=True)
            return None
    
    async def post_json(self, endpoint_key: str, data: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Make a POST request and return JSON response."""
        endpoint = self.endpoints.get(endpoint_key)
        if not endpoint:
            logger.error(f"Unknown endpoint key: {endpoint_key}")
            return None
            
        try:
            timeout = timeout or settings.api_timeout_default
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=data)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} error for {endpoint}: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {e}", exc_info=True)
            return None
    
    async def delete_json(self, endpoint_key: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Make a DELETE request and return JSON response."""
        endpoint = self.endpoints.get(endpoint_key)
        if not endpoint:
            logger.error(f"Unknown endpoint key: {endpoint_key}")
            return None
            
        try:
            timeout = timeout or settings.api_timeout_long
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.delete(endpoint)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} error for {endpoint}: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {e}", exc_info=True)
            return None

# Global client instance
api_client = APIClient() 