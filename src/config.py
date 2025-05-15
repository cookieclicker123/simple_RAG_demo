# Configuration settings for the RAG pipeline.

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Project root directory - Assuming this file (config.py) is in simple_RAG_demo/src/
# So, parent is src/, parent.parent is simple_RAG_demo/
ROOT_DIR = Path(__file__).resolve().parent.parent

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding='utf-8', extra='ignore')

    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here_if_not_in_env")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") # Optional, for Gemini models

    # LLM Configuration
    # Specific Gemini Models for ADK Agents
    gemini_flash_model_name: str = "gemini-2.0-flash" # Or specific version like gemini-1.5-flash-001
    gemini_pro_model_name: str = "gemini-2.5-pro-exp-03-25"   # Or specific version like gemini-1.5-pro-001
    
    # Default Agent LLM (can be overridden per agent)
    default_agent_llm_model_name: str = "gemini-1.5-flash-latest" # Default to faster model

    # OpenAI Model for RAG tool (or other specific components)
    openai_compatible_llm_model_name: str = "gpt-4o-mini"
    
    temperature: float = 0.1
    max_tokens: int = 1024 # Increased for potentially more complex agent responses

    # Embedding Model Configuration
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # Vector Store Configuration
    vector_store_path: str = str(ROOT_DIR / "local_db" / "faiss_index")
    documents_dir: str = str(ROOT_DIR / "data")
    # For LlamaIndex, index_id for vector store can be useful if managing multiple indexes
    index_id: str = "main_index"

    # Chunking Configuration (example values, can be tuned)
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # Logging Configuration (placeholder)
    log_level: str = "INFO"

settings = AppSettings()

# Ensure necessary directories exist (data and local_db parent)
if not Path(settings.documents_dir).exists():
    Path(settings.documents_dir).mkdir(parents=True, exist_ok=True)

vector_store_parent_dir = Path(settings.vector_store_path).parent
if not vector_store_parent_dir.exists():
    vector_store_parent_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # For testing the configuration loading
    print(f"Calculated ROOT_DIR: {ROOT_DIR}")
    print(f"OpenAI API Key Loaded: {'Yes' if settings.openai_api_key and settings.openai_api_key != 'your_openai_api_key_here_if_not_in_env' else 'No'}")
    print(f"Gemini API Key Loaded: {'Yes' if settings.gemini_api_key else 'No'}")
    print(f"Default Agent LLM (Gemini Flash): {settings.gemini_flash_model_name}")
    print(f"Advanced Agent LLM (Gemini Pro): {settings.gemini_pro_model_name}")
    print(f"Default for agents (can be overridden): {settings.default_agent_llm_model_name}")
    print(f"OpenAI-Compatible LLM Model (for RAG): {settings.openai_compatible_llm_model_name}")
    print(f"Embedding Model: {settings.embedding_model_name}")
    print(f"Vector Store Path: {settings.vector_store_path}")
    print(f"Documents Path: {settings.documents_dir}")
    print(f"Log Level: {settings.log_level}") 