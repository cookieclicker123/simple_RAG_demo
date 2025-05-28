# Configuration settings for the RAG pipeline.

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Suppress HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Project root directory - Assuming this file (config.py) is in simple_RAG_demo/src/
# So, parent is src/, parent.parent is simple_RAG_demo/
ROOT_DIR = Path(__file__).resolve().parent.parent

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding='utf-8', extra='ignore')

    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here_if_not_in_env")
    
    
    # Default Agent LLM (can be overridden per agent)
    llm_model_name: str = "gpt-4o-mini"
    
    temperature: float = 0.1
    max_tokens: int = 2048 # Increased for potentially more complex agent responses

    # Ollama LLM for Query Expansion
    ollama_model_for_query_expansion: str = "gemma3:4b" # Default to gemma:2b for query expansion
    ollama_request_timeout: float = 120.0

    # Embedding Model Configuration
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"

    # Vector Store Configuration
    vector_store_path: str = str(ROOT_DIR / "local_db" / "faiss_index")
    documents_dir: str = str(ROOT_DIR / "data")
    # For LlamaIndex, index_id for vector store can be useful if managing multiple indexes
    index_id: str = "main_index"

    # Chunking Configuration (example values, can be tuned)
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # Retriever Configuration
    retriever_similarity_top_k: int = 20  # Number of initial candidates to fetch by the dense retriever for fusion
    bm25_similarity_top_k: int = 10 # Number of initial candidates for BM25 retriever before fusion

    # Reranker Configuration
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 5  # Number of candidates to keep after reranking

    # Logging Configuration (placeholder)
    log_level: str = "DEBUG"

    # Chat Engine Configuration (examples, can be tuned)
    chat_memory_token_limit: int = 3000
    chat_engine_verbose: bool = True

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
    print(f"OpenAI-Compatible LLM Model (for RAG): {settings.openai_compatible_llm_model_name}")
    print(f"Embedding Model: {settings.embedding_model_name}")
    print(f"Vector Store Path: {settings.vector_store_path}")
    print(f"Documents Path: {settings.documents_dir}")
    print(f"Log Level: {settings.log_level}") 