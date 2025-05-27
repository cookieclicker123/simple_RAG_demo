# Service for document indexing: parsing, chunking, embedding, and storing in vector DB.

import logging
import os
from typing import List, Optional
from pathlib import Path
import shutil
import pickle # Re-adding for BM25 persistence

from llama_index.core import Document, Settings as LlamaSettings, VectorStoreIndex, StorageContext
from llama_index.core.embeddings import resolve_embed_model
# from llama_index.core.node_parser import SentenceSplitter # Not explicitly used if relying on LlamaSettings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # Import for type checking
from llama_index.retrievers.bm25 import BM25Retriever # Added for BM25
import faiss

from rank_bm25 import BM25Okapi # For type hinting the pickled object

# Corrected imports
from config import settings as initial_settings_from_config, AppSettings
from utils.file_handlers import load_documents_from_directory

logger = logging.getLogger(__name__)
logging.basicConfig(level=initial_settings_from_config.log_level)

# Store the currently active settings. This can be updated by the test script.
_active_settings = initial_settings_from_config

def get_active_settings() -> AppSettings:
    """Returns the currently active application settings."""
    return _active_settings

def configure_llama_index_globals(settings_to_use: AppSettings):
    """Sets LlamaIndex global settings based on the provided settings object."""
    logger.debug(f"Configuring LlamaIndex globals. OpenAI API Key present: {settings_to_use.openai_api_key and settings_to_use.openai_api_key != 'your_openai_api_key_here_if_not_in_env'}")
    logger.debug(f"Embedding model from settings: {settings_to_use.embedding_model_name}")
    
    # This will load the HuggingFaceEmbedding model from llama-index-embeddings-huggingface
    LlamaSettings.embed_model = resolve_embed_model(f"local:{settings_to_use.embedding_model_name}")
    
    # If you were to use OpenAI embeddings, you would configure it here, e.g.:
    # if module_settings.openai_api_key and module_settings.openai_api_key != "your_openai_api_key_here_if_not_in_env":
    #     from llama_index.embeddings.openai import OpenAIEmbedding
    #     LlamaSettings.embed_model = OpenAIEmbedding(
    #         api_key=module_settings.openai_api_key,
    #         model_name="text-embedding-ada-002" # Or another OpenAI embedding model
    #     )
    #     LlamaSettings.llm = "default" # Placeholder, as LLM might be OpenAI too
    # else:
    #     logger.warning("OpenAI API key not configured for OpenAI embeddings. Falling back to local.")
    #     LlamaSettings.embed_model = resolve_embed_model(f"local:{module_settings.embedding_model_name}")

    LlamaSettings.chunk_size = settings_to_use.chunk_size
    LlamaSettings.chunk_overlap = settings_to_use.chunk_overlap
    logger.info(f"LlamaIndex global embed_model set to: {LlamaSettings.embed_model} (type: {type(LlamaSettings.embed_model)})")
    logger.info(f"LlamaIndex global chunk_size set to: {LlamaSettings.chunk_size}, overlap: {LlamaSettings.chunk_overlap}")

# Configure LlamaIndex settings when the module is loaded with initial settings.
configure_llama_index_globals(get_active_settings())
# --- End LlamaIndex Global Settings Configuration ---

def get_embedding_dimension(embed_model: HuggingFaceEmbedding) -> int:
    """Safely retrieves the embedding dimension from a HuggingFaceEmbedding model."""
    if hasattr(embed_model, '_model') and hasattr(embed_model._model, 'get_sentence_embedding_dimension'):
        return embed_model._model.get_sentence_embedding_dimension()
    elif hasattr(embed_model, 'dimensions'): # Some embedding models might have a 'dimensions' attr
        return embed_model.dimensions
    # Fallback or raise error if dimension cannot be determined
    # For BAAI/bge-small-en-v1.5, it's 384. This could be a config fallback.
    logger.warning("Could not dynamically determine embedding dimension. Falling back to default 384 for BAAI/bge-small-en-v1.5. This might be incorrect for other models.")
    return 384

def create_and_persist_index(documents: List[Document], vector_store_path_str: str) -> bool:
    """Creates a FAISS vector index from documents and persists it to disk.
    Uses the globally configured LlamaIndex settings.
    """
    if not documents:
        logger.warning("No documents provided for indexing.")
        return False

    logger.info(f"Starting indexing for {len(documents)} documents.")
    logger.info(f"Using LlamaSettings: embed_model={LlamaSettings.embed_model}, chunk_size={LlamaSettings.chunk_size}, chunk_overlap={LlamaSettings.chunk_overlap}")

    try:
        if not isinstance(LlamaSettings.embed_model, HuggingFaceEmbedding):
            logger.error(f"LlamaSettings.embed_model is not a HuggingFaceEmbedding instance, but {type(LlamaSettings.embed_model)}. Cannot determine dimension automatically for FAISS.")
            return False
        
        embed_dim = get_embedding_dimension(LlamaSettings.embed_model)
        logger.info(f"Resolved embedding dimension: {embed_dim}")
        
        faiss_index_instance = faiss.IndexFlatL2(embed_dim)
        vector_store_path = Path(vector_store_path_str)
        
        # Define the path for the FaissVectorStore components (binary and its JSON metadata)
        faiss_component_dir = vector_store_path / "vector_store"
        faiss_component_dir.mkdir(parents=True, exist_ok=True)
        faiss_binary_persist_path = faiss_component_dir / "default__vector_store.faiss"
        
        # Create FaissVectorStore instance
        fvs = FaissVectorStore(faiss_index=faiss_index_instance)

        # Create a new StorageContext for this indexing operation
        # This will hold the docstore, vector_store, and index_store.
        storage_context = StorageContext.from_defaults(
            vector_store=fvs # Pass the FaissVectorStore instance
        )

        # The VectorStoreIndex will use the docstore from this storage_context to store nodes.
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context, 
            # transformations are implicitly handled by LlamaSettings.chunk_size/overlap if not overridden here
        )
        logger.info(f"Successfully created vector index in memory. Index ID: {index.index_id}")
        
        # Ensure the main persist directory exists
        if not vector_store_path.exists():
            vector_store_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created vector store directory: {vector_store_path}")

        # 1. Explicitly persist the FaissVectorStore to its subdirectory
        # This will create default__vector_store.faiss and default__vector_store.json in the faiss_component_dir.
        fvs.persist(persist_path=str(faiss_binary_persist_path)) # LlamaIndex handles naming for .json
        logger.info(f"Explicitly persisted FaissVectorStore components to: {faiss_component_dir}")

        # 2. Persist the rest of the storage_context (docstore, index_store) to the main directory.
        #    The vector_store metadata in the root (default__vector_store.json created by this call)
        #    might be problematic if it conflicts with the one in the subdirectory.
        #    Let's try persisting specific components instead of the whole context again,
        #    or ensure the loader prioritizes the one in the subdirectory.
        #    For now, our loader explicitly loads the FAISS binary and then uses the root docstore/indexstore.
        storage_context.docstore.persist(persist_path=str(vector_store_path / "docstore.json"))
        storage_context.index_store.persist(persist_path=str(vector_store_path / "index_store.json"))
        # We do NOT call storage_context.persist() here to avoid it writing its own default__vector_store.json
        # at the root, which might be the source of the UnicodeDecodeError if it gets corrupted.
        logger.info(f"Successfully persisted docstore and index_store to: {vector_store_path}")

        # --- BM25 Persistence (Pickling Attempt 2) --- 
        logger.info("Attempting to create and persist BM25 engine via pickling...")
        all_nodes_for_bm25 = list(storage_context.docstore.docs.values())
        if all_nodes_for_bm25:
            try:
                # Create a temporary BM25Retriever to build the BM25 engine
                temp_bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes_for_bm25, tokenizer=None)
                
                # Try to access the underlying rank_bm25 object via .bm25 (direct attribute)
                bm25_engine_to_persist = temp_bm25_retriever.bm25 
                
                bm25_engine_path = vector_store_path / "bm25_engine.pkl"
                with open(bm25_engine_path, "wb") as f:
                    pickle.dump(bm25_engine_to_persist, f)
                logger.info(f"Successfully persisted BM25 engine to: {bm25_engine_path}")
            except AttributeError as ae:
                logger.error(f"AttributeError accessing BM25 engine (tried .bm25): {ae}. BM25 will not be persisted.", exc_info=True)
            except Exception as e_bm25_persist:
                logger.error(f"Error during BM25 engine pickling: {e_bm25_persist}. BM25 will not be persisted.", exc_info=True)
        else:
            logger.warning("No nodes found in docstore for BM25 engine creation.")
        # --- End BM25 Persistence ---

        logger.info("BM25 engine (if created and pickled) saved. It will be loaded or created on-the-fly in qa_service.")

        return True

    except Exception as e:
        logger.error(f"Error during index creation or persistence: {e}", exc_info=True)
        return False

def run_indexing_pipeline(settings_override: Optional[AppSettings] = None):
    """Runs the full document indexing pipeline.
    Loads documents, creates embeddings, and stores them in the vector store.
    Uses active settings, or an override if provided (for testing).
    """
    current_settings = settings_override if settings_override else get_active_settings()
    if settings_override:
        logger.info("Running indexing pipeline with overridden settings.")
        # If settings are overridden, reconfigure LlamaIndex globals for this run context
        # This is a simplification; for true isolation, instances would be better.
        configure_llama_index_globals(current_settings)
    
    logger.info("Starting document indexing pipeline...")
    logger.info(f"Using document directory: {current_settings.documents_dir}")
    logger.info(f"Using vector store path: {current_settings.vector_store_path}")

    documents = load_documents_from_directory(current_settings.documents_dir)
    if not documents:
        logger.warning("No documents found to index. Exiting pipeline.")
        return

    success = create_and_persist_index(documents, current_settings.vector_store_path)
    if success:
        logger.info("Document indexing pipeline completed successfully.")
    else:
        logger.error("Document indexing pipeline failed.")
    
    # Restore initial LlamaIndex global config if it was overridden for a specific run
    if settings_override:
        configure_llama_index_globals(get_active_settings()) # Restore to module's active settings
        logger.info("Restored LlamaIndex globals to initial active settings after override.")

if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # AppSettings is already imported from simple_rag_pipeline.config at the top level
    # We also have `_active_settings` available from the module scope.

    logger.info("--- Starting Indexing Service Test --- ")
    
    current_test_settings = get_active_settings() # Start with initial settings

    env_file_path = PROJECT_ROOT / ".env"
    if not env_file_path.exists():
        with open(env_file_path, "w") as f:
            f.write("OPENAI_API_KEY=sk-dummykeyfortestingfromindexingservice\n")
        logger.info(f"Created dummy .env file at {env_file_path} for testing purposes.")
        # Reload settings from the newly created .env file for this test run
        current_test_settings = AppSettings() # This is a new local instance for testing
        logger.info(f"Reloaded AppSettings for test due to .env creation.")
        # Note: LlamaIndex globals will be reconfigured by run_indexing_pipeline if override is passed

    test_data_dir = Path(current_test_settings.documents_dir)
    if not test_data_dir.exists():
        test_data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_files_content = {
        "sample_doc1_is.txt": "The quick brown fox jumps over the lazy dog for indexing service.",
        "sample_doc2_is.txt": "LangChain and LlamaIndex are popular for RAG. This is another test document.",
    }
    for filename, content in sample_files_content.items():
        file_path = test_data_dir / filename
        if not file_path.exists():
            with open(file_path, "w") as f: f.write(content)
            logger.info(f"Created sample file for test: {file_path}")

    logger.info(f"Test using OpenAI API Key from settings: {current_test_settings.openai_api_key and current_test_settings.openai_api_key != 'your_openai_api_key_here_if_not_in_env'}")
    # LlamaSettings are global, their state depends on last call to configure_llama_index_globals
    logger.info(f"Test using Embedding Model from LlamaSettings (will be set by run_indexing_pipeline if overridden): {LlamaSettings.embed_model}")

    test_vector_store_dir = Path(current_test_settings.vector_store_path)
    if test_vector_store_dir.exists():
        logger.info(f"Removing existing test index at {test_vector_store_dir} for a clean run.")
        shutil.rmtree(test_vector_store_dir)
    test_vector_store_dir.parent.mkdir(parents=True, exist_ok=True)
    
    run_indexing_pipeline(settings_override=current_test_settings) # Pass the potentially modified settings
    logger.info("--- Indexing Service Test Run Complete ---")

    if test_vector_store_dir.exists() and any(f for f in test_vector_store_dir.iterdir() if f.name != '.gitkeep'):
        logger.info(f"Index files found in {test_vector_store_dir}. Content: {[f.name for f in test_vector_store_dir.iterdir()]}")
    else:
        logger.error(f"Index directory {test_vector_store_dir} is empty or does not exist after test indexing (or only contains .gitkeep).") 