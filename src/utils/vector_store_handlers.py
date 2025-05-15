# Utility functions for vector store interactions (creation, loading, saving, etc.).

import logging
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings as LlamaSettings
# FaissVectorStore and faiss library itself are used during index CREATION (in indexing_service.py)
# For LOADING with StorageContext, they are often not directly needed here, as LlamaIndex infers the type.
# from llama_index.vector_stores.faiss import FaissVectorStore 
# import faiss
# from llama_index.core.vector_stores.types import VectorStore # General type hint, remove if not used

from src.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

def load_faiss_index_from_storage(vector_store_path_str: str) -> VectorStoreIndex | None:
    """Loads a FAISS VectorStoreIndex from the specified storage path.
    Assumes LlamaIndex global settings (e.g., embed_model) are appropriately configured.
    Tries to load the default index if only one is present.
    """
    vector_store_path = Path(vector_store_path_str)
    if not vector_store_path.exists() or not any(f for f in vector_store_path.iterdir() if f.name != ".gitkeep"):
        logger.warning(f"Vector store path {vector_store_path} does not exist or is empty (ignoring .gitkeep). Cannot load index.")
        return None

    logger.info(f"Attempting to load FAISS index from: {vector_store_path}")
    logger.info(f"Using LlamaSettings.embed_model for loading: {LlamaSettings.embed_model}")

    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(vector_store_path))
        # Try loading without specifying index_id, LlamaIndex should load the default/only index.
        index = load_index_from_storage(storage_context=storage_context)
        
        logger.info(f"Successfully loaded index (ID: '{index.index_id if index else 'Unknown'}') from {vector_store_path}")
        return index
    except ValueError as ve:
        logger.error(f"ValueError loading index from {vector_store_path}: {ve}", exc_info=True)
        if "No index found" in str(ve) or "Failed to load index with ID" in str(ve):
            logger.error("This might indicate that no index was persisted or the structure is unexpected.")
        return None
    except FileNotFoundError:
        logger.error(f"FAISS index files not found in {vector_store_path}. Has it been created and persisted correctly?", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"General error loading FAISS index from {vector_store_path}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.core.indexing_service import configure_llama_index_globals, get_active_settings as get_is_settings
    # Use settings from the config module directly, it should be consistent
    # from src.config import settings as main_settings 

    active_test_settings = get_is_settings()
    configure_llama_index_globals(active_test_settings)
    logger.info(f"vector_store_handlers test: LlamaSettings.embed_model configured to: {LlamaSettings.embed_model}")

    logger.info(f"Attempting to load index from: {settings.vector_store_path}")
    # Pass settings.index_id from config to the loading function (it has a default in the func signature anyway)
    loaded_index = load_faiss_index_from_storage(settings.vector_store_path)

    if loaded_index:
        print(f"\n--- vector_store_handlers.py: Successfully loaded index ---")
        print(f"Index ID from loaded index: {loaded_index.index_id}") 
    else:
        print("\n--- vector_store_handlers.py: Failed to load index. ---")
        print(f"Please ensure an index exists at '{settings.vector_store_path}'. You can create one by running:")
        print(f"python -m src.core.indexing_service") # Corrected module path for direct run 