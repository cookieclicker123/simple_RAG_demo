# Utility functions for vector store interactions (creation, loading, saving, etc.).

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings as LlamaSettings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
# For vector store, FAISS is special. Its metadata might be handled differently or with SimpleVectorStore wrapper for JSON part.
# We'll try loading SimpleVectorStore for metadata and then load the actual FAISS index.
# from llama_index.core.vector_stores import SimpleVectorStore # SimpleVectorStore might still be from core
from llama_index.vector_stores.faiss import FaissVectorStore # Corrected import for FaissVectorStore
import faiss # For actual FAISS loading

from src.config import settings

logger = logging.getLogger(__name__)
# logging.basicConfig(level=settings.log_level) # Reverted: main app should configure this

FAISS_INDEX_FILENAME = "default__vector_store.faiss" # Standard name LlamaIndex might use or expect
VECTOR_STORE_SUBDIR = "vector_store" # Standard subdirectory for FaissVectorStore components

def load_faiss_index_from_storage(vector_store_path_str: str) -> tuple[Optional[VectorStoreIndex], Optional[StorageContext]]:
    """Loads a FAISS VectorStoreIndex and its StorageContext from the specified storage path.
    Assumes LlamaIndex global settings (e.g., embed_model) are appropriately configured.
    Returns a tuple (index, storage_context) or (None, None) on failure.
    """
    vector_store_root_path = Path(vector_store_path_str)
    # FAISS binary and its JSON metadata are typically in a 'vector_store' subdirectory by default
    faiss_component_path = vector_store_root_path / VECTOR_STORE_SUBDIR
    faiss_binary_file_path = faiss_component_path / FAISS_INDEX_FILENAME

    if not vector_store_root_path.exists() or not faiss_binary_file_path.exists():
        logger.warning(f"Vector store root path {vector_store_root_path} or FAISS binary {faiss_binary_file_path} does not exist. Cannot load index.")
        return None, None

    logger.info(f"Attempting to load FAISS index from component path: {faiss_component_path}, binary: {faiss_binary_file_path}")
    logger.info(f"Using LlamaSettings.embed_model for loading: {LlamaSettings.embed_model}")

    try:
        # 1. Load the FaissVectorStore directly from the .faiss binary file
        faiss_index_object = faiss.read_index(str(faiss_binary_file_path))
        # When FaissVectorStore is persisted by StorageContext, its own metadata (default__vector_store.json)
        # is also in the faiss_component_path. FaissVectorStore.from_persist_path should find it.
        # However, we are loading the binary directly, so we construct FaissVectorStore.
        vector_store = FaissVectorStore(faiss_index=faiss_index_object)
        logger.info(f"Successfully loaded FaissVectorStore from binary: {faiss_binary_file_path}")

        # 2. Load the rest of the StorageContext (docstore, index_store) from the root persistence directory.
        docstore_path = vector_store_root_path # Docstore and IndexStore are at the root
        index_store_path = vector_store_root_path
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, 
            docstore=SimpleDocumentStore.from_persist_dir(docstore_path),
            index_store=SimpleIndexStore.from_persist_dir(index_store_path)
        )
        logger.info("Successfully created StorageContext with pre-loaded FaissVectorStore and loaded docstore/index_store from root.")

        # 3. Load the main VectorStoreIndex using the constructed storage_context
        index = load_index_from_storage(storage_context=storage_context)
        
        logger.info(f"Successfully loaded index (ID: '{index.index_id if index else 'Unknown'}') using constructed StorageContext from {vector_store_root_path}")
        return index, storage_context
    except ValueError as ve:
        logger.error(f"ValueError loading index from {vector_store_root_path}: {ve}", exc_info=True)
        if "No index found" in str(ve) or "Failed to load index with ID" in str(ve):
            logger.error("This might indicate that no index was persisted or the structure is unexpected.")
        return None, None
    except FileNotFoundError:
        logger.error(f"FAISS index files not found in {vector_store_root_path}. Has it been created and persisted correctly?", exc_info=True)
        return None, None
    except Exception as e:
        logger.error(f"General error loading FAISS index from {vector_store_root_path}: {e}", exc_info=True)
        return None, None

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
    loaded_index, loaded_storage_context = load_faiss_index_from_storage(settings.vector_store_path)

    if loaded_index and loaded_storage_context:
        print(f"\n--- vector_store_handlers.py: Successfully loaded index and storage_context ---")
        print(f"Index ID from loaded index: {loaded_index.index_id}") 
        print(f"Docstore available in storage_context: {loaded_storage_context.docstore is not None}")
    else:
        print("\n--- vector_store_handlers.py: Failed to load index and/or storage_context. ---")
        print(f"Please ensure an index exists at '{settings.vector_store_path}'. You can create one by running:")
        print(f"python -m src.core.indexing_service") # Corrected module path for direct run 