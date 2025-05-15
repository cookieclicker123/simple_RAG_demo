# CLI script for running the document indexing process.

import logging
import sys
from pathlib import Path

# Ensure the project root is in PYTHONPATH for correct imports when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now that sys.path is configured, we can import from our package
# Assuming you want to stick to the 'from simple_rag_pipeline...' style which is standard for packages
from src.core.indexing_service import run_indexing_pipeline, configure_llama_index_globals, get_active_settings
from src.config import settings # For initial log level
from llama_index.core import Settings as LlamaSettings # Import LlamaSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level) # Use log level from config

def main():
    logger.info("===============================================")
    logger.info(" RAG Pipeline Indexer CLI Started ")
    logger.info("===============================================")
    
    active_settings = get_active_settings()
    # Ensure LlamaIndex globals are configured using the most current active settings
    # This is crucial if indexer_cli is the first point of LlamaIndex interaction.
    configure_llama_index_globals(active_settings)
    
    logger.info(f"Using document directory: {active_settings.documents_dir}")
    logger.info(f"Using vector store path: {active_settings.vector_store_path}")
    # Now LlamaSettings is available to be logged
    logger.info(f"Using embedding model (from LlamaSettings): {LlamaSettings.embed_model}")
    logger.info(f"Using chunk size (from LlamaSettings): {LlamaSettings.chunk_size}")

    try:
        # run_indexing_pipeline will use the settings active at the time of its call
        # (via get_active_settings or an override, and its own call to configure_llama_index_globals if overridden)
        run_indexing_pipeline()
        logger.info("-----------------------------------------------")
        logger.info(" Indexing process completed. ")
        logger.info("-----------------------------------------------")
    except Exception as e:
        logger.error(f"An error occurred during the indexing process: {e}", exc_info=True)
        logger.info("-----------------------------------------------")
        logger.info(" Indexing process failed. ")
        logger.info("-----------------------------------------------")

if __name__ == "__main__":
    main() 