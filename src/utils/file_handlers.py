# Utility functions for file handling, such as loading and preprocessing documents.

import logging
from pathlib import Path
from typing import List

from llama_index.core import Document, SimpleDirectoryReader

# Corrected import based on pyproject.toml [tool.setuptools.packages.find]
from src.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

def load_documents_from_directory(directory_path_str: str) -> List[Document]:
    """Loads all supported documents from the specified directory.

    Args:
        directory_path_str: The path to the directory containing documents.

    Returns:
        A list of LlamaIndex Document objects.
    """
    directory_path = Path(directory_path_str)
    if not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path_str}")
        return []

    logger.info(f"Loading documents from: {directory_path_str}")
    try:
        # SimpleDirectoryReader can handle various file types (PDF, txt, docx, etc.)
        # It will recursively search for files in the directory.
        reader = SimpleDirectoryReader(input_dir=str(directory_path))
        documents = reader.load_data()
        logger.info(f"Successfully loaded {len(documents)} document(s).")
        
        # Attach metadata from filename if not already present
        for doc in documents:
            if not doc.metadata.get("file_name") and doc.id_:
                 # LlamaIndex typically uses id_ as file_name for SimpleDirectoryReader
                file_path = Path(doc.id_)
                doc.metadata["file_name"] = file_path.name
                doc.metadata["file_path"] = str(file_path) # Store full path if needed

        return documents
    except Exception as e:
        logger.error(f"Error loading documents from {directory_path_str}: {e}", exc_info=True)
        return []