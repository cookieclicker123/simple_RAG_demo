# Utility functions for file handling, such as loading and preprocessing documents.

import logging
from pathlib import Path
from typing import List

from llama_index.core import Document, SimpleDirectoryReader

# Corrected import based on pyproject.toml [tool.setuptools.packages.find]
from config import settings

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

if __name__ == "__main__":
    # This example usage block will run if the script is executed directly.
    # For it to find `simple_rag_pipeline.config` correctly when run as `python src/utils/file_handlers.py`,
    # the PYTHONPATH might need to include the project root, or you run it as `python -m simple_rag_pipeline.utils.file_handlers`
    # after installing the package in editable mode.

    # Ensure the project root is in PYTHONPATH for direct script execution if not installed
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # Now the import from simple_rag_pipeline.config should work
    from src.config import settings as test_settings

    # Example usage: Make sure you have a 'data' directory with some sample files
    # For this test, create a dummy data directory and a file if it doesn't exist
    sample_data_dir = Path(test_settings.documents_dir)
    if not sample_data_dir.exists():
        sample_data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = sample_data_dir / "sample_document_fh_test.txt"
    if not sample_file.exists():
        with open(sample_file, "w") as f:
            f.write("This is a sample document for testing file_handlers.py loading.")
        print(f"Created sample file: {sample_file}")

    # Test loading
    loaded_docs = load_documents_from_directory(test_settings.documents_dir)
    if loaded_docs:
        print(f"\n--- file_handlers.py: Loaded {len(loaded_docs)} documents: ---")
        for doc in loaded_docs:
            print(f"ID: {doc.id_}, Metadata: {doc.metadata}, Text snippet: {doc.text[:100]}...")
    else:
        print("file_handlers.py: No documents were loaded. Check the data directory and logs.") 