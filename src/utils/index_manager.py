"""Index management utilities."""

import logging
from typing import Optional, List
from pathlib import Path

from src.server.schemas import (
    IndexStatus, IndexCheckResult,
    IndexingStatusCheck
)
from src.utils.http_client import api_client
from src.config import settings

logger = logging.getLogger(__name__)

class IndexManager:
    """Centralized index management operations."""
    
    @staticmethod
    async def get_enhanced_status() -> Optional[IndexCheckResult]:
        """Get enhanced index status with structured response."""
        server_data = await api_client.get_json('index_status')
        if not server_data:
            return None
            
        # Transform server response into enhanced schema
        exists = server_data.get('exists', False)
        doc_count = server_data.get('document_count_in_data_folder', 0)
        
        if not exists and doc_count == 0:
            status = IndexStatus.EMPTY_DATA_FOLDER
            needs_indexing = False
            can_proceed = False
            message = "No index found and no documents in the data folder. Please add PDF documents to the server's 'data' folder first."
        elif not exists and doc_count > 0:
            status = IndexStatus.MISSING
            needs_indexing = True
            can_proceed = False
            message = f"No index found, but {doc_count} document(s) are available for indexing."
        else:  # exists
            status = IndexStatus.EXISTS
            needs_indexing = False
            can_proceed = True
            message = f"Index exists with {doc_count} document(s) in the data folder."
            
        return IndexCheckResult(
            status=status,
            document_count=doc_count,
            message=message,
            needs_indexing=needs_indexing,
            can_proceed_without_indexing=can_proceed
        )
    
    @staticmethod
    async def trigger_indexing() -> bool:
        """Trigger server indexing process."""
        print("Sending request to server to start indexing documents from the 'data' folder...")
        
        response_data = await api_client.post_json('trigger_index')
        if response_data:
            print(f"Server response: {response_data.get('message', 'Request received.')}")
            print("Indexing is running in the background on the server. Check server logs for progress.")
            print("Please wait a moment for indexing to complete before starting a chat.")
            return True
        
        print("❌ Failed to trigger indexing. Check server connection and logs.")
        return False
    
    @staticmethod 
    async def cleanup_existing() -> bool:
        """Clean up existing index files before re-indexing."""
        cleanup_data = await api_client.delete_json('cleanup_index')
        if not cleanup_data:
            return False
            
        message = cleanup_data.get('message', 'Cleanup completed')
        files_deleted = cleanup_data.get('files_deleted', [])
        
        if files_deleted:
            print(f"🗑️  {message}")
            for file_info in files_deleted[:3]:  # Show first 3 files to avoid clutter
                print(f"    - Deleted {file_info}")
            if len(files_deleted) > 3:
                print(f"    - ... and {len(files_deleted) - 3} more file(s)")
        else:
            print(f"ℹ️  {message}")
        
        return True
    
    @staticmethod
    async def check_completion() -> Optional[IndexingStatusCheck]:
        """Check if indexing is complete by verifying file structure."""
        completion_data = await api_client.post_json(
            'check_completion', 
            {"check_files": True}
        )
        if not completion_data:
            return None
            
        return IndexingStatusCheck(**completion_data)

class IndexFileStructure:
    """Utilities for working with index file structure."""
    
    @staticmethod
    def get_required_files(base_path: Path) -> dict[str, Path]:
        """Get dictionary of required index files with their full paths."""
        return {
            "faiss_index": base_path / settings.required_index_files[0],
            "docstore": base_path / settings.required_index_files[1], 
            "index_store": base_path / settings.required_index_files[2],
            "bm25_engine": base_path / settings.required_index_files[3]
        }
    
    @staticmethod
    def check_files_exist(base_path: Path) -> tuple[List[str], List[str]]:
        """Check which required files exist and which are missing.
        
        Returns:
            Tuple of (files_found, files_missing) as lists of file paths.
        """
        required_files = IndexFileStructure.get_required_files(base_path)
        files_found = []
        files_missing = []
        
        for file_key, file_path in required_files.items():
            if file_path.exists() and file_path.is_file():
                files_found.append(str(file_path))
            else:
                files_missing.append(str(file_path))
        
        return files_found, files_missing

# Convenience instances
index_manager = IndexManager()
file_structure = IndexFileStructure() 