"""Polling utilities for monitoring async operations."""

import asyncio
import logging
import time
from typing import Optional

from src.server.schemas import IndexingState, IndexingStatusCheck
from src.utils.index_manager import index_manager
from src.config import settings

logger = logging.getLogger(__name__)

class IndexingPoller:
    """Polls for indexing completion with smart progress updates."""
    
    def __init__(
        self, 
        poll_interval: Optional[float] = None,
        max_wait_time: Optional[float] = None
    ):
        self.poll_interval = poll_interval or settings.polling_interval_seconds
        self.max_wait_time = max_wait_time or settings.polling_max_wait_seconds
        
    async def wait_for_completion(self) -> bool:
        """
        Polls server to check if indexing is complete.
        Returns True if completed successfully, False if timeout/error.
        """
        start_time = time.time()
        print("🔍 Monitoring indexing progress...")
        
        last_state = None
        last_files_count = -1
        shown_completion_message = False
        
        while True:
            try:
                status_check = await index_manager.check_completion()
                if not status_check:
                    logger.error("Failed to get indexing status")
                    return False
                
                # Only show updates on state changes or meaningful progress
                current_files_count = len(status_check.files_found)
                state_changed = last_state != status_check.state
                files_progress = current_files_count > last_files_count
                
                if status_check.is_complete and not shown_completion_message:
                    print("✅ Indexing completed successfully!")
                    return True
                elif status_check.state == IndexingState.FAILED:
                    print("❌ Indexing appears to have failed.")
                    return False
                elif state_changed or files_progress:
                    self._show_progress_update(status_check, current_files_count, state_changed)
                
                # Update tracking variables
                last_state = status_check.state
                last_files_count = current_files_count
                
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.max_wait_time:
                    print(f"⏰ Timeout reached ({self.max_wait_time}s). Indexing may still be in progress.")
                    print("You can check server logs for more details.")
                    return False
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Unexpected error during polling: {e}", exc_info=True)
                print(f"❌ Error monitoring indexing progress: {e}")
                return False
    
    def _show_progress_update(
        self, 
        status_check: IndexingStatusCheck,
        current_files_count: int,
        state_changed: bool
    ):
        """Show appropriate progress message based on state and file count."""
        if status_check.state == IndexingState.NOT_STARTED:
            if state_changed:  # Only show once when transitioning to NOT_STARTED
                print("⏳ Waiting for indexing to begin...")
        elif status_check.state == IndexingState.IN_PROGRESS:
            if current_files_count == 0 and state_changed:
                print("📝 Document processing started...")
            elif current_files_count > 0 and current_files_count != getattr(self, '_last_reported_count', -1):
                # Estimate progress based on files created (rough heuristic)
                if current_files_count >= 3:  # Most core files created
                    print("📄 Document processing nearly complete...")
                elif current_files_count >= 1:  # Some files created
                    print("📄 Document processing in progress...")
                self._last_reported_count = current_files_count

# Convenience function for simple polling
async def poll_for_indexing_completion(
    poll_interval: Optional[float] = None,
    max_wait_time: Optional[float] = None
) -> bool:
    """Convenience function for polling indexing completion."""
    poller = IndexingPoller(poll_interval, max_wait_time)
    return await poller.wait_for_completion() 