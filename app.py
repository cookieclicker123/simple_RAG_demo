import asyncio
import httpx
import json
import sys
import logging
from typing import Optional

# Import the new schemas for type safety
from src.server.schemas import IndexStatus, IndexCheckResult, UserConfirmation
# Import additional schemas for indexing completion detection
from src.server.schemas import IndexingState, IndexingStatusCheck, IndexCompletionRequest

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set httpx logger level to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

API_BASE_URL = "http://localhost:8000/api"
STREAM_CHAT_ENDPOINT = f"{API_BASE_URL}/chat/stream"
INDEX_STATUS_ENDPOINT = f"{API_BASE_URL}/index/status"
TRIGGER_INDEX_ENDPOINT = f"{API_BASE_URL}/index/documents"
CHECK_INDEXING_COMPLETION_ENDPOINT = f"{API_BASE_URL}/index/check-completion"

async def get_enhanced_index_status() -> Optional[IndexCheckResult]:
    """
    Gets index status from server and returns a structured IndexCheckResult.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(INDEX_STATUS_ENDPOINT)
            response.raise_for_status()
            server_data = response.json()
            
        # Transform server response into our enhanced schema
        exists = server_data.get('exists', False)
        doc_count = server_data.get('document_count_in_data_folder', 0)
        server_message = server_data.get('message', 'No message from server')
        
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
        
    except httpx.RequestError as e:
        logger.error(f"Error requesting index status: {e}")
        print(f"\nError: Could not connect to server to get index status. Is it running at {API_BASE_URL}?")
    except Exception as e:
        logger.error(f"Error parsing index status response: {e}", exc_info=True)
        print(f"\nError: Could not parse index status from server.")
    return None

def get_user_confirmation(prompt: str, action: str) -> UserConfirmation:
    """
    Gets user confirmation with type safety. Keeps asking until valid input.
    """
    while True:
        try:
            user_input = input(f"{prompt} (yes/no): ").strip().lower()
            if user_input == 'yes':
                return UserConfirmation(
                    confirmed=True,
                    action=action,
                    message=f"User confirmed: {action}"
                )
            elif user_input == 'no':
                return UserConfirmation(
                    confirmed=False,
                    action=action,
                    message=f"User declined: {action}"
                )
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return UserConfirmation(
                confirmed=False,
                action="cancelled",
                message="User cancelled the operation"
            )

async def get_index_status_from_server() -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(INDEX_STATUS_ENDPOINT)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error requesting index status: {e}")
        print(f"\nError: Could not connect to server to get index status. Is it running at {API_BASE_URL}?")
    except Exception as e:
        logger.error(f"Error parsing index status response: {e}", exc_info=True)
        print(f"\nError: Could not parse index status from server.")
    return None

async def trigger_server_indexing() -> bool:
    print("Sending request to server to start indexing documents from the 'data' folder...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client: # Short timeout, as it's a background task
            response = await client.post(TRIGGER_INDEX_ENDPOINT)
            response.raise_for_status()
            response_data = response.json()
            print(f"Server response: {response_data.get('message', 'Request received.')}")
            print("Indexing is running in the background on the server. Check server logs for progress.")
            print("Please wait a moment for indexing to complete before starting a chat.")
            # We might want a loop here to periodically check /index/status again until exists=true,
            # but for a simple CLI, just informing the user is a start.
            return True
    except httpx.RequestError as e:
        logger.error(f"Error triggering indexing: {e}")
        print(f"\nError: Could not connect to server to trigger indexing. Details: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error response from server when triggering indexing: {e.response.status_code} - {e.response.text}")
        try:
            detail = e.response.json().get("detail", e.response.text)
        except:
            detail = e.response.text
        print(f"\nError from server: {detail}")
    except Exception as e:
        logger.error(f"Unexpected error triggering indexing: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred while trying to start indexing: {e}")
    return False

async def check_and_manage_index() -> bool:
    """
    Enhanced index checking with continuous prompting and type safety.
    Returns True if ready to chat, never terminates the app for missing index.
    """
    print("Checking document index status...")
    
    # First check - determine initial state
    status_result = await get_enhanced_index_status()
    
    if not status_result:
        print("Could not retrieve index status from the server. Please ensure the server is running.")
        return False  # Cannot proceed due to server connectivity issues
    
    print(f"\nStatus: {status_result.message}")
    
    # Handle different index status scenarios
    if status_result.status == IndexStatus.EMPTY_DATA_FOLDER:
        print("❌ Cannot proceed: No documents found in the server's 'data' folder.")
        print("Please add PDF documents to the server's 'data' folder and restart the application.")
        return False  # Cannot proceed without documents
        
    elif status_result.status == IndexStatus.MISSING:
        # Index missing but documents available - require indexing
        print(f"\n📁 Found {status_result.document_count} document(s) ready for indexing.")
        print("🔄 The system will now index these documents to enable chat functionality.")
        
        # Keep asking until user confirms indexing (required for chat)
        while True:
            confirmation = get_user_confirmation(
                "Proceed with indexing the documents", 
                "index_documents"
            )
            
            if confirmation.confirmed:
                success = await trigger_server_indexing()
                if success:
                    print("✅ Indexing initiated successfully.")
                    
                    # Use automatic polling instead of manual user input
                    polling_success = await poll_for_indexing_completion(poll_interval=1.0, max_wait_time=300.0)
                    if polling_success:
                        print("✅ Proceeding to chat with newly created index.")
                        return True  # Ready to chat
                    else:
                        print("❌ Indexing completion could not be confirmed.")
                        # Ask if they want to try again or proceed anyway
                        proceed_confirmation = get_user_confirmation(
                            "Would you like to proceed to chat anyway (indexing may still be in progress)",
                            "proceed_anyway"
                        )
                        if proceed_confirmation.confirmed:
                            print("✅ Proceeding to chat (indexing may still be running).")
                            return True
                        else:
                            print("Please check server logs and try again.")
                            return False
                else:
                    print("❌ Failed to initiate indexing. Please check server logs.")
                    # Ask if they want to try again
                    retry_confirmation = get_user_confirmation(
                        "Would you like to try indexing again", 
                        "retry_indexing"
                    )
                    if not retry_confirmation.confirmed:
                        print("Cannot proceed without indexing. Please resolve the issue and restart.")
                        return False
                    # Continue loop to try again
                    continue
            else:
                # User declined indexing - but we need it, so ask again
                print("⚠️  Indexing is required to enable chat functionality.")
                print("Without indexing, the system cannot answer questions about your documents.")
                # Continue the loop to ask again
                continue
            
    elif status_result.status == IndexStatus.EXISTS:
        # Index already exists at startup - offer re-indexing option
        if status_result.document_count == 0:
            print("⚠️  Warning: Index exists, but no documents currently in the server's 'data' folder.")
            print("Chat will be based on the existing index content.")
        else:
            print(f"✅ Index ready with {status_result.document_count} document(s).")
        
        # Give user option to re-index or proceed directly to chat
        print("You can now start chatting, or optionally re-index the documents first.")
        reindex_confirmation = get_user_confirmation(
            "Would you like to re-index before chatting",
            "reindex_documents"
        )
        
        if reindex_confirmation.confirmed:
            success = await trigger_server_indexing()
            if success:
                print("✅ Re-indexing initiated successfully.")
                
                # Use automatic polling instead of manual user input
                polling_success = await poll_for_indexing_completion(poll_interval=1.0, max_wait_time=300.0)
                if polling_success:
                    print("✅ Proceeding to chat with updated index.")
                    return True  # Ready to chat
                else:
                    print("❌ Re-indexing completion could not be confirmed.")
                    print("Proceeding with existing index.")
                    return True  # Ready to chat with existing index
            else:
                print("❌ Failed to initiate re-indexing.")
                print("Proceeding with existing index.")
                return True  # Ready to chat with existing index
        else:
            # User declined re-indexing, proceed with existing index
            print("✅ Proceeding to chat with existing index.")
            return True  # Ready to chat
    
    # This point should not be reached
    print("Unexpected status. Please restart the application.")
    return False

async def get_streaming_chat_response(query: str):
    """
    Connects to the FastAPI streaming endpoint and yields response tokens.
    """
    payload = {"query": query}
    try:
        async with httpx.AsyncClient(timeout=None) as client: # Timeout None for long streams
            async with client.stream("POST", STREAM_CHAT_ENDPOINT, json=payload) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    logger.error(f"Error from server: {response.status_code} - {error_content.decode()}")
                    print(f"\nError: Received status {response.status_code} from server.", flush=True)
                    print(f"Details: {error_content.decode()}", flush=True)
                    yield {"event": "error", "data": f"Server error: {response.status_code}"} # Yield error event
                    return

                # Process the Server-Sent Events (SSE) stream
                buffer = ""
                async for line in response.aiter_lines():
                    if not line.strip(): # Empty lines separate events in SSE
                        if buffer.startswith("data:"):
                            data_json_str = buffer[len("data:"):].strip()
                            try:
                                data = json.loads(data_json_str)
                                # Always yield the full parsed data dictionary
                                yield data 
                                if data.get("event") == "stream_end":
                                    logger.info("app.py: Received stream_end event from server.")
                                    break
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode JSON from stream: {data_json_str}")
                                yield {"event": "error", "data": "Failed to decode stream data"}
                        buffer = ""
                    else:
                        buffer += line + "\n"
        # Ensure stream_end is yielded if the loop finishes because of a break after stream_end from server
        # or if the stream ends without an explicit server-side stream_end event (e.g. connection closed by server after data)
        # However, if stream_end was already processed and yielded above, this would be redundant.
        # The current logic yields data then breaks, so this additional yield might be okay or might need a flag.
        # For now, let's assume the server always sends stream_end. If not, this might be needed.
        # yield {"event": "stream_end"} # Re-evaluate if server doesn't always send stream_end

    except httpx.ConnectError as e:
        logger.error(f"Connection error: Could not connect to the server at {STREAM_CHAT_ENDPOINT}. Details: {e}")
        print(f"\nError: Could not connect to the server. Please ensure it's running.", flush=True)
        yield {"event": "error", "data": "Connection error"}
    except Exception as e:
        logger.error(f"An unexpected error occurred while streaming: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        yield {"event": "error", "data": f"Unexpected streaming error: {e}"}

async def poll_for_indexing_completion(poll_interval: float = 1.0, max_wait_time: float = 300.0) -> bool:
    """
    Polls the server to check if indexing is complete by verifying file structure.
    Returns True if indexing completed successfully, False if timeout or error.
    """
    import time
    start_time = time.time()
    print("🔍 Monitoring indexing progress...")
    
    last_state = None
    last_files_count = -1
    shown_completion_message = False
    
    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    CHECK_INDEXING_COMPLETION_ENDPOINT, 
                    json={"check_files": True}
                )
                response.raise_for_status()
                completion_data = response.json()
                
                # Parse the response into our schema
                status_check = IndexingStatusCheck(**completion_data)
                
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
                    if status_check.state == IndexingState.NOT_STARTED:
                        if state_changed:  # Only show this once when transitioning to NOT_STARTED
                            print("⏳ Waiting for indexing to begin...")
                    elif status_check.state == IndexingState.IN_PROGRESS:
                        if current_files_count == 0 and state_changed:
                            print("📝 Document processing started...")
                        elif current_files_count > 0 and files_progress:
                            # Estimate progress based on files created (rough heuristic)
                            if current_files_count >= 3:  # Most core files created
                                print("📄 Document processing nearly complete...")
                            elif current_files_count >= 1:  # Some files created
                                print("📄 Document processing in progress...")
                
                # Update tracking variables
                last_state = status_check.state
                last_files_count = current_files_count
                
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    print(f"⏰ Timeout reached ({max_wait_time}s). Indexing may still be in progress.")
                    print("You can check server logs for more details.")
                    return False
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
        except httpx.RequestError as e:
            logger.error(f"Error polling indexing completion: {e}")
            print(f"❌ Error checking indexing status: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during polling: {e}", exc_info=True)
            print(f"❌ Unexpected error: {e}")
            return False

async def main():
    print("Interactive Chat with RAG API (type 'exit' or 'quit' to end)")
    print("Ensure the FastAPI server is running on http://localhost:8000")
    print("-------------------------------------------------------------")

    ready_to_chat = await check_and_manage_index()
    if not ready_to_chat:
        print("Exiting application as indexing prerequisite not met or user chose to exit.")
        return

    print("-------------------------------------------------------------")
    print("Starting chat session...")

    while True:
        try:
            user_query = await asyncio.to_thread(input, "You: ")
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except EOFError:
            print("\nExiting chat due to EOF...")
            break

        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        if not user_query.strip():
            continue

        print("AI: ", end="", flush=True)
        ai_response_printed = False
        citations_to_print = []
        stream_ended_properly = False

        async for event_data in get_streaming_chat_response(user_query):
            if not isinstance(event_data, dict): # Should not happen with the fix
                logger.warning(f"Received non-dict event data: {event_data}")
                continue

            if event_data.get("event") == "stream_end":
                stream_ended_properly = True
                break
            
            if event_data.get("event") == "error":
                print(f"\n[Error: {event_data.get('data', 'Unknown stream error')}]", flush=True)
                ai_response_printed = True 
                break 

            if "token" in event_data:
                token_text = event_data["token"]
                if event_data.get("error"):
                    print(f"\n[Stream error from server: {token_text}]", flush=True)
                    ai_response_printed = True
                else:
                    print(token_text, end="", flush=True)
                    ai_response_printed = True
            elif "citations" in event_data:
                citations_to_print = event_data["citations"]
        
        if not ai_response_printed and not stream_ended_properly and not citations_to_print:
            # This condition tries to catch cases where the stream might have broken abruptly
            # without an explicit error event from get_streaming_chat_response or a proper stream_end.
            print("[No valid response or stream incomplete]", flush=True)
        elif not ai_response_printed and not citations_to_print:
             # This covers cases where stream ended (or errored out cleanly) but no actual tokens were printed
             print("[No text content in response]", flush=True)
        print() 

        if citations_to_print:
            print("\n--- Citations ---")
            for cit in citations_to_print:
                print(f"  Document ID:   {cit.get('document_id', 'N/A')}")
                print(f"  Document Name: {cit.get('document_name', 'N/A')}")
                print(f"  Title:         {cit.get('document_title', 'N/A')}")
                print(f"  File Path:     {cit.get('file_path', 'N/A')}")
                print(f"  Page Label:    {cit.get('page_label', 'N/A')}")
                print(f"  Snippet:       {cit.get('snippet', 'N/A')}\n")
        
        logger.info(f"User query: {user_query}")
        logger.info("app.py: Reached end of 'while True' iteration, should re-prompt.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user (KeyboardInterrupt).")
        print("\nApplication exited.")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}") 