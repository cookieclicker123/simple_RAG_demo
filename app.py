import asyncio
import httpx
import json
import logging
import uuid

# Import the new schemas for type safety
from src.server.schemas import IndexStatus
# Import additional schemas for indexing completion detection


# Import utilities for DRY code
from src.utils.index_manager import index_manager
from src.utils.polling import poll_for_indexing_completion
from src.utils.user_interaction import user_prompts
from src.utils.http_client import api_client
from src.config import settings

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set httpx logger level to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global conversation session ID for in-session memory
current_session_id = None

async def check_and_manage_index() -> bool:
    """
    Enhanced index checking with continuous prompting and type safety.
    Returns True if ready to chat, never terminates the app for missing index.
    """
    print("Checking document index status...")
    
    # First check - determine initial state using utility
    status_result = await index_manager.get_enhanced_status()
    
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
            confirmation = user_prompts.get_confirmation(
                "Proceed with indexing the documents", 
                "index_documents"
            )
            
            if confirmation.confirmed:
                success = await index_manager.trigger_indexing()
                if success:
                    print("✅ Indexing initiated successfully.")
                    
                    # Use automatic polling instead of manual user input
                    polling_success = await poll_for_indexing_completion()
                    if polling_success:
                        print("✅ Proceeding to chat with newly created index.")
                        return True  # Ready to chat
                    else:
                        print("❌ Indexing completion could not be confirmed.")
                        # Ask if they want to try again or proceed anyway
                        proceed_confirmation = user_prompts.get_confirmation(
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
                    retry_confirmation = user_prompts.get_confirmation(
                        "Would you like to try indexing again", 
                        "retry_indexing"
                    )
                    if not retry_confirmation.confirmed:
                        print("Cannot proceed without indexing. Please resolve the issue and restart.")
                        return False
                    # Continue loop to try again
                    continue
            elif confirmation.action == "exit":
                print("👋 Exiting application as requested.")
                return False
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
        reindex_confirmation = user_prompts.get_confirmation(
            "Would you like to re-index before chatting",
            "reindex_documents"
        )
        
        if reindex_confirmation.confirmed:
            # First, clean up existing index files to prevent false completion detection
            print("🧹 Cleaning up existing index files before re-indexing...")
            cleanup_success = await index_manager.cleanup_existing()
            if not cleanup_success:
                print("⚠️  Warning: Could not fully clean existing index files, but proceeding anyway.")
            
            success = await index_manager.trigger_indexing()
            if success:
                print("✅ Re-indexing initiated successfully.")
                
                # Use automatic polling instead of manual user input
                polling_success = await poll_for_indexing_completion()
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
        elif reindex_confirmation.action == "exit":
            print("👋 Exiting application as requested.")
            return False
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
    Now includes session management for conversation memory.
    """
    global current_session_id
    
    # Initialize session ID if not already set
    if not current_session_id:
        current_session_id = str(uuid.uuid4())
        logger.info(f"Started new conversation session: {current_session_id}")
    
    payload = {
        "query": query,
        "session_id": current_session_id
    }
    stream_endpoint = api_client.endpoints['stream_chat']
    
    try:
        async with httpx.AsyncClient(timeout=None) as client: # Timeout None for long streams
            async with client.stream("POST", stream_endpoint, json=payload) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    logger.error(f"Error from server: {response.status_code} - {error_content.decode()}")
                    print(f"\nError: Received status {response.status_code} from server.", flush=True)
                    print(f"Details: {error_content.decode()}", flush=True)
                    yield {"event": "error", "data": f"Server error: {response.status_code}"} # Yield error event
                    return

                # Check for new session ID in response headers
                new_session_id = response.headers.get("X-Session-ID")
                if new_session_id and new_session_id != current_session_id:
                    current_session_id = new_session_id
                    logger.info(f"Session ID updated: {current_session_id}")

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

    except httpx.ConnectError as e:
        logger.error(f"Connection error: Could not connect to the server at {stream_endpoint}. Details: {e}")
        print(f"\nError: Could not connect to the server. Please ensure it's running.", flush=True)
        yield {"event": "error", "data": "Connection error"}
    except Exception as e:
        logger.error(f"An unexpected error occurred while streaming: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        yield {"event": "error", "data": f"Unexpected streaming error: {e}"}

async def clear_conversation_memory():
    """Clear the current conversation memory."""
    global current_session_id
    
    if not current_session_id:
        print("No active conversation to clear.")
        return
    
    try:
        clear_endpoint = f"{api_client.base_url}/chat/memory/{current_session_id}"
        async with httpx.AsyncClient() as client:
            response = await client.delete(clear_endpoint)
            if response.status_code == 200:
                print(f"✅ Conversation memory cleared.")
                current_session_id = None  # Reset session
            else:
                print(f"⚠️  Failed to clear conversation memory: {response.status_code}")
    except Exception as e:
        logger.error(f"Error clearing conversation memory: {e}")
        print(f"⚠️  Error clearing conversation memory: {e}")

async def show_conversation_memory():
    """Show the current conversation history."""
    global current_session_id
    
    if not current_session_id:
        print("No active conversation.")
        return
    
    try:
        memory_endpoint = f"{api_client.base_url}/chat/memory/{current_session_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(memory_endpoint)
            if response.status_code == 200:
                data = response.json()
                print(f"\n📝 Conversation History (Session: {current_session_id[:8]}...)")
                print(f"Total turns: {data['total_turns']}")
                
                if data['turns']:
                    for i, turn in enumerate(data['turns'], 1):
                        print(f"\n--- Turn {i} ({turn['query_type']}) ---")
                        print(f"You: {turn['user_query']}")
                        print(f"AI: {turn['ai_response'][:100]}..." if len(turn['ai_response']) > 100 else f"AI: {turn['ai_response']}")
                else:
                    print("No conversation history yet.")
            else:
                print(f"⚠️  Failed to retrieve conversation memory: {response.status_code}")
    except Exception as e:
        logger.error(f"Error retrieving conversation memory: {e}")
        print(f"⚠️  Error retrieving conversation memory: {e}")

async def main():
    user_prompts.show_app_header()

    ready_to_chat = await check_and_manage_index()
    if not ready_to_chat:
        print("Exiting application as indexing prerequisite not met or user chose to exit.")
        return

    user_prompts.show_section_separator()
    user_prompts.show_chat_start()
    
    # Show conversation commands
    print("\n💡 Conversation Commands:")
    print("  /clear    - Clear conversation memory")
    print("  /history  - Show conversation history") 
    print("  /new      - Start a new conversation")
    print("  exit/quit - Exit the application\n")

    while True:
        try:
            user_query = await asyncio.to_thread(input, "You: ")
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except EOFError:
            print("\nExiting chat due to EOF...")
            break

        # Handle conversation commands
        if user_query.lower() in ["/clear"]:
            await clear_conversation_memory()
            continue
        elif user_query.lower() in ["/history"]:
            await show_conversation_memory()
            continue
        elif user_query.lower() in ["/new"]:
            await clear_conversation_memory()
            current_session_id = None  # Will create new session on next query
            print("✨ Started a new conversation.")
            continue
        elif user_query.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        elif not user_query.strip():
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

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user (KeyboardInterrupt).")
        print("\nApplication exited.")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}") 