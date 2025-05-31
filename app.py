import asyncio
import httpx
import json
import sys
import logging
from typing import Optional

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set httpx logger level to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

API_BASE_URL = "http://localhost:8000/api"
STREAM_CHAT_ENDPOINT = f"{API_BASE_URL}/chat/stream"
INDEX_STATUS_ENDPOINT = f"{API_BASE_URL}/index/status"
TRIGGER_INDEX_ENDPOINT = f"{API_BASE_URL}/index/documents"

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
    """Checks index status and prompts user to index if needed. Returns True if ready to chat."""
    print("Checking document index status...")
    status_data = await get_index_status_from_server()

    if not status_data:
        print("Could not retrieve index status from the server. Please ensure the server is running.")
        return False # Cannot proceed

    print(f"Server: {status_data.get('message', 'No status message.')}")

    if not status_data.get('exists'):
        if status_data.get('document_count_in_data_folder', 0) == 0:
            print("No documents found in the server's 'data' folder. Please add PDF documents there and try again.")
            return False # Cannot proceed
        
        while True:
            choice = input("No index found. Would you like to index documents from the server's 'data' folder now? (yes/no): ").strip().lower()
            if choice == 'yes':
                if await trigger_server_indexing():
                    print("Please allow some time for indexing to complete on the server before proceeding.")
                    # Ideally, we'd poll /index/status here, but for simplicity, we'll just proceed after a delay or user prompt.
                    input("Press Enter to continue once you believe indexing is complete...")
                    # Re-check status after indexing attempt
                    # status_data = await get_index_status_from_server()
                    # if status_data and status_data.get('exists'):
                    #     print(f"Server: {status_data.get('message', 'No status message.')}")
                    #     return True
                    # else:
                    #     print("Indexing may have failed or is still in progress. Please check server logs.")
                    #     return False
                    return True # Assume success for now, user will see if chat fails
                else:
                    print("Failed to initiate indexing process on the server.")
                    return False # Cannot proceed
            elif choice == 'no':
                print("Cannot proceed without an index. Exiting.")
                return False
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
    else: # Index exists
        if status_data.get('document_count_in_data_folder', 0) == 0:
            print("Warning: Index exists, but no documents currently in the server's 'data' folder. Chatting will be based on the existing index content.")
        
        reindex_choice = input("An index already exists. Would you like to re-index with the current documents in the server's 'data' folder? (yes/no): ").strip().lower()
        if reindex_choice == 'yes':
            if await trigger_server_indexing():
                print("Please allow some time for re-indexing to complete on the server.")
                input("Press Enter to continue once you believe re-indexing is complete...")
                return True # Assume success for now
            else:
                print("Failed to initiate re-indexing process on the server.")
                # Choice to proceed with old index or not
                proceed_anyway = input("Would you like to chat with the existing index anyway? (yes/no): ").strip().lower()
                return proceed_anyway == 'yes'
        # If 'no' to re-indexing, or if re-indexing failed and user chose not to proceed
    
    return True # Ready to chat (either with existing index or after indexing was triggered)

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