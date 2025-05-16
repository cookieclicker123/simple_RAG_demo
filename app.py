import asyncio
import httpx
import json
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set httpx logger level to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

API_BASE_URL = "http://localhost:8000/api"
STREAM_CHAT_ENDPOINT = f"{API_BASE_URL}/chat/stream"

async def get_streaming_response(query: str):
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

        async for event_data in get_streaming_response(user_query):
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
            # without an explicit error event from get_streaming_response or a proper stream_end.
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
        # Full AI response logging removed to avoid console duplication

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user (KeyboardInterrupt).")
        print("\nApplication exited.")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}") 