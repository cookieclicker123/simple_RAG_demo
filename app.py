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
                    yield None # Signal error or end
                    return

                # Process the Server-Sent Events (SSE) stream
                buffer = ""
                async for line in response.aiter_lines():
                    if not line.strip(): # Empty lines separate events in SSE
                        if buffer.startswith("data:"):
                            data_json_str = buffer[len("data:"):].strip()
                            try:
                                data = json.loads(data_json_str)
                                if data.get("event") == "stream_end":
                                    logger.info("Stream ended by server signal.")
                                    yield None # Signal end of stream
                                    break
                                if data.get("error"):
                                    error_message = data.get("token", "Unknown error from stream.")
                                    logger.error(f"Received error in stream: {error_message}")
                                    print(f"\nStream error: {error_message}", flush=True)
                                    yield f"ERROR_TOKEN: {error_message}"
                                elif "token" in data:
                                    yield data["token"]
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode JSON from stream: {data_json_str}")
                        buffer = ""
                    else:
                        buffer += line + "\n"

    except httpx.ConnectError as e:
        logger.error(f"Connection error: Could not connect to the server at {STREAM_CHAT_ENDPOINT}. Details: {e}")
        print(f"\nError: Could not connect to the server at {STREAM_CHAT_ENDPOINT}. Please ensure it's running.", flush=True)
        yield None
    except Exception as e:
        logger.error(f"An unexpected error occurred while streaming: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        yield None

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
        # Keep track of tokens for logging if needed, but don't log the full response to console here.
        # full_response_for_internal_log = [] 
        has_printed_error_token = False

        async for token in get_streaming_response(user_query):
            if token is None:
                if not print("[No response or connection failed]", flush=True) and not has_printed_error_token:
                    pass # Avoids double printing error messages if already handled
                break
            
            if token.startswith("ERROR_TOKEN:"):
                has_printed_error_token = True
            else:
                print(token, end="", flush=True)
                # full_response_for_internal_log.append(token)
        
        print() 
        logger.info(f"User query: {user_query}")
        # If you still want to log the full response to a file or a different handler (not console),
        # you could re-enable collecting tokens and log them using a logger not configured for console INFO.
        # For now, this line is removed to prevent console duplication:
        # if full_response_for_internal_log:
        #     logger.info(f"Streamed AI response: {''.join(full_response_for_internal_log)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user (KeyboardInterrupt).")
        print("\nApplication exited.")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}") 