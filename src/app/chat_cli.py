# CLI script for the QA chat interface.
import logging
import sys
from pathlib import Path

# Ensure the project root is in PYTHONPATH when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use 'src.' prefix for local imports as per current working setup
from src.core.qa_service import initialize_chat_engine, stream_chat_response
from src.config import settings # For initial log level
from src.core.indexing_service import configure_llama_index_globals, get_active_settings # To ensure LlamaSettings are set

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

# Global variable to hold the chat engine instance for the session
# This ensures the engine (and its memory) persists between queries in a single run of the CLI.
chat_engine_session = None

def main():
    global chat_engine_session
    logger.info("===============================================")
    logger.info(" RAG Pipeline Chat CLI Started (Streaming) ")
    logger.info("===============================================")

    # Ensure LlamaIndex globals (like embed_model and potentially a default LLM)
    # are configured based on the active settings before initializing chat engine.
    # This is important especially if this CLI is the first LlamaIndex interaction point.
    active_app_settings = get_active_settings()
    configure_llama_index_globals(active_app_settings)
    logger.info(f"Chat CLI using LLM (from LlamaSettings): {LlamaSettings.llm.model if LlamaSettings.llm and hasattr(LlamaSettings.llm, 'model') else LlamaSettings.llm}")
    logger.info(f"Chat CLI using Embedding Model (from LlamaSettings): {LlamaSettings.embed_model}")

    if chat_engine_session is None:
        logger.info("No active chat engine session. Initializing...")
        chat_engine_session = initialize_chat_engine()

    if not chat_engine_session:
        logger.error("Failed to initialize chat engine. Exiting.")
        print("Error: Could not initialize the chat engine. Please check logs and ensure the index exists.")
        return

    print("Chat interface initialized (streaming responses). Type your questions or 'exit'/'quit' to end.")
    
    try:
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                logger.info("User requested to exit chat session.")
                print("Exiting chat. Goodbye!")
                break
            
            if not query.strip():
                continue

            print("AI: ", end="", flush=True) # Start the AI response line
            full_response_for_log = []
            for token in stream_chat_response(query, chat_engine_session):
                print(token, end="", flush=True)
                full_response_for_log.append(token)
            print() # Add a newline after the full streamed response
            logger.info(f"Streamed AI response: {''.join(full_response_for_log)}")

    except KeyboardInterrupt:
        logger.info("Chat interrupted by user (KeyboardInterrupt).")
        print("\nChat session ended.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the chat loop: {e}", exc_info=True)
        print("An unexpected error occurred. Exiting chat.")
    finally:
        logger.info("-----------------------------------------------")
        logger.info(" Chat CLI Session Ended ")
        logger.info("-----------------------------------------------")

if __name__ == "__main__":
    # Need LlamaSettings for the logger line above main()
    from llama_index.core import Settings as LlamaSettings 
    main() 