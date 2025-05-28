# Simple RAG QA Pipeline

## Overview

This project implements a simple, command-line based Retrieval Augmented Generation (RAG) pipeline. It allows users to index their documents (e.g., PDFs) and then ask questions about the content of these documents through a chat interface. The system maintains conversation history for contextual understanding.

This project serves as a foundational step towards building a more complex agent framework, emphasizing clean code, professional structure, and the use of modern Python tools.

For a detailed plan of the agent-based system we are evolving towards, please see `spec.txt`.

## Core Technologies

*   **Orchestration & Frameworks**: `LangChain`, `LlamaIndex`
*   **Vector Store**: `FAISS` (local)
*   **Embedding Model**: e.g., `BAAI/bge-small-en-v1.5` (via HuggingFace)
*   **LLM**: e.g., `OpenAI gpt-4o-mini`
*   **Data Models**: `Pydantic`
*   **Configuration**: `Pydantic-Settings` (from `.env` files)
*   **Environment & Package Management**: `uv`

## Setup

This project uses `uv` for package and environment management.

1.  **Install `uv`**:
    If you don't have `uv` installed, follow the official instructions:
    ```bash
    # On macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Clone the repository**:
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd simple_RAG_demo
    ```

3.  **Create and activate virtual environment**:
    ```bash
    uv venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate    # On Windows
    ```

4.  **Configure Environment Variables**:
    Copy the example environment file and fill in your details:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file to add your `OPENAI_API_KEY` and any other configurations you wish to manage via environment variables. The application uses `python-dotenv` and `pydantic-settings` to load these values.

5.  **Install dependencies**:
    With `uv` and your virtual environment activated, install the project dependencies (which includes development tools):
    ```bash
    uv pip install -e ".[dev]"
    ```
    This command installs the package in editable mode (`-e`) along with its main dependencies and the optional `[dev]` dependencies specified in `pyproject.toml`.

## Usage

### 1. Indexing Documents

*   Place your documents (e.g., PDF files) into the `data/` directory. Ensure this directory exists, or it will be created by the application on first run if `config.py` is executed directly (e.g. during tests or initial setup).
*   Run the indexing script:
    ```bash
    python src/app/indexer_cli.py 
    # Or, to run as a module (often more robust for imports):
    # python -m simple_rag_pipeline.app.indexer_cli
    ```
    This will process the documents, create embeddings, and store them in the `local_db/faiss_index` directory (path configurable in `src/config.py`).

### 2. Chatting with Your Documents (CLI)

*   Once documents are indexed, start the chat interface:
    ```bash
    python src/app/chat_cli.py
    # Or, to run as a module:
    # python -m simple_rag_pipeline.app.chat_cli
    ```
*   Ask questions related to the content of your indexed documents. Type `exit` or `quit` to end the chat session.

### 3. Running the FastAPI Server (API)

*   Ensure dependencies are installed (including FastAPI and Uvicorn):
    ```bash
    uv pip install -e ".[dev]"
    ```
*   Start the FastAPI server using Uvicorn. From the project root directory (`simple_RAG_demo/`):
    ```bash
    uvicorn src.server.web_app:app --host 0.0.0.0 --port 8000 --reload
    ```
    Alternatively, you can run the `web_app.py` script directly (though `--reload` is more convenient for development):
    ```bash
    python src/server/web_app.py
    ```
*   Once the server is running, you can access the API documentation (Swagger UI) at `http://localhost:8000/api/docs` and ReDoc at `http://localhost:8000/api/redoc`.
*   The streaming chat endpoint will be available at `POST http://localhost:8000/api/chat/stream`.

### 4. Interacting with the Research Project Assistant (Agent System CLI)

*   (Coming Soon) Once the agent system is implemented, you will run:
    ```bash
    python src/app/agent_cli.py
    # Or, to run as a module:
    # python -m simple_rag_pipeline.app.agent_cli
    ```

## Development

*   **Linting & Formatting**: This project uses `ruff` for linting and formatting. You can run it from the project root:
    ```bash
    ruff check .
    ruff format .
    ```
*   **Tests**: (Details about running tests with `pytest` will be added here once tests are written.)

## Future Enhancements

*   Integration into a larger agent framework.
*   Support for more document types.
*   Advanced RAG techniques (e.g., hybrid search, re-ranking).
*   Web interface using FastAPI.
*   Containerization with Docker.
*   Migration to custom logic, reducing external dependencies where beneficial. 