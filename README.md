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

## Project Structure

The project follows a structured layout to promote modularity and maintainability:

```
simple_RAG_demo/
├── .venv/                               # Virtual environment (managed by uv)
├── data/                                # Directory for user's input documents (e.g., PDFs)
├── local_db/                            # Directory for storing the FAISS vector index
├── .env                                 # Local environment variables (Create from .env.example)
├── .env.example                         # Example environment variables file
├── .gitignore                           # Specifies intentionally untracked files
├── pyproject.toml                       # Project metadata and dependencies (PEP 621)
├── README.md                            # This file
├── spec.txt                             # Detailed project specification for the agent system
├── src/                                 # Source code
│   ├── __init__.py                      # Makes src a package (simple_rag_pipeline)
│   ├── models.py                        # Pydantic data models
│   ├── config.py                        # Configuration (API keys, paths, model names)
│   ├── core/                            # Core business logic
│   │   ├── __init__.py
│   │   ├── indexing_service.py          # Document parsing, chunking, embedding, storing
│   │   └── qa_service.py                # Conversational QA, memory, LLM interaction
│   ├── utils/                           # Utility functions
│   │   ├── __init__.py
│   │   └── file_handlers.py             # Document loading, preprocessing
│   │   └── vector_store_handlers.py     # Vector store creation, loading, saving
│   ├── agents/                            # Agent-related code (ADK based)
│   │   ├── __init__.py
│   │   ├── meta_agent.py                # Orchestrator Meta-Agent
│   │   ├── sub_agents/                  # Specialized sub-agents
│   │   │   ├── __init__.py
│   │   │   ├── rag_agent.py             # RAG sub-agent
│   │   │   ├── search_planner_agent.py  # Sub-agent for planning web searches
│   │   │   └── web_search_agent.py      # Web searching sub-agent
│   │   └── tools/                       # Custom ADK tools
│   │       ├── __init__.py
│   │       └── rag_tool.py              # ADK tool wrapping the RAG pipeline
│   ├── app/                             # Application entry points (CLIs)
│   │   ├── __init__.py
│   │   ├── indexer_cli.py               # CLI script for indexing documents
│   │   ├── chat_cli.py                  # CLI script for RAG QA chat interface
│   │   └── agent_cli.py                 # CLI for interacting with the Meta-Agent
└── tests/                               # Test suite
    ├── __init__.py
    ├── fixtures/                        # Test fixtures
    │   ├── __init__.py
    │   └── .gitkeep
    ├── unit/                            # Unit tests
    │   ├── __init__.py
    │   └── .gitkeep
    └── integration/                     # Integration tests
        ├── __init__.py
        └── .gitkeep                     # Added .gitkeep for integration tests
```

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

### 2. Chatting with Your Documents

*   Once documents are indexed, start the chat interface:
    ```bash
    python src/app/chat_cli.py
    # Or, to run as a module:
    # python -m simple_rag_pipeline.app.chat_cli
    ```
*   Ask questions related to the content of your indexed documents. Type `exit` or `quit` to end the chat session.

### 3. Interacting with the Research Project Assistant (Agent System)

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