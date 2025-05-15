# src/agents/sub_agents/rag_agent.py: RAG Sub-Agent implementation.

import logging
from typing import cast

from google.adk.agents import LlmAgent
from google.adk.models.llms import LlmOptions
from google.adk.orchestration import инструк션_sender # This should be InstructionSender if it exists, or similar
# ^^^ ADK import for sending instructions or invoking other agents/tools might be different.
# For an LlmAgent using tools, the invocation is often implicit via the LLM call.

# Use 'from src...' imports based on current working setup
from src.config import settings
from src.agents.tools.rag_tool import rag_document_tool, RagQueryInput, RagToolOutput

# Pydantic models for this agent's expected input/output might be useful later for clarity
# from pydantic import BaseModel, Field
# class RagAgentTask(BaseModel):
#     question_for_documents: str = Field(description="The specific question to be answered using local documents.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

# LLM Options for this specific agent - using the OpenAI model configured for RAG
# We need to map our settings to ADK's LlmOptions. ADK typically integrates with
# google.generativeai or vertexai. If using OpenAI via ADK, it might need a custom Llm client for ADK
# or ADK might have an OpenAI Llm client wrapper.
# For now, let's assume ADK LlmAgent can be pointed to an OpenAI model if ADK has an OpenAI LLM integration.
# If ADK primarily uses Gemini, and this agent's TOOL uses OpenAI, the LlmAgent itself might just need to be good at *formulating the call to the tool*.

# The LlmAgent for RAG doesn't necessarily need to use the same LLM as its tool, 
# especially if the tool is self-contained (which our rag_tool tries to be with its LLM override).
# Let's configure this agent to use a Gemini model for its own reasoning if needed,
# and rely on the tool to use its designated (OpenAI) LLM.
rag_agent_llm_options = LlmOptions(
    model_name=settings.gemini_flash_model_name, # Use Gemini Flash for this agent's own reasoning
    temperature=settings.temperature,
    max_output_tokens=settings.max_tokens # This agent primarily returns the tool's output
)
# Note: API key for Gemini would be handled by ADK's Llm client setup, often via environment variables
# or explicit client initialization if ADK requires it.

def create_rag_agent() -> LlmAgent:
    """Creates and returns the RAG Sub-Agent."""
    logger.info(f"Creating RAG Agent with LLM: {rag_agent_llm_options.model_name}")
    
    rag_agent = LlmAgent(
        llm_options=rag_agent_llm_options,
        tools=[rag_document_tool],
        system_instruction=(
            "You are an expert assistant specialized in finding information within a local document knowledge base. "
            "Your task is to answer questions based *only* on the information provided by the 'QueryLocalDocuments' tool. "
            "If the tool provides an answer, return that answer directly and concisely. "
            "If the tool indicates it cannot answer or returns an error, state that the information could not be found in the local documents. "
            "Do not use any external knowledge or make assumptions beyond what the tool provides."
            "If the user's request is not a question for the documents, you can say that you are specialized for document queries."
        )
        # ADK might have different ways to specify prompts or few-shot examples.
        # For a tool-using LlmAgent, the LLM's ability to correctly call the tool based on description is key.
    )
    logger.info(f"RAG Agent created. Equipped with tool: {rag_document_tool.name}")
    return rag_agent

# For testing this agent individually
if __name__ == "__main__":
    import asyncio
    # Ensure path for local imports if running directly
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Ensure LlamaIndex globals used by the RAG tool are configured
    # This is especially important for the embedding model.
    from src.core.indexing_service import configure_llama_index_globals as core_configure_globals, get_active_settings
    core_configure_globals(get_active_settings()) 

    logger.info("--- RAG Agent Test --- ")
    # Note: This test assumes OPENAI_API_KEY is set for the RAG tool internal LLM
    # and GEMINI_API_KEY is set for the RAG Agent's own LLM (if ADK uses it automatically via env).
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here_if_not_in_env":
        logger.warning("OpenAI API key not set. RAG tool may fail.")
    if not settings.gemini_api_key:
        logger.warning("Gemini API key not set. RAG Agent LLM (Gemini Flash) may fail.")

    test_rag_agent = create_rag_agent()

    async def run_agent_test():
        test_query = "Tell me about the mechanical data of speedmate"
        print(f"Testing RAG Agent with query: '{test_query}'")
        
        # ADK agents are typically called with an `invoke` or `run` method.
        # The input format might depend on how the LlmAgent is expected to receive tasks.
        # If it's meant to directly use the tool based on a natural language query that matches
        # the tool description, the query itself can be the input.
        try:
            # ADK LlmAgent.invoke typically takes a string or a dict matching inferred/specified input schema.
            # If the agent is expected to use its tools based on the prompt, a direct string might work.
            response = await test_rag_agent.invoke(test_query) 
            
            # The response from an ADK LlmAgent is usually an LlmResponse object or similar.
            # We need to extract the content.
            if hasattr(response, 'content'):
                print(f"RAG Agent Response: {response.content}")
            elif isinstance(response, dict) and 'output' in response: # Some ADK examples show dict output
                print(f"RAG Agent Response (dict): {response['output']}")
            else:
                print(f"RAG Agent Response (raw): {response}")

        except Exception as e:
            print(f"Error during RAG Agent test: {e}", exc_info=True)

    asyncio.run(run_agent_test())
    logger.info("--- RAG Agent Test Complete ---") 