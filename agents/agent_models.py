"""
Agent Data Models for Type-Safe Multi-Agent Architecture

This module defines the core Pydantic models that ensure type safety and structure
across the agent framework. These models serve as contracts between agents, tools,
and the orchestration layer.

Key Design Principles:
- Immutable data structures where possible
- Rich metadata for observability and debugging
- Extensible for future agent types and capabilities
- Integration with existing conversation memory system
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum

from src.models import ConversationMemory, DocumentCitation


class AgentRole(str, Enum):
    """
    Enumeration of agent roles within the multi-agent system.
    
    Each role defines the agent's primary responsibility and capabilities:
    - META: Orchestrates other agents and handles tool selection
    - RAG: Specialized for retrieval-augmented generation tasks
    - TRANSLATOR: Language translation and multi-language support
    - CALCULATOR: Mathematical computation and analysis
    - WEB_SEARCH: External information retrieval
    - CUSTOM: User-defined specialized agents
    """
    META = "meta"
    RAG = "rag" 
    TRANSLATOR = "translator"
    CALCULATOR = "calculator"
    WEB_SEARCH = "web_search"
    CUSTOM = "custom"


class ToolExecutionStatus(str, Enum):
    """Status tracking for tool execution within agent workflows."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentMetadata(BaseModel):
    """
    Metadata container for agent execution context and observability.
    
    Provides essential information for debugging, monitoring, and coordination
    between agents in complex multi-agent workflows.
    """
    agent_id: str = Field(..., description="Unique identifier for this agent instance")
    agent_role: AgentRole = Field(..., description="Role/type of the agent")
    session_id: str = Field(..., description="Conversation session identifier")
    execution_id: str = Field(..., description="Unique identifier for this execution")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_agent_id: Optional[str] = Field(None, description="ID of parent agent if this is a sub-agent")
    depth: int = Field(0, description="Nesting depth in agent hierarchy (0 for meta-agent)")
    
    class Config:
        frozen = True  # Immutable for thread safety


class AgentContext(BaseModel):
    """
    Execution context shared across agent operations.
    
    Contains conversation state, memory, and configuration needed for
    agents to operate within the broader system context.
    """
    conversation_memory: ConversationMemory = Field(..., description="Current conversation state and history")
    user_query: str = Field(..., description="Original user query that initiated this agent chain")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query after context processing")
    available_tools: List[str] = Field(default_factory=list, description="List of available tool names")
    execution_config: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")
    
    class Config:
        arbitrary_types_allowed = True  # Allow ConversationMemory


class ToolResult(BaseModel):
    """
    Standardized result structure from tool execution.
    
    Provides consistent interface for all tool outputs, enabling
    seamless integration across different tool types.
    """
    tool_name: str = Field(..., description="Name of the tool that was executed")
    status: ToolExecutionStatus = Field(..., description="Execution status")
    result_data: Any = Field(..., description="Primary result data from tool execution")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional tool-specific metadata")
    error_message: Optional[str] = Field(None, description="Error details if execution failed")
    execution_time_ms: Optional[float] = Field(None, description="Tool execution time in milliseconds")
    citations: List[DocumentCitation] = Field(default_factory=list, description="Document citations if applicable")
    
    class Config:
        arbitrary_types_allowed = True  # Allow DocumentCitation


class AgentInput(BaseModel):
    """
    Standardized input structure for all agent functions.
    
    This model serves as the primary interface contract for agent execution,
    ensuring consistent data flow and type safety across the framework.
    """
    query: str = Field(..., description="User query or task description for the agent")
    context: AgentContext = Field(..., description="Execution context including memory and configuration")
    metadata: AgentMetadata = Field(..., description="Agent metadata for tracking and coordination")
    tool_results: List[ToolResult] = Field(default_factory=list, description="Results from previously executed tools")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific parameters")
    
    class Config:
        arbitrary_types_allowed = True


class AgentOutput(BaseModel):
    """
    Standardized output structure for all agent functions.
    
    Provides consistent response format enabling easy composition
    and coordination between multiple agents.
    """
    response_text: str = Field(..., description="Primary text response from the agent")
    tool_calls_made: List[str] = Field(default_factory=list, description="Names of tools that were called")
    tool_results: List[ToolResult] = Field(default_factory=list, description="Results from tool executions")
    citations: List[DocumentCitation] = Field(default_factory=list, description="Document citations supporting the response")
    sub_agent_calls: List[str] = Field(default_factory=list, description="Sub-agents that were invoked")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in the response quality")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution statistics and debugging info")
    next_suggested_actions: List[str] = Field(default_factory=list, description="Suggested follow-up actions or queries")
    
    class Config:
        arbitrary_types_allowed = True


class AgentResult(BaseModel):
    """
    Complete result package from agent execution including both output and metadata.
    
    This model wraps the AgentOutput with additional tracking information
    for comprehensive observability and system monitoring.
    """
    input_metadata: AgentMetadata = Field(..., description="Metadata from the input request")
    output: AgentOutput = Field(..., description="Agent's response and results")
    execution_duration_ms: float = Field(..., description="Total execution time in milliseconds") 
    success: bool = Field(..., description="Whether execution completed successfully")
    error_details: Optional[str] = Field(None, description="Error information if execution failed")
    memory_updated: bool = Field(False, description="Whether conversation memory was updated")
    
    class Config:
        arbitrary_types_allowed = True


# Example specialized models for specific agent types
class RAGAgentParams(BaseModel):
    """Parameters specific to RAG agent operations."""
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_documents: int = Field(10, ge=1, le=100)
    include_citations: bool = Field(True)
    rerank_results: bool = Field(True)


class CalculatorAgentParams(BaseModel):
    """Parameters specific to calculator agent operations."""
    precision: int = Field(10, ge=1, le=50)
    allow_complex_numbers: bool = Field(False)
    return_step_by_step: bool = Field(True)


class TranslatorAgentParams(BaseModel):
    """Parameters specific to translator agent operations."""
    source_language: Optional[str] = Field(None)
    target_language: str = Field(...)
    preserve_formatting: bool = Field(True)
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0) 