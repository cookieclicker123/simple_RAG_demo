"""
Agent Framework for Conversational RAG System

This package provides a multi-agent architecture built on type-safe foundations,
designed for extensibility and stateless operation. The framework is inspired by
modern agent development practices while maintaining our existing DRY principles.

Core Components:
- Meta Agent: Primary orchestrator that coordinates tool usage
- Tool System: Standardized interface for RAG and future capabilities
- Type Safety: Pydantic models and type aliases for bulletproof interfaces
- Memory Integration: Built on existing conversation memory infrastructure
"""

from .agent_models import (
    AgentInput,
    AgentOutput, 
    AgentContext,
    AgentMetadata,
    AgentResult
)

from .agent_types import (
    AgentFunction,
    MetaAgentFunction,
    ToolFunction
)

from .meta_agent import meta_agent

__all__ = [
    "AgentInput",
    "AgentOutput", 
    "AgentContext",
    "AgentMetadata",
    "AgentResult",
    "AgentFunction",
    "MetaAgentFunction", 
    "ToolFunction",
    "meta_agent"
] 