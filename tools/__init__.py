"""
Tool Framework for Multi-Agent Architecture

This package provides a standardized interface for tools that can be used by agents
in the multi-agent system. Tools encapsulate specific capabilities like RAG,
calculation, translation, etc.

Key Components:
- Tool Interface: Standardized protocol for all tools
- RAG Tool: Existing RAG functionality wrapped as a tool
- Tool Registry: Discovery and management of available tools
- Type Safety: Pydantic models for tool inputs/outputs

Design Principles:
- Stateless operations following existing patterns
- Async interfaces for scalability
- Rich metadata for observability
- Extensible for future tool types
"""

from .tool_models import (
    ToolInput,
    ToolOutput,
    ToolParameter,
    ToolSchema,
    ToolMetadata
)

from .rag_tool import rag_tool
from .tool_registry import tool_registry

__all__ = [
    "ToolInput",
    "ToolOutput", 
    "ToolParameter",
    "ToolSchema",
    "ToolMetadata",
    "rag_tool",
    "tool_registry"
] 