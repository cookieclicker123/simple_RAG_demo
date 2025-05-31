"""
Tool Data Models for Standardized Tool Interface

This module defines the Pydantic models that ensure type safety and consistency
across all tools in the agent framework. These models provide contracts for
tool inputs, outputs, and metadata.

Key Design Principles:
- Consistent interface across all tool types
- Rich parameter validation and schema definition
- Comprehensive metadata for observability
- Integration with agent framework data models
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum

from src.models import DocumentCitation


class ToolParameterType(str, Enum):
    """Enumeration of supported tool parameter types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """
    Definition of a tool parameter including type, validation, and documentation.
    
    This model allows tools to define their parameter schema in a standardized way,
    enabling automatic validation and documentation generation.
    """
    name: str = Field(..., description="Parameter name")
    type: ToolParameterType = Field(..., description="Parameter data type")
    description: str = Field(..., description="Human-readable description of the parameter")
    required: bool = Field(True, description="Whether this parameter is required")
    default_value: Optional[Any] = Field(None, description="Default value if parameter is optional")
    allowed_values: Optional[List[Any]] = Field(None, description="List of allowed values (enum-style)")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value for numeric types")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value for numeric types")
    min_length: Optional[int] = Field(None, description="Minimum length for string/array types")
    max_length: Optional[int] = Field(None, description="Maximum length for string/array types")
    pattern: Optional[str] = Field(None, description="Regex pattern for string validation")
    
    class Config:
        use_enum_values = True


class ToolSchema(BaseModel):
    """
    Complete schema definition for a tool including metadata and parameters.
    
    This model serves as the contract that defines what a tool does and how
    to interact with it, enabling automatic tool discovery and usage.
    """
    tool_name: str = Field(..., description="Unique identifier for the tool")
    display_name: str = Field(..., description="Human-readable name for the tool")
    description: str = Field(..., description="Detailed description of tool functionality")
    version: str = Field("1.0.0", description="Tool version for compatibility tracking")
    category: str = Field("general", description="Tool category (e.g., 'rag', 'math', 'translation')")
    parameters: List[ToolParameter] = Field(default_factory=list, description="List of tool parameters")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for tool output")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Example usage scenarios")
    tags: List[str] = Field(default_factory=list, description="Tags for tool discovery and categorization")
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for tool parameters."""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in self.parameters:
            param_schema = {"type": param.type.value, "description": param.description}
            
            if param.allowed_values:
                param_schema["enum"] = param.allowed_values
            if param.min_value is not None:
                param_schema["minimum"] = param.min_value
            if param.max_value is not None:
                param_schema["maximum"] = param.max_value
            if param.min_length is not None:
                param_schema["minLength"] = param.min_length
            if param.max_length is not None:
                param_schema["maxLength"] = param.max_length
            if param.pattern:
                param_schema["pattern"] = param.pattern
            if param.default_value is not None:
                param_schema["default"] = param.default_value
                
            schema["properties"][param.name] = param_schema
            
            if param.required:
                schema["required"].append(param.name)
        
        return schema


class ToolMetadata(BaseModel):
    """
    Metadata container for tool execution tracking and observability.
    
    Provides essential information for debugging, monitoring, and optimization
    of tool usage within the agent framework.
    """
    tool_name: str = Field(..., description="Name of the tool being executed")
    execution_id: str = Field(..., description="Unique identifier for this execution")
    agent_id: Optional[str] = Field(None, description="ID of the agent calling this tool")
    session_id: Optional[str] = Field(None, description="Session identifier if applicable")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context information")
    
    class Config:
        frozen = True  # Immutable for thread safety


class ToolInput(BaseModel):
    """
    Standardized input structure for tool execution.
    
    This model ensures consistent interfaces across all tools while providing
    flexibility for tool-specific parameters.
    """
    query: str = Field(..., description="Primary query or task for the tool")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific parameters")
    metadata: ToolMetadata = Field(..., description="Execution metadata and context")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context from agent")
    
    class Config:
        arbitrary_types_allowed = True


class ToolOutput(BaseModel):
    """
    Standardized output structure for tool execution results.
    
    Provides consistent response format enabling easy integration with
    the agent framework and other tools.
    """
    tool_name: str = Field(..., description="Name of the tool that produced this output")
    success: bool = Field(..., description="Whether tool execution was successful")
    result_data: Any = Field(..., description="Primary result data from tool execution")
    error_message: Optional[str] = Field(None, description="Error details if execution failed")
    execution_time_ms: float = Field(..., description="Tool execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific metadata")
    citations: List[DocumentCitation] = Field(default_factory=list, description="Supporting citations if applicable")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in result quality")
    next_actions: List[str] = Field(default_factory=list, description="Suggested follow-up actions")
    
    class Config:
        arbitrary_types_allowed = True


# Specialized models for specific tool types
class RAGToolInput(BaseModel):
    """Specialized input model for RAG tool operations."""
    query: str = Field(..., description="Search query for document retrieval")
    max_documents: int = Field(10, ge=1, le=100, description="Maximum number of documents to retrieve")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    include_citations: bool = Field(True, description="Whether to include document citations")
    rerank_results: bool = Field(True, description="Whether to rerank results with cross-encoder")
    conversation_context: Optional[str] = Field(None, description="Conversation history for context")


class RAGToolOutput(BaseModel):
    """Specialized output model for RAG tool results."""
    answer: str = Field(..., description="Generated answer based on retrieved documents")
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved document metadata")
    citations: List[DocumentCitation] = Field(default_factory=list, description="Document citations")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in answer quality")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Retrieval process metadata")
    
    class Config:
        arbitrary_types_allowed = True


class CalculatorToolInput(BaseModel):
    """Specialized input model for calculator tool operations."""
    expression: str = Field(..., description="Mathematical expression to evaluate")
    precision: int = Field(10, ge=1, le=50, description="Decimal precision for results")
    return_steps: bool = Field(True, description="Whether to return step-by-step solution")
    allow_complex: bool = Field(False, description="Whether to allow complex number operations")


class CalculatorToolOutput(BaseModel):
    """Specialized output model for calculator tool results."""
    result: Union[int, float, str] = Field(..., description="Calculated result")
    steps: List[str] = Field(default_factory=list, description="Step-by-step solution if requested")
    expression_simplified: str = Field(..., description="Simplified version of input expression")
    result_type: str = Field(..., description="Type of result (integer, float, complex, etc.)")


class TranslatorToolInput(BaseModel):
    """Specialized input model for translator tool operations."""
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code (e.g., 'en', 'es', 'zh')")
    source_language: Optional[str] = Field(None, description="Source language code (auto-detect if None)")
    preserve_formatting: bool = Field(True, description="Whether to preserve text formatting")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence for translation")


class TranslatorToolOutput(BaseModel):
    """Specialized output model for translator tool results."""
    translated_text: str = Field(..., description="Translated text")
    source_language_detected: str = Field(..., description="Detected source language")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Translation confidence")
    alternative_translations: List[str] = Field(default_factory=list, description="Alternative translation options") 