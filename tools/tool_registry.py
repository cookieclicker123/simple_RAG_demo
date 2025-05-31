"""
Tool Registry for Multi-Agent Framework

This module provides a centralized registry for managing and discovering tools
available to agents. The registry supports automatic tool discovery, schema
validation, and dynamic tool loading.

Key Features:
- Centralized tool management
- Schema validation and documentation
- Tool discovery and categorization
- Runtime tool registration
- Type-safe tool access
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from agents.agent_models import ToolResult

from tools.tool_models import ToolSchema
from tools.rag_tool import rag_tool, get_rag_tool_schema

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for managing tools available to agents.
    
    This class provides a thread-safe way to register, discover, and execute
    tools within the agent framework. It maintains tool schemas, validates
    parameters, and provides introspection capabilities.
    """
    
    def __init__(self):
        """Initialize the tool registry with core tools."""
        self._tools: Dict[str, Callable[[str, Dict[str, Any]], Awaitable[ToolResult]]] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        self._categories: Dict[str, List[str]] = {}
        
        # Register core tools
        self._register_core_tools()
        
        logger.info("Tool registry initialized with core tools")
    
    def _register_core_tools(self) -> None:
        """Register the core tools that are always available."""
        # Register RAG tool
        self.register_tool("rag_tool", rag_tool, get_rag_tool_schema())
        
        logger.info("Core tools registered: rag_tool")
    
    def register_tool(
        self, 
        name: str, 
        tool_function: Callable[[str, Dict[str, Any]], Awaitable[ToolResult]],
        schema: ToolSchema
    ) -> None:
        """
        Register a new tool with the registry.
        
        Args:
            name: Unique identifier for the tool
            tool_function: Async function implementing the tool
            schema: Tool schema with parameters and documentation
            
        Raises:
            ValueError: If tool name already exists or schema is invalid
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        
        if not schema.tool_name:
            raise ValueError("Tool schema must have a tool_name")
        
        if schema.tool_name != name:
            logger.warning(f"Tool name mismatch: registry='{name}', schema='{schema.tool_name}'")
        
        # Register the tool
        self._tools[name] = tool_function
        self._schemas[name] = schema
        
        # Update category index
        category = schema.category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        logger.info(f"Registered tool: {name} (category: {category})")
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            return False
        
        schema = self._schemas[name]
        category = schema.category
        
        # Remove from main registries
        del self._tools[name]
        del self._schemas[name]
        
        # Remove from category index
        if category in self._categories:
            self._categories[category] = [t for t in self._categories[category] if t != name]
            if not self._categories[category]:
                del self._categories[category]
        
        logger.info(f"Unregistered tool: {name}")
        return True
    
    def get_tool(self, name: str) -> Optional[Callable[[str, Dict[str, Any]], Awaitable[ToolResult]]]:
        """
        Get a tool function by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool function if found, None otherwise
        """
        return self._tools.get(name)
    
    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """
        Get a tool schema by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool schema if found, None otherwise
        """
        return self._schemas.get(name)
    
    def list_tools(self) -> List[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """
        Get list of all tool categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """
        Get list of tools in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of tool names in the category
        """
        return self._categories.get(category, [])
    
    def search_tools(self, query: str) -> List[str]:
        """
        Search for tools by name, description, or tags.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching tool names
        """
        query_lower = query.lower()
        matches = []
        
        for name, schema in self._schemas.items():
            # Check tool name
            if query_lower in name.lower():
                matches.append(name)
                continue
            
            # Check display name and description
            if (query_lower in schema.display_name.lower() or 
                query_lower in schema.description.lower()):
                matches.append(name)
                continue
            
            # Check tags
            if any(query_lower in tag.lower() for tag in schema.tags):
                matches.append(name)
                continue
        
        return matches
    
    async def execute_tool(
        self, 
        name: str, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute a tool by name with given parameters.
        
        Args:
            name: Name of the tool to execute
            query: Query string for the tool
            parameters: Tool-specific parameters
            
        Returns:
            ToolResult from tool execution
            
        Raises:
            ValueError: If tool not found
            Exception: If tool execution fails
        """
        tool_function = self.get_tool(name)
        if not tool_function:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        if parameters is None:
            parameters = {}
        
        logger.info(f"Executing tool: {name}")
        logger.debug(f"Query: {query}")
        logger.debug(f"Parameters: {parameters}")
        
        return await tool_function(query, parameters)
    
    def validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for a specific tool.
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate
            
        Returns:
            Validated parameters with defaults applied
            
        Raises:
            ValueError: If tool not found or parameters invalid
        """
        schema = self.get_schema(tool_name)
        if not schema:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        validated = {}
        
        # Check required parameters
        for param in schema.parameters:
            if param.required and param.name not in parameters:
                raise ValueError(f"Required parameter '{param.name}' missing for tool '{tool_name}'")
            
            value = parameters.get(param.name, param.default_value)
            
            if value is not None:
                # Type validation would go here
                # For now, just pass through
                validated[param.name] = value
        
        # Add any extra parameters (flexible approach)
        for key, value in parameters.items():
            if key not in validated:
                validated[key] = value
        
        return validated
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a tool.
        
        Args:
            name: Name of the tool
            
        Returns:
            Dictionary with tool information, None if not found
        """
        schema = self.get_schema(name)
        if not schema:
            return None
        
        return {
            "name": schema.tool_name,
            "display_name": schema.display_name,
            "description": schema.description,
            "version": schema.version,
            "category": schema.category,
            "parameter_count": len(schema.parameters),
            "required_parameters": [p.name for p in schema.parameters if p.required],
            "optional_parameters": [p.name for p in schema.parameters if not p.required],
            "tags": schema.tags,
            "parameter_schema": schema.get_parameter_schema()
        }
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tool registry.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_tools": len(self._tools),
            "categories": len(self._categories),
            "tools_by_category": {cat: len(tools) for cat, tools in self._categories.items()},
            "all_tools": list(self._tools.keys())
        }


# Global tool registry instance
tool_registry = ToolRegistry()


# Utility functions for common operations
async def execute_rag_query(query: str, **kwargs) -> ToolResult:
    """
    Convenience function to execute RAG queries.
    
    Args:
        query: The search query
        **kwargs: Additional parameters for RAG tool
        
    Returns:
        ToolResult from RAG execution
    """
    return await tool_registry.execute_tool("rag_tool", query, kwargs)


def get_available_tools() -> List[str]:
    """
    Get list of all available tool names.
    
    Returns:
        List of tool names
    """
    return tool_registry.list_tools()


def find_tools_for_task(task_description: str) -> List[str]:
    """
    Find tools suitable for a given task description.
    
    Args:
        task_description: Description of the task
        
    Returns:
        List of tool names that might be suitable
    """
    return tool_registry.search_tools(task_description) 