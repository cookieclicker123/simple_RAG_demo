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
        self._core_tools_registered = False
        
        logger.info("Tool registry initialized - core tools will be registered on first access")
    
    def _ensure_core_tools_registered(self) -> None:
        """Ensure core tools are registered (deferred to avoid circular imports)."""
        if self._core_tools_registered:
            return
            
        try:
            # Import here to avoid circular imports
            from tools.rag_tool import rag_tool, get_rag_tool_schema
            from tools.translator_tool import translator_tool, get_translator_tool_schema
            
            # Register RAG tool
            self.register_tool("rag_tool", rag_tool, get_rag_tool_schema())
            
            # Register translator tool
            self.register_tool("translator_tool", translator_tool, get_translator_tool_schema())
            
            self._core_tools_registered = True
            logger.info("Core tools registered: rag_tool, translator_tool")
            
        except Exception as e:
            logger.error(f"Failed to register core tools: {e}", exc_info=True)
    
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
        self._ensure_core_tools_registered()
        return self._tools.get(name)
    
    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """
        Get a tool schema by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool schema if found, None otherwise
        """
        self._ensure_core_tools_registered()
        return self._schemas.get(name)
    
    def list_tools(self) -> List[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool names
        """
        self._ensure_core_tools_registered()
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """
        Get list of all tool categories.
        
        Returns:
            List of category names
        """
        self._ensure_core_tools_registered()
        return list(self._categories.keys())
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """
        Get list of tools in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of tool names in the category
        """
        self._ensure_core_tools_registered()
        return self._categories.get(category, [])
    
    def search_tools(self, query: str) -> List[str]:
        """
        Search for tools by name, description, or tags.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching tool names
        """
        self._ensure_core_tools_registered()
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
        tool_name: str, 
        query: str, 
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            query: Query/description for the tool execution
            parameters: Parameters to pass to the tool
            
        Returns:
            ToolResult containing execution results
            
        Raises:
            ValueError: If tool is not found or parameters are invalid
        """
        self._ensure_core_tools_registered()
        
        tool_func = self._tools.get(tool_name)
        if not tool_func:
            available_tools = list(self._tools.keys())
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
        
        # Get schema for validation
        schema = self._schemas.get(tool_name)
        if schema:
            # Validate parameters against schema
            validation_errors = self.validate_parameters(tool_name, parameters)
            if validation_errors:
                raise ValueError(f"Parameter validation failed for tool '{tool_name}': {validation_errors}")
        
        # Execute the tool
        logger.debug(f"Executing tool '{tool_name}' with query: {query}")
        result = await tool_func(query, parameters)
        logger.debug(f"Tool '{tool_name}' execution completed with status: {result.status}")
        
        return result
    
    def validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters against the tool's schema.
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate
            
        Returns:
            Dictionary of validation errors (empty if valid)
        """
        self._ensure_core_tools_registered()
        
        schema = self._schemas.get(tool_name)
        if not schema:
            return {"schema": f"No schema found for tool '{tool_name}'"}
        
        errors = {}
        
        # Check required parameters
        required_params = [p.name for p in schema.parameters if p.required]
        for param in required_params:
            if param not in parameters:
                errors[param] = f"Required parameter '{param}' is missing"
        
        # Validate each parameter
        for param in schema.parameters:
            if param.name in parameters:
                value = parameters[param.name]
                param_errors = self._validate_parameter(param, value)
                if param_errors:
                    errors[param.name] = param_errors
        
        return errors
    
    def _validate_parameter(self, param: Any, value: Any) -> Optional[str]:
        """
        Validate a single parameter value against its schema.
        
        Args:
            param: Parameter definition from schema
            value: Value to validate
            
        Returns:
            Error message if invalid, None if valid
        """
        # Basic type validation
        if hasattr(param, 'allowed_values') and param.allowed_values:
            if value not in param.allowed_values:
                return f"Value must be one of {param.allowed_values}, got '{value}'"
        
        if hasattr(param, 'min_length') and param.min_length is not None:
            if isinstance(value, str) and len(value) < param.min_length:
                return f"String too short, minimum length is {param.min_length}"
        
        if hasattr(param, 'max_length') and param.max_length is not None:
            if isinstance(value, str) and len(value) > param.max_length:
                return f"String too long, maximum length is {param.max_length}"
        
        return None
    
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