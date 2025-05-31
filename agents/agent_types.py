"""
Agent Type Definitions and Function Signatures

This module defines the type aliases and function signatures that form the core
contracts of the agent framework. These types ensure consistency and enable
static type checking across the multi-agent system.

Key Components:
- Agent function signatures as callable types
- Tool function interfaces 
- Type unions for flexible agent composition
- Generic types for extensibility
"""

from typing import Callable, Awaitable, Union, TypeVar, Generic, Protocol, runtime_checkable
from typing_extensions import TypeAlias

from .agent_models import AgentInput, AgentOutput, AgentResult, ToolResult


# Core type variables for generic agent operations
AgentInputType = TypeVar('AgentInputType', bound=AgentInput)
AgentOutputType = TypeVar('AgentOutputType', bound=AgentOutput)
ToolResultType = TypeVar('ToolResultType', bound=ToolResult)


# Primary agent function signatures
AgentFunction: TypeAlias = Callable[[AgentInput], Awaitable[AgentResult]]
"""
Type alias for standard agent functions.

All agents in the framework must conform to this signature:
- Input: AgentInput containing query, context, and metadata
- Output: Awaitable[AgentResult] with response and execution details

This ensures consistent interfaces across different agent types while
maintaining async operation for scalability.
"""

MetaAgentFunction: TypeAlias = Callable[[AgentInput], Awaitable[AgentResult]]
"""
Type alias for the meta-agent orchestrator function.

The meta-agent uses the same signature as other agents but has special
responsibilities for tool selection, sub-agent coordination, and workflow
orchestration.
"""

ToolFunction: TypeAlias = Callable[[str, dict], Awaitable[ToolResult]]
"""
Type alias for tool functions that can be called by agents.

Tools have a simplified interface:
- Input: query string and parameters dictionary
- Output: Awaitable[ToolResult] with tool-specific results

This design allows tools to be easily wrapped and standardized while
maintaining flexibility for different tool types.
"""


@runtime_checkable
class Agent(Protocol):
    """
    Protocol defining the interface that all agents must implement.
    
    This protocol ensures that agents can be used interchangeably while
    providing a clear contract for agent behavior.
    """
    
    async def execute(self, agent_input: AgentInput) -> AgentResult:
        """
        Execute the agent with the given input.
        
        Args:
            agent_input: Standardized input containing query and context
            
        Returns:
            AgentResult containing the agent's response and metadata
        """
        ...
    
    @property
    def agent_role(self) -> str:
        """Return the role/type of this agent."""
        ...
    
    @property
    def available_tools(self) -> list[str]:
        """Return list of tools this agent can use."""
        ...


@runtime_checkable
class Tool(Protocol):
    """
    Protocol defining the interface that all tools must implement.
    
    Tools provide specific capabilities that agents can use to accomplish
    their tasks, such as RAG, calculation, translation, etc.
    """
    
    async def execute(self, query: str, parameters: dict) -> ToolResult:
        """
        Execute the tool with the given query and parameters.
        
        Args:
            query: The query or task for the tool
            parameters: Tool-specific configuration and parameters
            
        Returns:
            ToolResult containing the tool's output and metadata
        """
        ...
    
    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        ...
    
    @property
    def description(self) -> str:
        """Return a description of what this tool does."""
        ...
    
    @property
    def parameter_schema(self) -> dict:
        """Return the JSON schema for tool parameters."""
        ...


# Union types for agent composition
AnyAgent: TypeAlias = Union[AgentFunction, Agent]
"""Union type allowing both function-based and class-based agents."""

AnyTool: TypeAlias = Union[ToolFunction, Tool] 
"""Union type allowing both function-based and class-based tools."""


# Generic types for extensible agent framework
class AgentRegistry(Generic[AgentInputType, AgentOutputType]):
    """
    Generic registry for managing different types of agents.
    
    This class provides a type-safe way to register and retrieve agents
    while maintaining flexibility for different agent implementations.
    """
    
    def __init__(self):
        self._agents: dict[str, AnyAgent] = {}
    
    def register_agent(self, name: str, agent: AnyAgent) -> None:
        """Register an agent with the given name."""
        self._agents[name] = agent
    
    def get_agent(self, name: str) -> AnyAgent | None:
        """Retrieve an agent by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> list[str]:
        """Return list of registered agent names."""
        return list(self._agents.keys())


class ToolRegistry(Generic[ToolResultType]):
    """
    Generic registry for managing different types of tools.
    
    This class provides a type-safe way to register and retrieve tools
    while maintaining flexibility for different tool implementations.
    """
    
    def __init__(self):
        self._tools: dict[str, AnyTool] = {}
    
    def register_tool(self, name: str, tool: AnyTool) -> None:
        """Register a tool with the given name."""
        self._tools[name] = tool
    
    def get_tool(self, name: str) -> AnyTool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())


# Type aliases for common agent patterns
RAGAgentFunction: TypeAlias = Callable[[AgentInput], Awaitable[AgentResult]]
CalculatorAgentFunction: TypeAlias = Callable[[AgentInput], Awaitable[AgentResult]]
TranslatorAgentFunction: TypeAlias = Callable[[AgentInput], Awaitable[AgentResult]]

# Type aliases for common tool patterns  
RAGToolFunction: TypeAlias = Callable[[str, dict], Awaitable[ToolResult]]
CalculatorToolFunction: TypeAlias = Callable[[str, dict], Awaitable[ToolResult]]
TranslatorToolFunction: TypeAlias = Callable[[str, dict], Awaitable[ToolResult]] 