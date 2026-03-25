"""
Tool registry for swarm agents.

Provides a registry of available tools that agents can use.
Adapted from backend_reference/tools.py with swarm-specific extensions.
"""

import logging
import inspect
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class ToolRegistry:
    """
    Registry of available tools for agent function calling.

    Tools are registered with:
    - A name
    - A description
    - Parameter schema (OpenAI function format)
    - An async handler function
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ) -> None:
        """
        Register a tool with its definition and handler.

        Args:
            name: Unique tool name
            description: Human-readable description
            parameters: JSON Schema for parameters
            handler: Async function to execute the tool
        """
        self.tools[name] = {
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            },
            "handler": handler
        }
        logger.debug(f"Registered tool: {name}")

    def register_from_definition(
        self,
        definition: Dict[str, Any],
        handler: Callable
    ) -> None:
        """
        Register a tool from an OpenAI-format definition.

        Args:
            definition: Tool definition in OpenAI format
            handler: Async function to execute the tool
        """
        name = definition["function"]["name"]
        self.tools[name] = {
            "definition": definition,
            "handler": handler
        }
        logger.debug(f"Registered tool from definition: {name}")

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get list of tool definitions in OpenAI format."""
        return [tool["definition"] for tool in self.tools.values()]

    def get_tool_handler(self, name: str) -> Optional[Callable]:
        """Get the execution handler for a tool."""
        if name in self.tools:
            return self.tools[name]["handler"]
        return None

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self.tools

    def list_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self.tools.keys())

    async def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        **context
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments.

        Args:
            name: Tool name to execute
            arguments: Arguments to pass to the tool
            **context: Additional context (e.g., agent_id, workspace_path)

        Returns:
            Dict with success status, result or error, and metadata
        """
        handler = self.get_tool_handler(name)
        if not handler:
            return {
                "success": False,
                "error": f"Tool '{name}' not found",
                "tool": name
            }

        try:
            logger.debug(f"Executing tool: {name} with args: {arguments}")

            # Check which parameters the handler accepts
            sig = inspect.signature(handler)
            handler_kwargs = {}

            # Pass through any context parameters the handler accepts
            for param_name in sig.parameters:
                if param_name in context:
                    handler_kwargs[param_name] = context[param_name]

            # Call handler with arguments and context
            result = await handler(**arguments, **handler_kwargs)

            logger.debug(f"Tool {name} completed successfully")

            # If result is already a dict with success key, return as-is
            if isinstance(result, dict) and "success" in result:
                result["tool"] = name
                return result

            # Otherwise wrap in standard format
            return {
                "success": True,
                "result": result,
                "tool": name
            }

        except Exception as e:
            logger.error(f"Tool {name} execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": name
            }


def create_swarm_registry(primitives) -> ToolRegistry:
    """
    Create a ToolRegistry pre-populated with swarm primitives.

    Args:
        primitives: SwarmPrimitives instance

    Returns:
        ToolRegistry with all swarm tools registered
    """
    registry = ToolRegistry()

    # Get tool definitions and handlers from primitives
    definitions = primitives.get_tool_definitions()
    handlers = primitives.get_tool_handlers()

    # Register each tool
    for definition in definitions:
        name = definition["function"]["name"]
        if name in handlers:
            registry.register_from_definition(definition, handlers[name])
        else:
            logger.warning(f"No handler found for tool: {name}")

    return registry
