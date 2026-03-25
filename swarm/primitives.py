"""
Swarm tool primitives for agent coordination.

These tools enable agents to:
- Communicate via messaging (inbox/outbox)
- Share state via persistent storage
- Discover other agents

All tools are async and designed to work with the swarm runtime.
"""

from typing import Any, Dict, List, Optional
from contextvars import ContextVar
import json

from .storage import InMemoryStorage
from .messaging import MessagingSystem
from .telemetry import TelemetryCollector

# Async-safe context for current agent ID (isolated per coroutine chain)
_current_agent_context: ContextVar[Optional[str]] = ContextVar('current_agent', default=None)

# Turn-scoped character budget tracker
_turn_char_budget: ContextVar[Dict[str, Any]] = ContextVar(
    'turn_char_budget',
    default={'used': 0, 'limit': None}
)


def reset_turn_budget(limit: Optional[int]) -> None:
    """Reset budget at start of agent turn."""
    _turn_char_budget.set({'used': 0, 'limit': limit})


def get_turn_budget() -> Dict[str, Any]:
    """Get current budget state."""
    return _turn_char_budget.get()


class SwarmPrimitives:
    """
    Container for swarm coordination primitives.

    This class manages the shared state and messaging systems,
    and provides tool handlers that can be registered with the ToolRegistry.
    """

    def __init__(
        self,
        storage: InMemoryStorage,
        messaging: MessagingSystem,
        telemetry: TelemetryCollector
    ):
        self.storage = storage
        self.messaging = messaging
        self.telemetry = telemetry

        # Agent registry - maintained by runtime
        self._agents: Dict[str, Dict[str, Any]] = {}

    @property
    def _current_agent_id(self) -> Optional[str]:
        """Get current agent ID from async-safe context."""
        return _current_agent_context.get()

    def set_current_agent(self, agent_id: Optional[str]) -> None:
        """Set the current agent context for tool execution (async-safe)."""
        _current_agent_context.set(agent_id)
        self.storage.set_current_agent(agent_id)

    def register_agent(self, agent_id: str, info: Dict[str, Any]) -> None:
        """Register an agent in the directory."""
        self._agents[agent_id] = info

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the directory."""
        self._agents.pop(agent_id, None)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-format tool definitions for all swarm primitives."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_inbox",
                    "description": "Read messages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unread_only": {
                                "type": "boolean",
                                "default": True
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {
                                "type": "string",
                            },
                            "content": {
                                "type": "string",
                            }
                        },
                        "required": ["to", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_storage",
                    "description": "Read value",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                            }
                        },
                        "required": ["key"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_storage",
                    "description": "Write value, overwrites any existing value",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                            },
                            "value": {
                                "type": "string",
                            }
                        },
                        "required": ["key", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "append_storage",
                    "description": "Append value",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                            },
                            "value": {
                                "type": "string",
                            }
                        },
                        "required": ["key", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_agents",
                    "description": "Discover agents you can collaborate with",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_storage_keys",
                    "description": "List keys",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_my_id",
                    "description": "Get your ID",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

    # =========================================================================
    # Tool Handler Implementations
    # =========================================================================

    async def read_inbox(self, unread_only: bool = True) -> Dict[str, Any]:
        """Read messages from the current agent's inbox."""
        if not self._current_agent_id:
            return {
                "success": False,
                "error": "No agent context set",
                "messages": []
            }

        messages = await self.messaging.read_inbox(
            agent_id=self._current_agent_id,
            unread_only=unread_only
        )

        return {
            "success": True,
            "count": len(messages),
            "messages": [
                {
                    "from": m.from_agent,
                    "content": m.content,
                    "cycle": m.cycle
                }
                for m in messages
            ]
        }

    async def send_message(self, to: str, content: str) -> Dict[str, Any]:
        """Send a message to another agent."""
        if not to or not content:
            return {
                "success": False,
                "error": "Malformed call, check arguments"
            }

        if not self._current_agent_id:
            return {
                "success": False,
                "error": "No agent context set"
            }

        # Check if target agent exists
        if to not in self._agents:
            return {
                "success": False,
                "error": f"Agent '{to}' not found. Use list_agents() to see available agents."
            }

        # Check character budget
        chars = len(content)
        budget = _turn_char_budget.get()
        if budget['limit'] is not None:
            remaining = budget['limit'] - budget['used']
            if chars > remaining:
                return {
                    "success": False,
                    "error": f"Insufficient character budget: {remaining} remaining"
                }
            budget['used'] += chars

        message = await self.messaging.send(
            from_agent=self._current_agent_id,
            to_agent=to,
            content=content
        )

        # Build response with budget info if applicable
        result = {
            "success": True,
            "message_id": message.id,
        }
        if budget['limit'] is not None:
            remaining = budget['limit'] - budget['used']
            result["note"] = f"Message sent to {to}. Character budget: {chars} sent, {remaining} remaining"
            result["budget_chars_used"] = chars
            result["budget_chars_remaining"] = remaining
        else:
            result["note"] = f"Message sent to {to}"

        return result

    async def read_storage(self, key: str) -> Dict[str, Any]:
        """Read a value from shared storage."""
        if not key:
            return {
                "success": False,
                "error": "Malformed call, check arguments"
            }

        value = await self.storage.read(key)

        return {
            "success": True,
            "key": key,
            "value": value,
            "exists": value is not None
        }

    async def write_storage(self, key: str, value: Any) -> Dict[str, Any]:
        """Write a value to shared storage."""
        if not key:
            return {
                "success": False,
                "error": "Malformed call, check arguments"
            }

        # Prevent writing to internal keys
        if key.startswith("_"):
            return {
                "success": False,
                "error": "Cannot write to keys starting with '_' (reserved for system use)"
            }

        # Check character budget
        chars = len(str(value))
        budget = _turn_char_budget.get()
        if budget['limit'] is not None:
            remaining = budget['limit'] - budget['used']
            if chars > remaining:
                return {
                    "success": False,
                    "error": f"Insufficient character budget: {remaining} remaining"
                }
            budget['used'] += chars

        await self.storage.write(key, value)

        # Build response with budget info if applicable
        result = {
            "success": True,
            "key": key,
        }
        if budget['limit'] is not None:
            remaining = budget['limit'] - budget['used']
            result["note"] = f"Value written to storage. Character budget: {chars} written, {remaining} remaining"
            result["budget_chars_used"] = chars
            result["budget_chars_remaining"] = remaining
        else:
            result["note"] = "Value written to storage"

        return result

    async def append_storage(self, key: str, value: Any) -> Dict[str, Any]:
        """Append a value to a list in shared storage."""
        if not key:
            return {
                "success": False,
                "error": "Malformed call, check arguments"
            }

        # Prevent writing to internal keys
        if key.startswith("_"):
            return {
                "success": False,
                "error": "Cannot write to keys starting with '_' (reserved for system use)"
            }

        # Check character budget
        chars = len(str(value))
        budget = _turn_char_budget.get()
        if budget['limit'] is not None:
            remaining = budget['limit'] - budget['used']
            if chars > remaining:
                return {
                    "success": False,
                    "error": f"Insufficient character budget: {remaining} remaining"
                }
            budget['used'] += chars

        await self.storage.append(key, value)

        # Build response with budget info if applicable
        result = {
            "success": True,
            "key": key,
        }
        if budget['limit'] is not None:
            remaining = budget['limit'] - budget['used']
            result["note"] = f"Value appended to list. Character budget: {chars} written, {remaining} remaining"
            result["budget_chars_used"] = chars
            result["budget_chars_remaining"] = remaining
        else:
            result["note"] = "Value appended to list"

        return result

    async def list_agents(self) -> Dict[str, Any]:
        """List all active agents in the swarm."""
        agents_info = []
        for agent_id, info in self._agents.items():
            agents_info.append({
                "id": agent_id,
                "status": info.get("status", "unknown"),
                "cycles_active": info.get("cycles_active", 0)
            })

        return {
            "success": True,
            "count": len(agents_info),
            "agents": agents_info
        }

    async def list_storage_keys(self) -> Dict[str, Any]:
        """List all storage keys, optionally filtered by prefix."""
        keys = await self.storage.list_keys(None)

        # Filter out internal keys
        public_keys = [k for k in keys if not k.startswith("_")]

        return {
            "success": True,
            "count": len(public_keys),
            "keys": public_keys
        }

    async def get_my_id(self) -> Dict[str, Any]:
        """Return the current agent's ID."""
        return {
            "success": True,
            "agent_id": self._current_agent_id
        }

    def get_tool_handlers(self) -> Dict[str, callable]:
        """Get a mapping of tool names to their handler methods."""
        return {
            "read_inbox": self.read_inbox,
            "send_message": self.send_message,
            "read_storage": self.read_storage,
            "write_storage": self.write_storage,
            "append_storage": self.append_storage,
            "list_agents": self.list_agents,
            "list_storage_keys": self.list_storage_keys,
            "get_my_id": self.get_my_id
        }
