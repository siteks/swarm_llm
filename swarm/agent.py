"""Agent model and lifecycle states for the swarm system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class AgentStatus(Enum):
    """Agent lifecycle states."""
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class ToolCall:
    """Represents a tool call made by an agent."""
    id: str
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_openai_format(cls, tool_call: Dict) -> "ToolCall":
        """Create from OpenAI tool_call format."""
        import json
        return cls(
            id=tool_call.get("id", str(uuid.uuid4())),
            name=tool_call["function"]["name"],
            arguments=json.loads(tool_call["function"]["arguments"])
        )


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any = None
    error: Optional[str] = None


@dataclass
class Agent:
    """
    Represents an autonomous agent in the swarm.

    Agents have minimal per-cycle context - they start fresh each cycle
    with only their system prompt. Long-term state must be externalized
    to shared storage and messaging primitives.
    """
    id: str
    system_prompt: str
    model: str  # LLM model ID for this agent
    spawn_cycle: int = 0  # Cycle when agent was spawned

    # Resource tracking
    tokens_used: int = 0
    cycles_active: int = 0
    last_activity_cycle: int = 0

    # Status
    status: AgentStatus = AgentStatus.ACTIVE

    # Per-turn state (reset each cycle by runtime)
    pending_tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)

    # Turn metrics (reset each cycle)
    turn_tokens: int = 0
    turn_tool_calls: int = 0

    @classmethod
    def create(
        cls,
        system_prompt: str,
        model: str,
        agent_id: Optional[str] = None
    ) -> "Agent":
        """Factory method to create a new agent."""
        return cls(
            id=agent_id or f"agent_{uuid.uuid4().hex[:8]}",
            system_prompt=system_prompt,
            model=model
        )

    def reset_turn_state(self) -> None:
        """Reset per-turn state at the start of a new cycle."""
        self.pending_tool_calls = []
        self.tool_results = []
        self.turn_tokens = 0
        self.turn_tool_calls = 0

    def record_turn_usage(self, tokens: int, tool_calls: int) -> None:
        """Record token and tool usage for this turn."""
        self.turn_tokens = tokens
        self.turn_tool_calls = tool_calls
        self.tokens_used += tokens

    def is_active(self) -> bool:
        """Check if agent is active and can participate in cycles."""
        return self.status == AgentStatus.ACTIVE

    def pause(self) -> None:
        """Pause the agent (skip in future cycles)."""
        self.status = AgentStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused agent."""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.ACTIVE

    def terminate(self) -> None:
        """Terminate the agent (cannot be resumed)."""
        self.status = AgentStatus.TERMINATED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state for telemetry/export."""
        return {
            "id": self.id,
            "model": self.model,
            "system_prompt": self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt,
            "spawn_cycle": self.spawn_cycle,
            "tokens_used": self.tokens_used,
            "cycles_active": self.cycles_active,
            "last_activity_cycle": self.last_activity_cycle,
            "status": self.status.value,
        }
