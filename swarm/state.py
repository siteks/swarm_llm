"""State serialization/deserialization for swarm resume capability.

Provides data structures and methods for saving and restoring complete
swarm state to enable continuing runs from where they left off.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# State format version for compatibility checking
STATE_VERSION = "2.0"


@dataclass
class AgentState:
    """Serializable state of a single agent."""
    id: str
    model: str
    system_prompt: str  # Full prompt, not truncated
    tokens_used: int
    cycles_active: int
    last_activity_cycle: int
    status: str  # "active", "paused", "terminated"
    spawn_cycle: int  # Cycle when agent was spawned

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tokens_used": self.tokens_used,
            "cycles_active": self.cycles_active,
            "last_activity_cycle": self.last_activity_cycle,
            "status": self.status,
            "spawn_cycle": self.spawn_cycle
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            model=data["model"],
            system_prompt=data["system_prompt"],
            tokens_used=data["tokens_used"],
            cycles_active=data["cycles_active"],
            last_activity_cycle=data["last_activity_cycle"],
            status=data["status"],
            spawn_cycle=data["spawn_cycle"]
        )


@dataclass
class SwarmState:
    """
    Complete swarm state for save/restore.

    Contains all information needed to resume a swarm run:
    - Metadata (version)
    - Cycle counters
    - Complete storage (including _inbox_* keys)
    - Agent states
    - Telemetry history
    """
    version: str
    current_cycle: int
    messaging_cycle: int
    storage: Dict[str, Any]  # Complete storage including _inbox_* keys
    agents: List[AgentState]
    telemetry_events: List[Dict[str, Any]]
    cycle_summaries: List[Dict[str, Any]]
    entropy_rot_levels: Dict[str, float] = field(default_factory=dict)  # Per-key rot levels for marginal decay
    can_resume: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "version": self.version,
            "can_resume": self.can_resume,
            "current_cycle": self.current_cycle,
            "messaging_cycle": self.messaging_cycle,
            "storage": self.storage,
            "agents": [a.to_dict() for a in self.agents],
            "entropy_rot_levels": self.entropy_rot_levels,
            "telemetry": {
                "events": self.telemetry_events,
                "cycle_summaries": self.cycle_summaries
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmState":
        """Deserialize from dictionary (e.g., from JSON file)."""
        return cls(
            version=data["version"],
            can_resume=data.get("can_resume", True),
            current_cycle=data["current_cycle"],
            messaging_cycle=data["messaging_cycle"],
            storage=data["storage"],
            agents=[AgentState.from_dict(a) for a in data["agents"]],
            telemetry_events=data.get("telemetry", {}).get("events", []),
            cycle_summaries=data.get("telemetry", {}).get("cycle_summaries", []),
            entropy_rot_levels=data.get("entropy_rot_levels", {})
        )

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "SwarmState":
        """
        Parse SwarmState from JSON export file data.

        Args:
            json_data: The parsed JSON from a swarm export file

        Returns:
            SwarmState instance

        Raises:
            ValueError: If swarm_state is missing or invalid
        """
        if "swarm_state" not in json_data:
            raise ValueError("JSON file does not contain swarm_state for resuming")

        state_data = json_data["swarm_state"]

        # Validate version
        version = state_data.get("version", "unknown")
        if version != STATE_VERSION:
            raise ValueError(
                f"Incompatible state version: {version} (expected {STATE_VERSION})"
            )

        if not state_data.get("can_resume", True):
            raise ValueError("This swarm state is marked as non-resumable")

        return cls.from_dict(state_data)
