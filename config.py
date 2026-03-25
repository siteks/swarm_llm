"""
Configuration for the swarm system.

Environment variables:
- LLM_MODEL: Default model for agents (default: claude-sonnet-4-5-20250929)
- LLM_MAX_TOKENS: Max tokens per completion (default: 4096)
- MODEL_*: Model nicknames (e.g., MODEL_CLAUDE=claude-sonnet-4-5-20250929)
- SWARM_MAX_CYCLES: Default max cycles (default: 50)
- SWARM_MAX_CONCURRENT: Max concurrent agent calls (default: 10)
- SWARM_MAX_TOOL_ITERATIONS: Max tool iterations per turn (default: 10)
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


def load_model_nicknames() -> Dict[str, str]:
    """Load MODEL_* env vars into nickname->model mapping."""
    nicknames = {}
    for key, value in os.environ.items():
        if key.startswith("MODEL_") and value:
            nickname = key[6:].lower()  # MODEL_CLAUDE -> claude
            nicknames[nickname] = value
    return nicknames


def resolve_model(nickname_or_model: str) -> str:
    """Resolve a model nickname to full model ID.

    Args:
        nickname_or_model: Either a nickname (e.g., 'claude') or full model ID

    Returns:
        Full model ID
    """
    nicknames = load_model_nicknames()
    if nickname_or_model.lower() in nicknames:
        return nicknames[nickname_or_model.lower()]
    # If not a nickname, assume it's a full model ID
    return nickname_or_model


def get_default_model() -> str:
    """Get the default model from LLM_MODEL env var."""
    return os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929")


@dataclass
class SwarmConfig:
    """Configuration for swarm execution."""

    # LLM settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096

    # Execution settings
    max_cycles: int = 50
    max_concurrent_agents: int = 10
    max_tool_iterations: int = 10

    # Budget settings
    char_budget_per_turn: Optional[int] = None  # From SWARM_CHAR_BUDGET env var

    # Entropy settings (sigmoid decay)
    entropy_enabled: bool = True
    entropy_tipping_point: float = 15.0  # Cycles before decay accelerates
    entropy_steepness: float = 0.5  # Decay curve sharpness (higher = sharper)

    # Output settings
    telemetry_json_path: str = "swarm_telemetry.json"
    telemetry_graphml_path: str = "swarm_messages.graphml"

    @classmethod
    def from_env(cls) -> "SwarmConfig":
        """Create config from environment variables."""
        return cls(
            model=os.getenv("LLM_MODEL", cls.model),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(cls.max_tokens))),
            max_cycles=int(os.getenv("SWARM_MAX_CYCLES", str(cls.max_cycles))),
            max_concurrent_agents=int(os.getenv("SWARM_MAX_CONCURRENT", str(cls.max_concurrent_agents))),
            max_tool_iterations=int(os.getenv("SWARM_MAX_TOOL_ITERATIONS", str(cls.max_tool_iterations))),
            char_budget_per_turn=int(os.getenv("SWARM_CHAR_BUDGET")) if os.getenv("SWARM_CHAR_BUDGET") else None,
            entropy_enabled=os.getenv("SWARM_ENTROPY_ENABLED", "true").lower() == "true",
            entropy_tipping_point=float(os.getenv("SWARM_ENTROPY_TIPPING_POINT", str(cls.entropy_tipping_point))),
            entropy_steepness=float(os.getenv("SWARM_ENTROPY_STEEPNESS", str(cls.entropy_steepness)))
        )


# Example agent prompts
AGENT_PROMPTS = {
    "minimal": """You are a curious creative agent. Explore and play!""",
}
