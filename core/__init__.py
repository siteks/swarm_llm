# Core components adapted from backend_reference
# Note: Requires litellm to be installed

from .tools import ToolRegistry

# LLM client imports require litellm
try:
    from .llm_client import LLMClient, UsageStats, TurnResult
    __all__ = [
        "LLMClient",
        "UsageStats",
        "TurnResult",
        "ToolRegistry",
    ]
except ImportError as e:
    # LiteLLM not installed - partial functionality
    LLMClient = None
    UsageStats = None
    TurnResult = None
    __all__ = ["ToolRegistry"]
