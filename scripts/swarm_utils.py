"""Utilities for analyzing swarm JSON artifacts."""

import gzip
import json
import math
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Iterator, Optional, Union
from collections import defaultdict


@dataclass
class SwarmRun:
    """Loaded swarm run with convenient accessors."""
    data: Dict[str, Any]

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SwarmRun":
        """Load a swarm JSON file."""
        with open(path) as f:
            return cls(json.load(f))

    @property
    def total_cycles(self) -> int:
        return self.data.get("total_cycles", 0)

    @property
    def total_events(self) -> int:
        return self.data.get("total_events", 0)

    @property
    def agents(self) -> List[Dict[str, Any]]:
        return self.data.get("swarm_state", {}).get("agents", [])

    @property
    def agent_ids(self) -> List[str]:
        return [a["id"] for a in self.agents]

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self.data.get("swarm_state", {}).get("telemetry", {}).get("events", [])

    @property
    def storage(self) -> Dict[str, Any]:
        return self.data.get("swarm_state", {}).get("storage", {})

    def events_by_type(self, event_type: str) -> Iterator[Dict[str, Any]]:
        """Iterate over events of a specific type."""
        for e in self.events:
            if e.get("event_type") == event_type:
                yield e

    def events_by_cycle(self, cycle: int) -> Iterator[Dict[str, Any]]:
        """Iterate over events in a specific cycle."""
        for e in self.events:
            if e.get("cycle") == cycle:
                yield e

    def events_by_agent(self, agent_id: str) -> Iterator[Dict[str, Any]]:
        """Iterate over events for a specific agent."""
        for e in self.events:
            if e.get("agent_id") == agent_id:
                yield e


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def agent_content_words_by_cycle(run: SwarmRun) -> Dict[int, Dict[str, int]]:
    """
    Count words of agent content per cycle per agent.

    Returns: {cycle: {agent_id: word_count}}
    """
    result = defaultdict(lambda: defaultdict(int))
    for e in run.events_by_type("agent_content"):
        cycle = e.get("cycle", 0)
        agent_id = e.get("agent_id", "unknown")
        content = e.get("data", {}).get("content", "")
        result[cycle][agent_id] += word_count(content)
    return dict(result)


def token_usage_by_cycle(run: SwarmRun) -> Dict[int, Dict[str, int]]:
    """
    Get token usage per cycle per agent from agent_turn_end events.

    Returns: {cycle: {agent_id: token_count}}
    """
    result = defaultdict(lambda: defaultdict(int))
    for e in run.events_by_type("agent_turn_end"):
        cycle = e.get("cycle", 0)
        agent_id = e.get("agent_id", "unknown")
        tokens = e.get("data", {}).get("tokens", 0)
        result[cycle][agent_id] += tokens
    return dict(result)


def tool_calls_from_turns_by_cycle(run: SwarmRun) -> Dict[int, Dict[str, int]]:
    """
    Get tool call counts per cycle per agent from agent_turn_end events.

    Returns: {cycle: {agent_id: tool_call_count}}
    """
    result = defaultdict(lambda: defaultdict(int))
    for e in run.events_by_type("agent_turn_end"):
        cycle = e.get("cycle", 0)
        agent_id = e.get("agent_id", "unknown")
        tool_calls = e.get("data", {}).get("tool_calls", 0)
        result[cycle][agent_id] += tool_calls
    return dict(result)


LLM_ERROR_PATTERN = re.compile(r'ServiceUnavailableError|MidStreamFallbackError|InternalServerError')


def llm_errors_by_cycle(run: SwarmRun) -> Dict[int, List[Dict[str, Any]]]:
    """
    Find LLM provider errors per cycle from swarm.runtime events.

    Returns: {cycle: [{"agent_id": str, "error": str}, ...]}
    """
    result = defaultdict(list)
    for e in run.events_by_type("tool_error"):
        data = e.get("data", {})
        error_msg = data.get("error", "")
        source = data.get("source", "")
        if source == "swarm.runtime" and LLM_ERROR_PATTERN.search(error_msg):
            cycle = e.get("cycle", 0)
            agent_id = e.get("agent_id", "unknown")
            result[cycle].append({"agent_id": agent_id, "error": error_msg})
    return dict(result)


def send_message_stats_by_cycle(run: SwarmRun) -> Dict[int, Dict[str, Dict[str, int]]]:
    """
    Count send_message operations and words per cycle per agent.

    Returns: {cycle: {agent_id: {"count": int, "words": int}}}
    """
    result = defaultdict(lambda: defaultdict(lambda: {"count": 0, "chars": 0, "words": 0}))
    for e in run.events_by_type("tool_call"):
        if e.get("data", {}).get("tool") == "send_message":
            if not e["data"]["result"]["success"]:
                continue
            cycle = e.get("cycle", 0)
            agent_id = e.get("agent_id", "unknown")
            content = e.get("data", {}).get("args", {}).get("content", "")
            result[cycle][agent_id]["count"] += 1
            result[cycle][agent_id]["chars"] += len(content)
            result[cycle][agent_id]["words"] += word_count(content)
    return dict(result)


def storage_write_stats_by_cycle(run: SwarmRun) -> Dict[int, Dict[str, Dict[str, int]]]:
    """
    Count write_storage and append_storage operations and words per cycle per agent.

    Returns: {cycle: {agent_id: {"count": int, "words": int}}}
    """
    result = defaultdict(lambda: defaultdict(lambda: {"count": 0, "chars":0, "words": 0}))
    for e in run.events_by_type("tool_call"):
        tool = e.get("data", {}).get("tool", "")
        if tool in ("write_storage", "append_storage"):
            if not e["data"]["result"]["success"]:
                continue
            cycle = e.get("cycle", 0)
            agent_id = e.get("agent_id", "unknown")
            value = e.get("data", {}).get("args", {}).get("value", "")
            if isinstance(value, str):
                text = value
            else:
                text = str(value)
            # print(len(text), text)
            result[cycle][agent_id]["count"] += 1
            result[cycle][agent_id]["chars"] += len(text)
            result[cycle][agent_id]["words"] += word_count(text)
    return dict(result)



def short_model_name(model: str) -> str:
    """Extract short provider name from model string."""
    model_lower = model.lower()
    if 'kimi' in model_lower:
        return 'kimi'
    elif 'claude' in model_lower:
        return 'claude'
    elif 'gemini' in model_lower:
        return 'gemini'
    elif 'gpt' in model_lower or 'openai' in model_lower:
        return 'openai'
    elif 'deepseek' in model_lower:
        return 'deepseek'
    return model.split('/')[0].split('-')[0]


def agent_combination(run: SwarmRun) -> str:
    """Get sorted model combination label, e.g., 'kimi-kimi' or 'gemini-kimi-kimi'."""
    models = sorted(short_model_name(a['model']) for a in run.agents)
    return '-'.join(models)

