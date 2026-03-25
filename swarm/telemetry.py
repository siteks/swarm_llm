"""Telemetry collection and export for observing swarm emergence."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from . import time_provider


class EventType(Enum):
    """Types of telemetry events."""
    # Cycle events
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"

    # Agent events
    AGENT_SPAWN = "agent_spawn"
    AGENT_TERMINATE = "agent_terminate"
    AGENT_TURN_START = "agent_turn_start"
    AGENT_TURN_END = "agent_turn_end"
    AGENT_THINKING = "agent_thinking"
    AGENT_CONTENT = "agent_content"

    # Tool events
    TOOL_CALL = "tool_call"
    TOOL_ERROR = "tool_error"

    # System events
    SWARM_START = "swarm_start"
    SWARM_STOP = "swarm_stop"
    SWARM_RESUME = "swarm_resume"
    STORAGE_ENTROPY = "storage_entropy"


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    event_type: EventType
    timestamp: datetime = field(default_factory=time_provider.now)
    cycle: int = 0
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for export."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "cycle": self.cycle,
            "agent_id": self.agent_id,
            "data": self.data
        }


@dataclass
class CycleSummary:
    """Summary statistics for a cycle."""
    cycle: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    agents_active: int = 0
    total_tool_calls: int = 0
    total_messages_sent: int = 0
    total_tokens: int = 0
    storage_reads: int = 0
    storage_writes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for export."""
        return {
            "cycle": self.cycle,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "agents_active": self.agents_active,
            "total_tool_calls": self.total_tool_calls,
            "total_messages_sent": self.total_messages_sent,
            "total_tokens": self.total_tokens,
            "storage_reads": self.storage_reads,
            "storage_writes": self.storage_writes
        }


class TelemetryCollector:
    """
    Collects and indexes all swarm events for analysis.

    Provides methods for:
    - Recording events
    - Building message flow graphs
    - Computing storage access patterns
    - Exporting data in various formats
    """

    def __init__(self):
        self._events: List[TelemetryEvent] = []
        self._cycle_summaries: Dict[int, CycleSummary] = {}
        self._current_cycle = 0

    def record(self, event: TelemetryEvent) -> None:
        """Record a telemetry event."""
        event.cycle = self._current_cycle
        self._events.append(event)

    def record_event(
        self,
        event_type: EventType,
        agent_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> TelemetryEvent:
        """Convenience method to create and record an event."""
        event = TelemetryEvent(
            event_type=event_type,
            cycle=self._current_cycle,
            agent_id=agent_id,
            data=data or {}
        )
        self._events.append(event)
        return event

    def start_cycle(self, cycle: int, agents_active: int) -> None:
        """Record the start of a cycle."""
        self._current_cycle = cycle
        self._cycle_summaries[cycle] = CycleSummary(
            cycle=cycle,
            start_time=time_provider.now(),
            agents_active=agents_active
        )
        self.record_event(EventType.CYCLE_START, data={"agents_active": agents_active})

    def end_cycle(
        self,
        tool_calls: int = 0,
        messages_sent: int = 0,
        tokens: int = 0,
        storage_reads: int = 0,
        storage_writes: int = 0
    ) -> None:
        """Record the end of a cycle with summary stats."""
        if self._current_cycle in self._cycle_summaries:
            summary = self._cycle_summaries[self._current_cycle]
            summary.end_time = time_provider.now()
            summary.duration_ms = (summary.end_time - summary.start_time).total_seconds() * 1000
            summary.total_tool_calls = tool_calls
            summary.total_messages_sent = messages_sent
            summary.total_tokens = tokens
            summary.storage_reads = storage_reads
            summary.storage_writes = storage_writes

        self.record_event(EventType.CYCLE_END, data={
            "tool_calls": tool_calls,
            "messages_sent": messages_sent,
            "tokens": tokens
        })

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        agent_id: Optional[str] = None,
        cycle: Optional[int] = None
    ) -> List[TelemetryEvent]:
        """Get filtered events."""
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if cycle is not None:
            events = [e for e in events if e.cycle == cycle]

        return events

    def get_message_graph(self) -> Dict[str, Any]:
        """
        Build a message flow graph from TOOL_CALL events.

        Returns:
            Dict with nodes (agent IDs) and edges (from, to, count, cycles)
        """
        nodes = set()
        edge_data: Dict[tuple, Dict[str, Any]] = {}

        for event in self._events:
            if event.event_type == EventType.TOOL_CALL:
                tool_name = event.data.get("tool", "")
                if tool_name == "send_message":
                    from_agent = event.agent_id
                    args = event.data.get("args", {})
                    to_agent = args.get("to")

                    if from_agent and to_agent:
                        nodes.add(from_agent)
                        nodes.add(to_agent)

                        edge = (from_agent, to_agent)
                        if edge not in edge_data:
                            edge_data[edge] = {"count": 0, "first_cycle": event.cycle, "last_cycle": event.cycle}

                        edge_data[edge]["count"] += 1
                        edge_data[edge]["last_cycle"] = max(edge_data[edge]["last_cycle"], event.cycle)

        return {
            "nodes": [{"id": n} for n in nodes],
            "edges": [
                {
                    "from": f,
                    "to": t,
                    "count": d["count"],
                    "first_cycle": d["first_cycle"],
                    "last_cycle": d["last_cycle"]
                }
                for (f, t), d in edge_data.items()
            ]
        }

    def get_storage_heatmap(self) -> Dict[str, Any]:
        """
        Build storage access pattern data from TOOL_CALL events.

        Returns:
            Dict with per-key access counts and timeline.
        """
        key_access: Dict[str, Dict[str, int]] = {}
        timeline: List[Dict[str, Any]] = []

        # Map tool names to operation types
        storage_tools = {
            "read_storage": "read",
            "write_storage": "write",
            "append_storage": "append"
        }

        for event in self._events:
            if event.event_type == EventType.TOOL_CALL:
                tool_name = event.data.get("tool", "")
                if tool_name in storage_tools:
                    op = storage_tools[tool_name]
                    args = event.data.get("args", {})
                    key = args.get("key", "unknown")

                    if key not in key_access:
                        key_access[key] = {"reads": 0, "writes": 0, "appends": 0}

                    key_access[key][f"{op}s"] += 1

                    timeline.append({
                        "cycle": event.cycle,
                        "key": key,
                        "operation": op,
                        "agent_id": event.agent_id,
                        "timestamp": event.timestamp.isoformat()
                    })

        # Sort keys by total access count
        sorted_keys = sorted(
            key_access.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )

        return {
            "by_key": dict(sorted_keys),
            "timeline": timeline,
            "hotspots": [k for k, v in sorted_keys[:10]]  # Top 10 most accessed
        }

    def get_agent_activity(self, agent_id: str) -> Dict[str, Any]:
        """Get activity metrics for a specific agent."""
        agent_events = self.get_events(agent_id=agent_id)

        tool_calls = len([e for e in agent_events if e.event_type == EventType.TOOL_CALL])

        # Count message operations from TOOL_CALL events
        messages_sent = len([
            e for e in agent_events
            if e.event_type == EventType.TOOL_CALL and e.data.get("tool") == "send_message"
        ])
        messages_read = len([
            e for e in agent_events
            if e.event_type == EventType.TOOL_CALL and e.data.get("tool") == "read_inbox"
        ])

        total_tokens = sum(
            e.data.get("tokens", 0)
            for e in agent_events
            if e.event_type == EventType.AGENT_TURN_END
        )

        cycles_active = len(set(
            e.cycle for e in agent_events
            if e.event_type == EventType.AGENT_TURN_START
        ))

        return {
            "agent_id": agent_id,
            "tool_calls": tool_calls,
            "messages_sent": messages_sent,
            "messages_read": messages_read,
            "total_tokens": total_tokens,
            "cycles_active": cycles_active
        }

    def get_cycle_summaries(self) -> List[CycleSummary]:
        """Get all cycle summaries."""
        return [self._cycle_summaries[c] for c in sorted(self._cycle_summaries.keys())]

    def export_json(
        self,
        path: str,
        swarm_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Export telemetry to JSON file.

        The swarm_state is the primary data repository containing all state.
        Additional computed/derived fields are included for convenience.

        Args:
            path: Output file path
            swarm_state: Complete swarm state for resume capability (required)
        """
        export_data = {
            "generated_at": time_provider.now().isoformat(),
            "total_events": len(self._events),
            "total_cycles": len(self._cycle_summaries),
            "swarm_state": swarm_state or {},
            "message_graph": self.get_message_graph(),
            "storage_heatmap": self.get_storage_heatmap()
        }

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def export_graphml(self, path: str) -> None:
        """Export message graph to GraphML file."""
        graph = self.get_message_graph()

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="weight" for="edge" attr.name="weight" attr.type="int"/>',
            '  <key id="first_cycle" for="edge" attr.name="first_cycle" attr.type="int"/>',
            '  <key id="last_cycle" for="edge" attr.name="last_cycle" attr.type="int"/>',
            '  <graph id="G" edgedefault="directed">'
        ]

        for node in graph["nodes"]:
            lines.append(f'    <node id="{node["id"]}"/>')

        for i, edge in enumerate(graph["edges"]):
            lines.append(f'    <edge id="e{i}" source="{edge["from"]}" target="{edge["to"]}">')
            lines.append(f'      <data key="weight">{edge["count"]}</data>')
            lines.append(f'      <data key="first_cycle">{edge["first_cycle"]}</data>')
            lines.append(f'      <data key="last_cycle">{edge["last_cycle"]}</data>')
            lines.append('    </edge>')

        lines.append('  </graph>')
        lines.append('</graphml>')

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def clear(self) -> None:
        """Clear all telemetry data."""
        self._events.clear()
        self._cycle_summaries.clear()
        self._current_cycle = 0

    def print_cycle_summary(self, cycle: int) -> None:
        """Print a human-readable summary of a cycle."""
        if cycle not in self._cycle_summaries:
            print(f"No data for cycle {cycle}")
            return

        s = self._cycle_summaries[cycle]
        print(f"\n=== Cycle {cycle} ===")
        print(f"Duration: {s.duration_ms:.1f}ms" if s.duration_ms else "Duration: N/A")
        print(f"Agents: {s.agents_active}")
        print(f"Tool calls: {s.total_tool_calls}")
        print(f"Messages: {s.total_messages_sent}")
        print(f"Tokens: {s.total_tokens}")
        print(f"Storage R/W: {s.storage_reads}/{s.storage_writes}")
