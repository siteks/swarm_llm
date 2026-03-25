#!/usr/bin/env python3
"""Interactive TUI for exploring swarm JSON event logs.

A folding event viewer with vim-style navigation and filtering.
Uses virtual rendering for performance with large files.

Usage:
    python scripts/swarm_viewer.py path/to/swarm.json

Requirements:
    pip install textual

Navigation:
    j/↓         Move down (scrolls within expanded events)
    k/↑         Move up (scrolls within expanded events)
    gg/Home     Jump to first event
    G/End       Jump to last event
    Ctrl+d      Page down
    Ctrl+u      Page up
    Enter/l     Expand event
    h/Esc       Collapse event
    Space       Toggle expand/collapse

Filtering (toggles):
    1-9         Toggle agent visibility by index
    t           Toggle TOOL_CALL events
    c           Toggle AGENT_CONTENT events
    i           Toggle AGENT_THINKING events
    s           Toggle system events
    a           Reset all filters
    /           Search mode
    n           Next search result
    N           Previous search result

Other:
    ?           Show help
    q           Quit
"""

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from textual.app import App, ComposeResult
    from textual.screen import ModalScreen
    from textual.widgets import Static
    from rich.text import Text
except ImportError:
    print("Error: textual not installed. Run: pip install textual", file=sys.stderr)
    sys.exit(1)


# Agent color palette - dark colors for light background
AGENT_COLORS = [
    "dark_cyan",
    "dark_orange",
    "dark_magenta",
    "blue",
    "dark_green",
    "red",
    "dark_cyan",
    "orange3",
    "purple",
    "dodger_blue2",
]

# Event type display colors - dark text for light background
EVENT_TYPE_COLORS = {
    "tool_call": "dark_green",
    "agent_content": "black",
    "agent_thinking": "grey50",
    "cycle_start": "bold orange3",
    "cycle_end": "grey50",
    "agent_turn_start": "grey62",
    "agent_turn_end": "grey62",
    "agent_spawn": "green",
    "swarm_start": "bold blue",
    "swarm_stop": "bold blue",
    "storage_entropy": "dark_magenta",
}


@dataclass
class SwarmEvent:
    """Parsed swarm event."""
    id: str
    event_type: str
    timestamp: datetime
    cycle: int
    agent_id: Optional[str]
    data: dict
    raw: dict

    @classmethod
    def from_dict(cls, d: dict) -> "SwarmEvent":
        ts = d.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(ts) if ts else datetime.min
        except ValueError:
            timestamp = datetime.min
        return cls(
            id=d.get("id", ""),
            event_type=d.get("event_type", "unknown"),
            timestamp=timestamp,
            cycle=d.get("cycle", 0),
            agent_id=d.get("agent_id"),
            data=d.get("data", {}),
            raw=d,
        )

    def summary_line(self, swarm_data: Optional["SwarmData"] = None) -> str:
        """Get single-line summary for collapsed view."""
        if swarm_data and swarm_data.sim_start_time and self.timestamp != datetime.min:
            sim_seconds = swarm_data.get_sim_time(self.timestamp)
            minutes = int(sim_seconds // 60)
            seconds = sim_seconds % 60
            time_str = f"{minutes:02d}:{seconds:05.2f}"
        else:
            time_str = "00:00.00"

        agent_str = self.agent_id or "system"
        type_str = self.event_type.upper()[:15].ljust(15)
        summary = self._get_summary()

        return f"[{time_str}] [{agent_str}] Cycle {self.cycle} | {type_str} | {summary}"

    def _get_summary(self) -> str:
        """Get event-specific summary text."""
        if self.event_type == "tool_call":
            tool = self.data.get("tool", "?")
            args = self.data.get("args", {})
            result = self.data.get("result", {})
            success = result.get("success", False)
            status = "OK" if success else "FAIL"

            if tool == "read_storage":
                key = args.get("key", "?")
                return f'{tool}(key="{key}") -> {status}'
            elif tool == "write_storage":
                key = args.get("key", "?")
                budget_used = result.get("budget_chars_used")
                budget_left = result.get("budget_chars_remaining")
                if budget_used is not None and budget_left is not None:
                    return f'{tool}(key="{key}") -> {status} [{budget_used} chars, {budget_left} left]'
                return f'{tool}(key="{key}") -> {status}'
            elif tool == "append_storage":
                key = args.get("key", "?")
                budget_used = result.get("budget_chars_used")
                budget_left = result.get("budget_chars_remaining")
                if budget_used is not None and budget_left is not None:
                    return f'{tool}(key="{key}") -> {status} [{budget_used} chars, {budget_left} left]'
                return f'{tool}(key="{key}") -> {status}'
            elif tool == "send_message":
                to = args.get("to", "?")
                budget_used = result.get("budget_chars_used")
                budget_left = result.get("budget_chars_remaining")
                if budget_used is not None and budget_left is not None:
                    return f'{tool}(to="{to}") -> {status} [{budget_used} chars, {budget_left} left]'
                return f'{tool}(to="{to}") -> {status}'
            elif tool == "read_inbox":
                msgs = len(result.get("messages", []))
                return f"{tool}() -> {status} ({msgs} msgs)"
            elif tool == "list_agents":
                agents = len(result.get("agents", []))
                return f"{tool}() -> {status} ({agents} agents)"
            elif tool == "list_storage_keys":
                keys = len(result.get("keys", []))
                return f"{tool}() -> {status} ({keys} keys)"
            else:
                return f"{tool}() -> {status}"

        elif self.event_type == "agent_content":
            content = self.data.get("content", "")
            content = content.replace("\n", " ")[:60]
            if len(self.data.get("content", "")) > 60:
                content += "..."
            return f'"{content}"'

        elif self.event_type == "agent_thinking":
            content = self.data.get("content", "")
            content = content.replace("\n", " ")[:50]
            if len(self.data.get("content", "")) > 50:
                content += "..."
            return f'"{content}"'

        elif self.event_type == "cycle_start":
            agents = self.data.get("agents_active", 0)
            return f"{agents} agents active"

        elif self.event_type == "cycle_end":
            return ""

        elif self.event_type == "agent_spawn":
            model = self.data.get("model", "?")
            return f"model={model}"

        elif self.event_type == "swarm_start":
            cycles = self.data.get("max_cycles", "?")
            agents = self.data.get("agents", "?")
            return f"{agents} agents, {cycles} max cycles"

        elif self.event_type == "swarm_stop":
            reason = self.data.get("reason", "")
            return reason

        elif self.event_type == "storage_entropy":
            return "entropy applied"

        else:
            return ""

    def expanded_content(self, wrap_width: int = 70) -> str:
        """Get full content for expanded view with word wrapping."""
        lines = []

        def wrap_text(text: str, indent: str = "  ") -> list[str]:
            """Word wrap text with indent."""
            wrapped = textwrap.wrap(text, width=wrap_width, break_long_words=False, break_on_hyphens=False)
            return [f"{indent}{line}" for line in wrapped] if wrapped else [f"{indent}(empty)"]

        def format_value(value, max_len: int = 500) -> str:
            """Format a value for display, truncating if needed."""
            if isinstance(value, str):
                text = value
            else:
                text = json.dumps(value, ensure_ascii=False)
            if len(text) > max_len:
                text = text[:max_len] + "..."
            return text

        if self.event_type == "tool_call":
            tool = self.data.get("tool", "?")
            args = self.data.get("args", {})
            result = self.data.get("result", {})

            if tool in ("write_storage", "append_storage"):
                key = args.get("key", "?")
                value = args.get("value", "")
                lines.append(f"Key: {key}")
                if isinstance(value, list):
                    lines.append(f"Content ({len(value)} items, {len(json.dumps(value))} chars):")
                    for i, item in enumerate(value[:20]):  # Limit to 20 items
                        lines.append(f"  Item [{i}]:")
                        lines.extend(wrap_text(format_value(item), indent="    "))
                    if len(value) > 20:
                        lines.append(f"  ... and {len(value) - 20} more items")
                else:
                    lines.append(f"Content ({len(json.dumps(value))} chars):")
                    lines.extend(wrap_text(format_value(value)))

            elif tool == "send_message":
                to = args.get("to", "?")
                content = args.get("content", "")
                lines.append(f"To: {to}")
                lines.append(f"Message ({len(content)} chars):")
                lines.extend(wrap_text(format_value(content)))

            elif tool == "read_inbox":
                messages = result.get("messages", [])
                lines.append(f"Messages ({len(messages)}):")
                for i, msg in enumerate(messages[:20]):  # Limit to first 20 messages
                    cycle = msg.get("cycle", "?")
                    sender = msg.get("from", "?")
                    content = msg.get("content", "")
                    lines.append(f"  Message [{i}] - Cycle {cycle}, from {sender}:")
                    # Show full message content without truncation
                    lines.extend(wrap_text(format_value(content, max_len=10000), indent="    "))
                if len(messages) > 20:
                    lines.append(f"  ... and {len(messages) - 20} more messages")

            elif tool == "read_storage":
                key = args.get("key", "?")
                value = result.get("value")
                lines.append(f"Key: {key}")
                if value is None:
                    lines.append("Value: (not found)")
                elif isinstance(value, list):
                    lines.append(f"Values ({len(value)} items):")
                    for i, item in enumerate(value[:20]):  # Limit to 20 items
                        lines.append(f"  Value [{i}]:")
                        lines.extend(wrap_text(format_value(item), indent="    "))
                    if len(value) > 20:
                        lines.append(f"  ... and {len(value) - 20} more items")
                else:
                    lines.append("Value:")
                    lines.extend(wrap_text(format_value(value)))

            elif tool == "list_agents":
                agents = result.get("agents", [])
                lines.append(f"Agents ({len(agents)}):")
                for agent in agents:
                    lines.append(f"  - {agent}")

            elif tool == "list_storage_keys":
                keys = result.get("keys", [])
                lines.append(f"Keys ({len(keys)}):")
                for key in keys[:20]:  # Limit display
                    lines.append(f"  - {key}")
                if len(keys) > 20:
                    lines.append(f"  ... and {len(keys) - 20} more keys")

            else:
                # Generic tool display
                lines.append(f"Tool: {tool}")
                lines.append(f"Args: {json.dumps(args, indent=2)}")
                lines.append(f"Result: {json.dumps(result, indent=2)}")

        elif self.event_type in ("agent_content", "agent_thinking"):
            content = self.data.get("content", "")
            # Don't wrap agent content - preserve formatting
            lines.append(content)

        elif self.event_type == "agent_spawn":
            lines.append(f"Model: {self.data.get('model', '?')}")
            prompt = self.data.get("system_prompt", "")
            if prompt:
                lines.append("System prompt:")
                lines.extend(wrap_text(prompt[:800]))
                if len(prompt) > 800:
                    lines.append(f"  ... ({len(prompt) - 800} more chars)")

        else:
            lines.append(json.dumps(self.data, indent=2))

        return "\n".join(lines)


@dataclass
class SwarmData:
    """Container for loaded swarm data."""
    events: list[SwarmEvent] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    cycle_start_times: dict[int, datetime] = field(default_factory=dict)
    sim_start_time: Optional[datetime] = None
    pause_gaps: list[tuple[datetime, float]] = field(default_factory=list)  # (resume_time, gap_seconds)
    total_cycles: int = 0
    filename: str = ""

    def get_sim_time(self, timestamp: datetime) -> float:
        """Get simulation time in seconds, correcting for pause gaps."""
        if self.sim_start_time is None or timestamp == datetime.min:
            return 0.0
        elapsed = (timestamp - self.sim_start_time).total_seconds()
        # Subtract all pause gaps that occurred before this timestamp
        for resume_time, gap_seconds in self.pause_gaps:
            if timestamp >= resume_time:
                elapsed -= gap_seconds
        return max(0.0, elapsed)

    @classmethod
    def load(cls, path: Path) -> "SwarmData":
        with open(path) as f:
            data = json.load(f)

        raw_events = data.get("swarm_state", {}).get("telemetry", {}).get("events", [])
        events = [SwarmEvent.from_dict(e) for e in raw_events]
        agents = sorted(set(e.agent_id for e in events if e.agent_id))

        cycle_start_times = {}
        for e in events:
            if e.event_type == "cycle_start":
                cycle_start_times[e.cycle] = e.timestamp

        # Find simulation start time (first swarm_start)
        sim_start_time = None
        for e in events:
            if e.event_type == "swarm_start":
                sim_start_time = e.timestamp
                break

        # Calculate pause gaps (time between swarm_stop and swarm_resume)
        pause_gaps = []
        last_stop_time = None
        for e in events:
            if e.event_type == "swarm_stop":
                last_stop_time = e.timestamp
            elif e.event_type == "swarm_resume" and last_stop_time is not None:
                gap_seconds = (e.timestamp - last_stop_time).total_seconds()
                pause_gaps.append((e.timestamp, gap_seconds))
                last_stop_time = None

        # Get total cycles from swarm_state
        total_cycles = data.get("swarm_state", {}).get("current_cycle", 0)

        return cls(events=events, agents=agents, cycle_start_times=cycle_start_times,
                   sim_start_time=sim_start_time, pause_gaps=pause_gaps,
                   total_cycles=total_cycles, filename=path.name)


class HelpScreen(ModalScreen):
    """Modal help screen."""

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-box {
        width: 68;
        height: auto;
        padding: 1 2;
        border: solid green;
    }
    """

    def compose(self) -> ComposeResult:
        help_text = """\
SWARM VIEWER HELP

NAVIGATION
  j / Down      Move down (scrolls within expanded)
  k / Up        Move up (scrolls within expanded)
  g g / Home    Jump to first event
  G / End       Jump to last event
  PgDn          Page down
  PgUp          Page up

EXPAND/COLLAPSE
  Enter / l     Expand current event
  h / Escape    Collapse current event
  Space         Toggle expand/collapse

When an event is expanded, j/k scrolls through
its content line by line. Reach the end to move
to the next event.

FILTERING (toggles)
  1-9           Toggle agent by index
  t             Toggle TOOL_CALL
  c             Toggle AGENT_CONTENT
  i             Toggle AGENT_THINKING
  s             Toggle system events
  a             Reset all filters

OTHER
  ?             Show this help
  q             Quit

Press any key to close"""
        yield Static(help_text, id="help-box")

    def on_key(self, event) -> None:
        event.prevent_default()
        event.stop()
        self.dismiss()


class SwarmViewer(App):
    """Main swarm viewer application."""

    # Use light theme
    theme = "textual-light"

    CSS = """
    #event-list {
        height: 1fr;
        padding: 0 1;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 1;
    }
    """

    # Key handling is done in on_key() method
    BINDINGS = []

    def __init__(self, data: SwarmData, filepath: str):
        super().__init__()
        self.data = data
        self.filepath = filepath

        # Filtered event indices (into data.events)
        self.filtered_indices: list[int] = list(range(len(data.events)))

        # Set of expanded event indices
        self.expanded_events: set[int] = set()

        # Filter state
        self.hidden_agents: set[str] = set()
        self.hidden_event_types: set[str] = set()  # Event types to hide
        self.search_query: str = ""

        # Event type categories for toggling
        self.SYSTEM_EVENT_TYPES = {
            "cycle_start", "cycle_end", "agent_turn_start", "agent_turn_end",
            "agent_spawn", "swarm_start", "swarm_stop", "swarm_resume", "storage_entropy"
        }

        # Navigation state
        self.current_index: int = 0
        self._pending_gg = False
        self.expanded_scroll_offset: int = 0  # Line offset within current expanded event
        self._viewport_scroll: int = 0  # Viewport scroll position

        # Assign colors to agents
        self.agent_colors: dict[str, str] = {}
        for i, agent in enumerate(data.agents):
            self.agent_colors[agent] = AGENT_COLORS[i % len(AGENT_COLORS)]

    def compose(self) -> ComposeResult:
        yield Static("", id="event-list")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self.title = f"Swarm Viewer"
        self._rebuild_filter()
        self._refresh_display()
        # Schedule a refresh after layout is complete to get correct viewport height
        self.set_timer(0.1, self._refresh_display)

    def _rebuild_filter(self) -> None:
        """Rebuild filtered_indices based on current filters."""
        self.filtered_indices.clear()

        for i, event in enumerate(self.data.events):
            # Agent filter
            if event.agent_id and event.agent_id in self.hidden_agents:
                continue
            # Event type filter
            if event.event_type in self.hidden_event_types:
                continue
            # Search filter
            if self.search_query:
                search_lower = self.search_query.lower()
                match = False
                summary = event.summary_line(self.data)
                if search_lower in summary.lower():
                    match = True
                content = event.data.get("content", "")
                if search_lower in content.lower():
                    match = True
                if event.event_type == "tool_call":
                    args_str = json.dumps(event.data.get("args", {}))
                    result_str = json.dumps(event.data.get("result", {}))
                    if search_lower in args_str.lower() or search_lower in result_str.lower():
                        match = True
                if not match:
                    continue

            self.filtered_indices.append(i)

        # Clamp current index and reset scroll offsets
        if self.filtered_indices:
            self.current_index = min(self.current_index, len(self.filtered_indices) - 1)
        else:
            self.current_index = 0
        self.expanded_scroll_offset = 0
        self._viewport_scroll = 0

    def _refresh_display(self) -> None:
        """Refresh the event list and status bar."""
        event_list = self.query_one("#event-list", Static)
        event_list.update(self._render_events())

        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(self._render_status())

    def _render_events(self) -> Text:
        """Render visible events."""
        text = Text()

        if not self.filtered_indices:
            text.append("No events match current filters", style="italic grey50")
            return text

        # Get viewport height
        try:
            event_list = self.query_one("#event-list", Static)
            viewport_height = event_list.size.height if event_list.size.height > 0 else 40
        except Exception:
            viewport_height = 40

        # Build line positions map
        line_positions = []
        current_line = 0
        max_expanded_lines = 500  # Must match _render_event

        for idx in self.filtered_indices:
            line_positions.append(current_line)
            if idx in self.expanded_events:
                event = self.data.events[idx]
                expanded_line_count = event.expanded_content().count('\n') + 1
                # Account for truncation in _render_event
                if expanded_line_count > max_expanded_lines:
                    expanded_line_count = max_expanded_lines + 1  # +1 for "more lines" indicator
                current_line += 1 + expanded_line_count
            else:
                current_line += 1

        total_lines = current_line

        # Find scroll position to keep current line visible
        # Account for expanded_scroll_offset when scrolling within expanded content
        if self.current_index < len(line_positions):
            current_event_line = line_positions[self.current_index]
            # Add scroll offset within expanded event (offset 0 = summary, 1+ = expanded lines)
            current_line_pos = current_event_line + self.expanded_scroll_offset
        else:
            current_line_pos = 0

        # Use stored scroll position, adjusting to keep current line visible
        if not hasattr(self, '_viewport_scroll'):
            self._viewport_scroll = 0

        # Ensure current line is visible with some margin
        margin = 2  # Keep a few lines of context
        if current_line_pos < self._viewport_scroll + margin:
            # Current line is above visible area - scroll up
            self._viewport_scroll = max(0, current_line_pos - margin)
        elif current_line_pos >= self._viewport_scroll + viewport_height - margin:
            # Current line is below visible area - scroll down
            self._viewport_scroll = current_line_pos - viewport_height + margin + 1

        # Clamp scroll position
        self._viewport_scroll = max(0, self._viewport_scroll)
        self._viewport_scroll = min(self._viewport_scroll, max(0, total_lines - viewport_height))

        visible_start = self._viewport_scroll
        visible_end = self._viewport_scroll + viewport_height

        # Render visible events
        current_line = 0
        last_cycle = None

        for list_idx, event_idx in enumerate(self.filtered_indices):
            event = self.data.events[event_idx]
            is_expanded = event_idx in self.expanded_events

            if is_expanded:
                expanded_line_count = event.expanded_content().count('\n') + 1
                if expanded_line_count > max_expanded_lines:
                    expanded_line_count = max_expanded_lines + 1  # +1 for "more lines" indicator
                event_height = 1 + expanded_line_count
            else:
                event_height = 1

            event_start = current_line
            event_end = current_line + event_height

            # Only render if visible
            if event_end > visible_start and event_start < visible_end:
                # Cycle header
                if event.event_type == "cycle_start" and event.cycle != last_cycle:
                    last_cycle = event.cycle
                    if text.plain:
                        text.append("\n")
                    text.append(f"═══ CYCLE {event.cycle} ", style="bold orange3")
                    text.append("═" * 50, style="grey62")
                    text.append("\n")

                # Render event
                is_current = (list_idx == self.current_index)
                scroll_offset = self.expanded_scroll_offset if is_current else 0
                self._render_event(text, event, is_current, is_expanded,
                                   scroll_offset=scroll_offset,
                                   event_start_line=event_start,
                                   visible_start=visible_start,
                                   visible_end=visible_end)

            current_line = event_end

        return text

    def _render_event(self, text: Text, event: SwarmEvent, is_current: bool, is_expanded: bool,
                       max_expanded_lines: int = 500, scroll_offset: int = 0,
                       event_start_line: int = 0, visible_start: int = 0, visible_end: int = 1000):
        """Render a single event, clipping to visible range."""
        summary = event.summary_line(self.data)
        event_color = EVENT_TYPE_COLORS.get(event.event_type, "black")

        # Line 0 of this event is the summary line
        summary_line_num = event_start_line

        # Only render summary if it's in visible range
        if visible_start <= summary_line_num < visible_end:
            # Current indicator
            if is_current:
                text.append("► ", style="bold blue")
            else:
                text.append("  ")

            # Expand indicator
            if is_expanded:
                text.append("▼ ", style="bold")
            else:
                text.append("▶ ", style="grey50")

            # Summary line - highlight if current and not scrolled into expanded content
            if is_current and (not is_expanded or scroll_offset == 0):
                text.append(summary, style=f"{event_color} reverse")
            else:
                text.append(summary, style=event_color)
            text.append("\n")

        # Expanded content - only render lines within visible range
        if is_expanded:
            expanded = event.expanded_content()
            expanded_lines = expanded.split("\n")
            lines_to_show = expanded_lines[:max_expanded_lines]
            has_more = len(expanded_lines) > max_expanded_lines

            for line_idx, line in enumerate(lines_to_show):
                # Calculate absolute line number for this expanded line
                # Summary is at event_start_line, expanded lines start at event_start_line + 1
                abs_line_num = event_start_line + 1 + line_idx

                # Only render if within visible range
                if abs_line_num < visible_start or abs_line_num >= visible_end:
                    continue

                # Highlight the current line if scrolled into expanded content
                # scroll_offset 0 = summary line, 1 = first expanded line, etc.
                is_current_line = is_current and (line_idx + 1 == scroll_offset)
                if is_current_line:
                    text.append(f"    ► {line}\n", style="grey50 reverse")
                else:
                    text.append(f"      {line}\n", style="grey50")

            if has_more:
                # Show "more lines" indicator if it's in visible range
                more_line_num = event_start_line + 1 + len(lines_to_show)
                if visible_start <= more_line_num < visible_end:
                    remaining = len(expanded_lines) - max_expanded_lines
                    text.append(f"      ... ({remaining} more lines)\n", style="italic grey62")

    def _render_status(self) -> Text:
        """Render status bar."""
        text = Text()

        # Filename and cycles
        text.append(f"{self.data.filename}", style="bold")
        text.append(f" ({self.data.total_cycles} cycles)")
        text.append(" │ ")

        # Position
        if self.filtered_indices:
            text.append(f"Event {self.current_index + 1}/{len(self.filtered_indices)}")
            if len(self.filtered_indices) != len(self.data.events):
                text.append(f" (of {len(self.data.events)})")
        else:
            text.append("No events")

        text.append(" │ ")

        # Filters
        filters = []
        if self.hidden_agents:
            filters.append(f"agents: -{', '.join(sorted(self.hidden_agents))}")
        if self.hidden_event_types:
            # Summarize hidden types
            hidden = self.hidden_event_types
            parts = []
            if "tool_call" in hidden:
                parts.append("-t")
            if "agent_content" in hidden:
                parts.append("-c")
            if "agent_thinking" in hidden:
                parts.append("-i")
            sys_hidden = hidden & self.SYSTEM_EVENT_TYPES
            if sys_hidden == self.SYSTEM_EVENT_TYPES:
                parts.append("-s")
            elif sys_hidden:
                parts.append(f"-sys:{len(sys_hidden)}")
            if parts:
                filters.append(f"types: {' '.join(parts)}")
        if self.search_query:
            filters.append(f"search: '{self.search_query}'")

        if filters:
            text.append("Filters: " + "; ".join(filters), style="italic")
            text.append(" │ ")

        # Agent legend
        for i, agent in enumerate(self.data.agents[:9]):
            if i > 0:
                text.append(" ")
            style = "dim" if agent in self.hidden_agents else ""
            text.append(f"[{i+1}]", style=style)
            text.append(f"{agent}", style=f"{style} {self.agent_colors.get(agent, '')}")

        return text

    def _get_page_size(self) -> int:
        """Get page size based on viewport."""
        try:
            event_list = self.query_one("#event-list", Static)
            if event_list.size.height > 0:
                return max(1, event_list.size.height - 2)
        except Exception:
            pass
        return 20

    # Navigation
    def action_move_down(self) -> None:
        self._pending_gg = False
        idx = self._current_event_idx()
        if idx is not None and idx in self.expanded_events:
            event = self.data.events[idx]
            max_lines = event.expanded_content().count('\n') + 1
            # scroll_offset 0 = summary line, 1..max_lines = expanded content lines
            if self.expanded_scroll_offset < max_lines:
                self.expanded_scroll_offset += 1
                self._refresh_display()
                return
        # Move to next event
        if self.current_index < len(self.filtered_indices) - 1:
            self.current_index += 1
            self.expanded_scroll_offset = 0
            self._refresh_display()

    def action_move_up(self) -> None:
        self._pending_gg = False
        idx = self._current_event_idx()
        if idx is not None and idx in self.expanded_events:
            if self.expanded_scroll_offset > 0:
                self.expanded_scroll_offset -= 1
                self._refresh_display()
                return
        # Move to previous event
        if self.current_index > 0:
            self.current_index -= 1
            self.expanded_scroll_offset = 0
            # If moving to an expanded event, start at bottom (last expanded line)
            new_idx = self._current_event_idx()
            if new_idx is not None and new_idx in self.expanded_events:
                event = self.data.events[new_idx]
                max_lines = event.expanded_content().count('\n') + 1
                self.expanded_scroll_offset = max_lines  # Last line in expanded content
            self._refresh_display()

    def action_goto_start_prefix(self) -> None:
        if self._pending_gg:
            self.action_goto_start()
            self._pending_gg = False
        else:
            self._pending_gg = True

    def action_goto_start(self) -> None:
        self._pending_gg = False
        self.current_index = 0
        self.expanded_scroll_offset = 0
        self._viewport_scroll = 0
        self._refresh_display()

    def action_goto_end(self) -> None:
        self._pending_gg = False
        if self.filtered_indices:
            self.current_index = len(self.filtered_indices) - 1
            self.expanded_scroll_offset = 0
            self._viewport_scroll = max(0, len(self.filtered_indices) * 2)  # Approximate, will be clamped
            self._refresh_display()

    def _get_line_position(self) -> int:
        """Get current absolute line position."""
        line_pos = 0
        for i, idx in enumerate(self.filtered_indices):
            if i == self.current_index:
                return line_pos + self.expanded_scroll_offset
            if idx in self.expanded_events:
                event = self.data.events[idx]
                line_pos += 1 + event.expanded_content().count('\n') + 1
            else:
                line_pos += 1
        return line_pos

    def _set_line_position(self, target_line: int) -> None:
        """Set current_index and expanded_scroll_offset to reach target line."""
        line_pos = 0
        for i, idx in enumerate(self.filtered_indices):
            if idx in self.expanded_events:
                event = self.data.events[idx]
                expanded_lines = event.expanded_content().count('\n') + 1
                event_height = 1 + expanded_lines
            else:
                event_height = 1

            # Check if target is within this event
            if line_pos + event_height > target_line:
                self.current_index = i
                self.expanded_scroll_offset = max(0, target_line - line_pos)
                # Clamp scroll offset to valid range
                if idx in self.expanded_events:
                    max_offset = expanded_lines
                    self.expanded_scroll_offset = min(self.expanded_scroll_offset, max_offset)
                else:
                    self.expanded_scroll_offset = 0
                return

            line_pos += event_height

        # Past the end - go to last event
        if self.filtered_indices:
            self.current_index = len(self.filtered_indices) - 1
            idx = self.filtered_indices[self.current_index]
            if idx in self.expanded_events:
                event = self.data.events[idx]
                self.expanded_scroll_offset = event.expanded_content().count('\n') + 1
            else:
                self.expanded_scroll_offset = 0

    def action_page_down(self) -> None:
        self._pending_gg = False
        page_size = self._get_page_size()
        current_line = self._get_line_position()
        self._set_line_position(current_line + page_size)
        self._refresh_display()

    def action_page_up(self) -> None:
        self._pending_gg = False
        page_size = self._get_page_size()
        current_line = self._get_line_position()
        self._set_line_position(max(0, current_line - page_size))
        self._refresh_display()

    # Expand/collapse
    def _current_event_idx(self) -> Optional[int]:
        if 0 <= self.current_index < len(self.filtered_indices):
            return self.filtered_indices[self.current_index]
        return None

    def action_expand(self) -> None:
        self._pending_gg = False
        idx = self._current_event_idx()
        if idx is not None:
            self.expanded_events.add(idx)
            self._refresh_display()

    def action_collapse(self) -> None:
        self._pending_gg = False
        idx = self._current_event_idx()
        if idx is not None:
            self.expanded_events.discard(idx)
            self.expanded_scroll_offset = 0
            self._refresh_display()

    def action_collapse_or_clear(self) -> None:
        self._pending_gg = False
        # Clear search if active
        if self.search_query:
            self.search_query = ""
            self._rebuild_filter()
            self._refresh_display()
        else:
            idx = self._current_event_idx()
            if idx is not None:
                self.expanded_events.discard(idx)
                self.expanded_scroll_offset = 0
                self._refresh_display()

    def action_toggle_expand(self) -> None:
        self._pending_gg = False
        idx = self._current_event_idx()
        if idx is not None:
            if idx in self.expanded_events:
                self.expanded_events.discard(idx)
                self.expanded_scroll_offset = 0
            else:
                self.expanded_events.add(idx)
            self._refresh_display()

    # Filtering
    def _toggle_event_type(self, event_type: str) -> None:
        """Toggle visibility of an event type."""
        if event_type in self.hidden_event_types:
            self.hidden_event_types.discard(event_type)
        else:
            self.hidden_event_types.add(event_type)
        self._rebuild_filter()
        self._refresh_display()

    def action_filter_tool(self) -> None:
        self._pending_gg = False
        self._toggle_event_type("tool_call")

    def action_filter_content(self) -> None:
        self._pending_gg = False
        self._toggle_event_type("agent_content")

    def action_filter_thinking(self) -> None:
        self._pending_gg = False
        self._toggle_event_type("agent_thinking")

    def action_filter_system(self) -> None:
        self._pending_gg = False
        # Toggle all system event types together
        all_hidden = all(t in self.hidden_event_types for t in self.SYSTEM_EVENT_TYPES)
        if all_hidden:
            # Show all system events
            self.hidden_event_types -= self.SYSTEM_EVENT_TYPES
        else:
            # Hide all system events
            self.hidden_event_types |= self.SYSTEM_EVENT_TYPES
        self._rebuild_filter()
        self._refresh_display()

    def action_filter_all(self) -> None:
        self._pending_gg = False
        self.hidden_event_types.clear()
        self.hidden_agents.clear()
        self.search_query = ""
        self._rebuild_filter()
        self._refresh_display()

    def _toggle_agent(self, index: int) -> None:
        self._pending_gg = False
        if index < len(self.data.agents):
            agent = self.data.agents[index]
            if agent in self.hidden_agents:
                self.hidden_agents.remove(agent)
            else:
                self.hidden_agents.add(agent)
            self._rebuild_filter()
            self._refresh_display()

    def action_toggle_agent_1(self) -> None:
        self._toggle_agent(0)

    def action_toggle_agent_2(self) -> None:
        self._toggle_agent(1)

    def action_toggle_agent_3(self) -> None:
        self._toggle_agent(2)

    def action_toggle_agent_4(self) -> None:
        self._toggle_agent(3)

    def action_toggle_agent_5(self) -> None:
        self._toggle_agent(4)

    def action_toggle_agent_6(self) -> None:
        self._toggle_agent(5)

    def action_toggle_agent_7(self) -> None:
        self._toggle_agent(6)

    def action_toggle_agent_8(self) -> None:
        self._toggle_agent(7)

    def action_toggle_agent_9(self) -> None:
        self._toggle_agent(8)

    # Search (simplified - no input widget for now)
    def action_start_search(self) -> None:
        self._pending_gg = False
        self.notify("Search not yet implemented")

    def action_next_search(self) -> None:
        self._pending_gg = False
        if self.filtered_indices:
            if self.current_index < len(self.filtered_indices) - 1:
                self.current_index += 1
            else:
                self.current_index = 0
            self._refresh_display()

    def action_prev_search(self) -> None:
        self._pending_gg = False
        if self.filtered_indices:
            if self.current_index > 0:
                self.current_index -= 1
            else:
                self.current_index = len(self.filtered_indices) - 1
            self._refresh_display()

    def action_help(self) -> None:
        self._pending_gg = False
        self.push_screen(HelpScreen())

    def on_key(self, event) -> None:
        """Handle keys directly."""
        key = event.key

        # Direct key handling
        handled = True
        if key == "j" or key == "down":
            self.action_move_down()
        elif key == "k" or key == "up":
            self.action_move_up()
        elif key == "space":
            self.action_toggle_expand()
        elif key == "question_mark":
            self.action_help()
        elif key == "enter":
            self.action_expand()
        elif key == "h":
            self.action_collapse()
        elif key == "escape":
            self.action_collapse_or_clear()
        elif key == "l":
            self.action_expand()
        elif key == "g":
            self.action_goto_start_prefix()
        elif key == "G":
            self.action_goto_end()
        elif key == "home":
            self.action_goto_start()
        elif key == "end":
            self.action_goto_end()
        elif key == "pagedown":
            self.action_page_down()
        elif key == "pageup":
            self.action_page_up()
        elif key == "t":
            self.action_filter_tool()
        elif key == "c":
            self.action_filter_content()
        elif key == "i":
            self.action_filter_thinking()
        elif key == "s":
            self.action_filter_system()
        elif key == "a":
            self.action_filter_all()
        elif key in "123456789":
            self._toggle_agent(int(key) - 1)
        elif key == "q":
            self.exit()
        else:
            handled = False

        if handled:
            event.prevent_default()
            event.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive TUI for exploring swarm JSON event logs"
    )
    parser.add_argument("json_file", help="Path to swarm JSON export")
    parser.add_argument("--filter", help="Initial filter expression (not implemented)")
    parser.add_argument("--cycle", type=int, help="Start at cycle N (not implemented)")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    args = parser.parse_args()

    try:
        data = SwarmData.load(Path(args.json_file))
    except FileNotFoundError:
        print(f"Error: File not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not data.events:
        print("Error: No events found in file", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data.events)} events, {len(data.agents)} agents")

    app = SwarmViewer(data, args.json_file)
    app.run()


if __name__ == "__main__":
    main()
