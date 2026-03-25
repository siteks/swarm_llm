#!/usr/bin/env python3
"""
Extract memory events from swarm JSON files in temporal order.

Events:
- create: New key created (write_storage or append_storage to non-existent key)
- overwrite: Existing key overwritten (write_storage to existing key)
- append: Existing key appended to (append_storage to existing key)

Output columns:
- timestamp: Corrected for resume delta (seconds from swarm start)
- event: create, overwrite, or append
- key: Storage key name
- index: Element index (0 for create/overwrite, list position for append)
- agent: Agent ID
- chars: Number of characters in the value
- delta: Change in character count for the key

Usage:
    python scripts/extract_memevents.py path/to/swarm.json
    python scripts/extract_memevents.py path/to/*.json  # Multiple files
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import textwrap

@dataclass
class MemEvent:
    """A memory event."""
    timestamp: float  # Seconds from swarm start
    event: str  # create, overwrite, append
    key: str
    index: int  # 0 for create/overwrite, list index for append
    agent: str
    chars: int  # Characters in the value
    delta: int  # Change in characters for this key
    cycle: int  # Cycle number
    contents: str
    # Preservation metrics (only set for overwrite events)
    edit_distance: Optional[int] = None
    preservation_score: Optional[float] = None
    growth_ratio: Optional[float] = None


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp to datetime."""
    # Handle both with and without microseconds
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts}")



def get_value_chars(value: Any) -> int:
    """Get character count for a value."""
    if isinstance(value, str):
        return len(value)
    else:
        # JSON serialize for complex values
        return len(json.dumps(value))


def value_to_str(value: Any) -> str:
    """Convert value to string for preservation analysis."""
    if isinstance(value, str):
        return value
    else:
        return json.dumps(value)

def wordwrap_with_indent(text: str, width: int = 100) -> str:
    wrapped_lines = []

    for line in text.splitlines(keepends=False):
        # Preserve empty lines exactly
        if not line.strip():
            wrapped_lines.append(line)
            continue

        # Detect indentation
        indent = len(line) - len(line.lstrip())
        indent_str = " " * indent

        # If the line has no spaces, wrapping would corrupt ASCII art
        if " " not in line.strip():
            wrapped_lines.append(line)
            continue

        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=indent_str,
            subsequent_indent=indent_str,
            break_long_words=False,
            break_on_hyphens=False,
        )

        wrapped_lines.append(wrapper.fill(line.strip()))

    return "\n".join(wrapped_lines)
    
def extract_memevents(json_path: str) -> List[MemEvent]:
    """
    Extract memory events from a swarm JSON file.

    Args:
        json_path: Path to swarm JSON file

    Returns:
        List of MemEvent in temporal order
    """
    with open(json_path) as f:
        data = json.load(f)

    events_data = data.get("swarm_state", {}).get("telemetry", {}).get("events", [])

    # Find swarm_start to get base timestamp
    base_time: Optional[datetime] = None
    resume_deltas: List[Tuple[datetime, float]] = []  # (resume_time, accumulated_delta)

    for e in events_data:
        if e["event_type"] == "swarm_start":
            base_time = parse_timestamp(e["timestamp"])
            break

    if base_time is None:
        print(f"Warning: No swarm_start event found in {json_path}", file=sys.stderr)
        # Fall back to first event
        if events_data:
            base_time = parse_timestamp(events_data[0]["timestamp"])
        else:
            return []

    # Find resume events to calculate time gaps
    # A resume creates a time discontinuity we need to account for
    last_event_time = base_time
    accumulated_delta = 0.0

    for e in events_data:
        event_time = parse_timestamp(e["timestamp"])

        if e["event_type"] == "swarm_resume":
            # Gap between last event before resume and resume time
            # This is the "pause" time we want to subtract
            gap = (event_time - last_event_time).total_seconds()
            if gap > 60:  # Only count gaps > 1 minute as pauses
                accumulated_delta += gap
            resume_deltas.append((event_time, accumulated_delta))

        last_event_time = event_time

    def get_adjusted_time(ts: datetime) -> float:
        """Get time in seconds from start, adjusted for resume gaps."""
        raw_delta = (ts - base_time).total_seconds()

        # Find the most recent resume delta to apply
        delta_to_apply = 0.0
        for resume_time, delta in resume_deltas:
            if ts >= resume_time:
                delta_to_apply = delta

        return raw_delta - delta_to_apply

    # Track key state: key -> (total_chars, num_elements)
    key_state: Dict[str, Tuple[int, int]] = {}

    # Track key contents: key -> list of string values (for preservation analysis)
    key_contents: Dict[str, List[str]] = {}

    # Extract memory events
    mem_events: List[MemEvent] = []

    for e in events_data:
        if e["event_type"] != "tool_call":
            continue

        tool = e.get("data", {}).get("tool", "")
        if tool not in ("write_storage", "append_storage"):
            continue

        args = e.get("data", {}).get("args", {})
        result = e.get("data", {}).get("result", {})

        # Skip failed operations
        if not result.get("success", False):
            continue

        key = args.get("key", "unknown")
        value = args.get("value")
        agent = e.get("agent_id", "unknown")
        cycle = e.get("cycle", 0)
        timestamp = parse_timestamp(e["timestamp"])

        # Skip internal keys
        if key.startswith("_"):
            continue

        value_chars = get_value_chars(value)
        value_str = value_to_str(value)

        # Preservation metrics (only for overwrite)
        edit_distance: Optional[int] = None
        preservation_score: Optional[float] = None
        growth_ratio: Optional[float] = None

        if key not in key_state:
            # New key - create event
            event_type = "create"
            index = 0
            delta = value_chars
            key_state[key] = (value_chars, 1)
            key_contents[key] = [value_str]
        elif tool == "write_storage":
            # Existing key - overwrite
            event_type = "overwrite"
            index = 0
            old_chars = key_state[key][0]
            delta = value_chars - old_chars


            key_state[key] = (value_chars, 1)
            key_contents[key] = [value_str]
        else:
            # append_storage to existing key
            event_type = "append"
            old_chars, num_elements = key_state[key]
            index = num_elements
            delta = value_chars  # Append adds this many chars
            key_state[key] = (old_chars + value_chars, num_elements + 1)
            key_contents.setdefault(key, []).append(value_str)

        mem_events.append(MemEvent(
            timestamp=get_adjusted_time(timestamp),
            event=event_type,
            key=key,
            index=index,
            agent=agent,
            chars=value_chars,
            delta=delta,
            cycle=cycle,
            edit_distance=edit_distance,
            preservation_score=preservation_score,
            growth_ratio=growth_ratio,
            contents=key_contents[key]
        ))

    # Sort by timestamp (should already be sorted, but ensure it)
    mem_events.sort(key=lambda x: x.timestamp)

    return mem_events


def format_memevents(events: List[MemEvent]) -> str:
    """Format memory events as a columnar table."""
    if not events:
        return "# No memory events found\n"

    lines = []

    # Data rows
    for e in events:

        # lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:-1])))
        # lines.append(f"## Cycle {e.cycle} elapsed time {e.timestamp:.2f}s {e.event.upper()} `{e.key}[{e.index}]` ({e.agent}, {e.chars} chars)")
        lines.append(f"## Cycle {e.cycle} {e.event.upper()} `{e.key}[{e.index}]` ({e.agent}, {e.chars} chars)")
        #print(row[-1])
        content = e.contents[e.index] if isinstance(e.contents, list) else e.contents
        content = wordwrap_with_indent(content)
        # if isinstance(content, list):
        #     content = content[-1]
        lines.append(f"```\n{content}\n```")

    return "\n".join(lines) + "\n"


def format_raw(events: List[MemEvent], sep: int = 1) -> str:
    """Format memory events as raw text only, wordwrapped."""
    if not events:
        return ""

    lines = []
    for e in events:
        content = e.contents[e.index] if isinstance(e.contents, list) else e.contents
        lines.append(wordwrap_with_indent(content))

    return ("\n" * (sep + 1)).join(lines) + "\n"


def process_file(json_path: str, raw: bool = False, sep: int = 1) -> None:
    """Process a single JSON file and write output."""
    path = Path(json_path)
    if not path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        return

    events = extract_memevents(json_path)

    if raw:
        output_path = path.with_name(path.stem + "_storage_raw.txt")
        output = format_raw(events, sep=sep)
        with open(output_path, 'w', errors='replace') as f:
            f.write(output)
    else:
        output_path = path.with_name(path.stem + "_storage.md")
        output = format_memevents(events)
        with open(output_path, 'w', errors='replace') as f:
            f.write(f"# Memory events from {path.name}\n")
            f.write(output)

    print(f"Processing: {json_path}")
    print(f"  -> {output_path} ({len(events)} events)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract memory events from swarm JSON files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="JSON files to process"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of file"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw text only (no metadata), to <stem>_storage_raw.txt"
    )
    parser.add_argument(
        "--sep",
        type=int,
        default=1,
        metavar="N",
        help="Number of blank lines between events in --raw output (default: 1)"
    )

    args = parser.parse_args()

    for json_path in args.files:
        if args.stdout:
            events = extract_memevents(json_path)
            if args.raw:
                print(format_raw(events, sep=args.sep))
            else:
                print(format_memevents(events))
        else:
            process_file(json_path, raw=args.raw, sep=args.sep)


if __name__ == "__main__":
    main()
