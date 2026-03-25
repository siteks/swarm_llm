#!/usr/bin/env python3
"""
Generate a summary CSV of swarm runs.

For each JSON file, extracts:
- Run name, agents, total words, LLM errors, run time
- If *_storage_analysis.md exists: broad category, archetype, dominant vibe

Usage:
    python scripts/summarize_runs.py analysis_short_runs/swarm_*.json -o summary.csv
    python scripts/summarize_runs.py analysis_short_runs/swarm_*.json  # stdout
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from swarm_utils import (
    SwarmRun,
    agent_combination,
    agent_content_words_by_cycle,
    send_message_stats_by_cycle,
    storage_write_stats_by_cycle,
    llm_errors_by_cycle,
    token_usage_by_cycle,
    tool_calls_from_turns_by_cycle,
)


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp from event."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def get_run_time_seconds(run: SwarmRun) -> float:
    """Calculate run time by summing cycle durations.

    This correctly handles resumed sims by excluding paused time.
    """
    cycle_starts = {}
    cycle_ends = {}

    for e in run.events:
        event_type = e.get('event_type')
        cycle = e.get('cycle', 0)
        if event_type == 'cycle_start':
            cycle_starts[cycle] = parse_timestamp(e['timestamp'])
        elif event_type == 'cycle_end':
            cycle_ends[cycle] = parse_timestamp(e['timestamp'])

    total_seconds = 0.0
    for cycle, start_ts in cycle_starts.items():
        if cycle in cycle_ends:
            total_seconds += (cycle_ends[cycle] - start_ts).total_seconds()

    return total_seconds


def get_total_words(run: SwarmRun) -> int:
    """Sum all words across content, messages, and storage."""
    content = agent_content_words_by_cycle(run)
    messages = send_message_stats_by_cycle(run)
    storage = storage_write_stats_by_cycle(run)

    total = 0

    # Content words
    for cycle_data in content.values():
        total += sum(cycle_data.values())

    # Message words
    for cycle_data in messages.values():
        for agent_data in cycle_data.values():
            total += agent_data.get('words', 0)

    # Storage words
    for cycle_data in storage.values():
        for agent_data in cycle_data.values():
            total += agent_data.get('words', 0)

    return total


def get_chars_per_agent(run: SwarmRun) -> dict[str, int]:
    """Get total characters (storage + messages) per agent.

    Returns: {agent_id: total_chars}
    """
    messages = send_message_stats_by_cycle(run)
    storage = storage_write_stats_by_cycle(run)

    totals = defaultdict(int)

    # Message chars
    for cycle_data in messages.values():
        for agent_id, agent_data in cycle_data.items():
            totals[agent_id] += agent_data.get('chars', 0)

    # Storage chars
    for cycle_data in storage.values():
        for agent_id, agent_data in cycle_data.items():
            totals[agent_id] += agent_data.get('chars', 0)

    return dict(totals)


def get_total_llm_errors(run: SwarmRun) -> int:
    """Count total LLM errors across all cycles."""
    errors = llm_errors_by_cycle(run)
    return sum(len(cycle_errors) for cycle_errors in errors.values())


def get_total_tokens(run: SwarmRun) -> int:
    """Sum all tokens across all cycles and agents."""
    usage = token_usage_by_cycle(run)
    return sum(sum(agents.values()) for agents in usage.values())


def get_storage_chars_stats(run: SwarmRun) -> tuple[float, float]:
    """Mean and std dev of chars written/appended to storage per cycle."""
    storage = storage_write_stats_by_cycle(run)
    per_cycle = [
        sum(agent_data.get('chars', 0) for agent_data in cycle_data.values())
        for cycle_data in storage.values()
    ]
    if not per_cycle:
        return 0.0, 0.0
    mean = sum(per_cycle) / len(per_cycle)
    variance = sum((x - mean) ** 2 for x in per_cycle) / len(per_cycle)
    return round(mean, 1), round(math.sqrt(variance), 1)


def get_total_tool_calls(run: SwarmRun) -> int:
    """Sum all tool calls across all cycles and agents."""
    calls = tool_calls_from_turns_by_cycle(run)
    return sum(sum(agents.values()) for agents in calls.values())


def get_agents_string(run: SwarmRun) -> str:
    """Get comma-separated agent IDs with models."""
    agents = run.agents
    parts = []
    for a in agents:
        agent_id = a.get('id', 'unknown')
        model = a.get('model', 'unknown')
        # Shorten model name
        if '/' in model:
            model = model.split('/')[-1]
        parts.append(f"{agent_id}({model})")
    return ', '.join(parts)


def parse_storage_analysis(md_path: Path) -> dict:
    """Extract broad category, archetype, and dominant vibe from analysis markdown."""
    result = {
        'broad_category': '',
        'specific_archetype': '',
        'dominant_vibe': '',
    }

    if not md_path.exists():
        return result

    text = md_path.read_text(encoding='utf-8')

    # Match **Broad Category:** **Value** or *Broad Category:* **Value**
    broad_match = re.search(r'\*{1,2}Broad Category[:\*]{1,3}\s*\*{0,2}([^*\n]+)', text)
    if broad_match:
        result['broad_category'] = broad_match.group(1).strip().strip('*').strip()

    # Match **Specific Archetype:** **Value**
    arch_match = re.search(r'\*{1,2}Specific Archetype[:\*]{1,3}\s*\*{0,2}([^*\n]+)', text)
    if arch_match:
        result['specific_archetype'] = arch_match.group(1).strip().strip('*').strip()

    # Match **Dominant Vibe:** followed by text until end of line or next bullet
    vibe_match = re.search(r'\*{1,2}Dominant Vibe[:\*]{1,3}\s*(.+?)(?:\n\n|\n\*|\Z)', text, re.DOTALL)
    if vibe_match:
        vibe = vibe_match.group(1).strip()
        # Clean up any trailing markdown
        vibe = re.sub(r'\s+', ' ', vibe)  # Normalize whitespace
        result['dominant_vibe'] = vibe.strip()

    return result


def find_analysis_file(json_path: Path) -> Optional[Path]:
    """Find the storage_analysis.md file, checking multiple locations."""
    stem = json_path.stem
    analysis_name = f"{stem}_storage_raw_analysis.md"

    # Check same directory as JSON
    local_path = json_path.with_name(analysis_name)
    if local_path.exists():
        return local_path

    # Check analysis/ directory relative to project root
    project_root = json_path.parent.parent
    analysis_dir = project_root / 'analysis'
    if analysis_dir.exists():
        analysis_path = analysis_dir / analysis_name
        if analysis_path.exists():
            return analysis_path

    # Check analysis_short_runs/ directory
    short_runs_dir = project_root / 'analysis_short_runs'
    if short_runs_dir.exists():
        analysis_path = short_runs_dir / analysis_name
        if analysis_path.exists():
            return analysis_path

    return None



def process_run(json_path: Path) -> dict:
    """Process a single run and return summary dict."""
    run = SwarmRun.load(json_path)

    # Basic info
    run_name = json_path.stem
    combination = agent_combination(run)
    agents_str = get_agents_string(run)
    total_words = get_total_words(run)
    total_errors = get_total_llm_errors(run)
    run_time = get_run_time_seconds(run)
    total_cycles = run.total_cycles

    # Storage chars stats
    storage_chars_mean, storage_chars_std = get_storage_chars_stats(run)

    # Per-agent character counts
    chars_per_agent = get_chars_per_agent(run)
    agent_ids = run.agent_ids  # Ordered list of agent IDs

    # Check for storage analysis file in multiple locations
    analysis_path = find_analysis_file(json_path)
    analysis = parse_storage_analysis(analysis_path) if analysis_path else {
        'broad_category': '',
        'specific_archetype': '',
        'dominant_vibe': '',
    }

    result = {
        'run_name': run_name,
        'combination': combination,
        'agents': agents_str,
        'cycles': total_cycles,
        'total_words': total_words,
        'total_tokens': get_total_tokens(run),
        'total_tool_calls': get_total_tool_calls(run),
        'llm_errors': total_errors,
        'run_time_sec': round(run_time, 1),
        'storage_chars_mean': storage_chars_mean,
        'storage_chars_std': storage_chars_std,
        'has_analysis': analysis_path is not None,
        'broad_category': analysis['broad_category'],
        'specific_archetype': analysis['specific_archetype'],
        'dominant_vibe': analysis['dominant_vibe'],
    }

    # Add per-agent character columns (agent_1_chars, agent_2_chars, etc.)
    for i, agent_id in enumerate(agent_ids, start=1):
        result[f'agent_{i}_chars'] = chars_per_agent.get(agent_id, 0)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate summary CSV of swarm runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'json_files',
        nargs='+',
        help='JSON files to process'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file (default: stdout)'
    )
    args = parser.parse_args()

    # Look for vocab.txt in the same directory as the first JSON file

    # Process all files
    results = []
    for json_file in args.json_files:
        path = Path(json_file)
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        if not path.suffix == '.json':
            print(f"Warning: {path} is not a JSON file, skipping", file=sys.stderr)
            continue

        try:
            result = process_run(path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            continue

    if not results:
        print("No results to output", file=sys.stderr)
        sys.exit(1)

    # Find max number of agents across all runs
    max_agents = 0
    for r in results:
        agent_cols = [k for k in r.keys() if k.startswith('agent_') and k.endswith('_chars')]
        max_agents = max(max_agents, len(agent_cols))

    # Build fieldnames with per-agent columns
    fieldnames = [
        'run_name', 'combination', 'agents', 'cycles', 'total_words',
        'total_tokens', 'total_tool_calls',
        'llm_errors', 'run_time_sec',
        'storage_chars_mean', 'storage_chars_std',
    ]
    # Add per-agent char columns
    for i in range(1, max_agents + 1):
        fieldnames.append(f'agent_{i}_chars')

    fieldnames.extend([
        'has_analysis', 'broad_category', 'specific_archetype', 'dominant_vibe'
    ])

    # Ensure all results have all agent columns (fill missing with 0)
    for r in results:
        for i in range(1, max_agents + 1):
            col = f'agent_{i}_chars'
            if col not in r:
                r[col] = 0

    if args.output:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} rows to {args.output}", file=sys.stderr)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        print(f"\nProcessed {len(results)} runs", file=sys.stderr)


if __name__ == '__main__':
    main()
