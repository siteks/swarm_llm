#!/usr/bin/env python3
"""
Swarm LLM - Emergent multi-agent coordination system.

Run a swarm of autonomous agents that coordinate through
shared primitives (messaging, storage) with minimal per-turn context.

Usage:
    python run_swarm.py [options]

Agent Specs:
    Agents are specified as MODEL:PROMPT or just PROMPT.
    - PROMPT alone uses the default model (from LLM_MODEL env var)
    - MODEL:PROMPT uses a specific model for that agent

    Available prompts: curious_explorer, coordinator, specialist, observer, minimal

    Model nicknames (claude, gemini, opus, etc.) are resolved via MODEL_* env
    vars in .env (e.g., MODEL_CLAUDE=claude-sonnet-4-5-20250929)

Examples:
    # Run with default agents
    python run_swarm.py

    # Run for 20 cycles
    python run_swarm.py --cycles 20

    # Run with specific agents (using default model)
    python run_swarm.py -a curious_explorer observer specialist

    # Run with specific models per agent (model:prompt format)
    python run_swarm.py -a claude:minimal gemini:minimal

    # Mix of models and default
    python run_swarm.py -a opus:coordinator claude:observer minimal

    # Resume from a previous run
    python run_swarm.py --resume output/swarm_20250116_120000.json

    # Resume and run more cycles
    python run_swarm.py -r output/swarm_20250116.json -c 10
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from config import SwarmConfig, AGENT_PROMPTS, resolve_model, get_default_model
from core.llm_client import LLMClientPool
from swarm.runtime import SwarmRuntime, CycleResult


def setup_logging(output_dir: Path, verbose: bool = False, quiet: bool = False):
    """
    Configure logging to both console and file.

    Args:
        output_dir: Directory for log files
        verbose: Enable debug logging
        quiet: Only show warnings and errors
    """
    # Suppress noisy third-party loggers
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Determine log level
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler - always log at DEBUG level
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"swarm_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    return log_file


def get_output_dir() -> Path:
    """Get output directory from environment or default."""
    output_dir = Path(os.getenv("SWARM_OUTPUT_DIR", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_agent_spec(spec: str) -> Tuple[str, str]:
    """
    Parse agent specification into (model, prompt_name).

    Format: 'model_nickname:prompt_name' or just 'prompt_name'
    If no model specified, uses default from LLM_MODEL env var.

    Examples:
        'claude:minimal' -> (resolved_claude_model, 'minimal')
        'gemini:coordinator' -> (resolved_gemini_model, 'coordinator')
        'minimal' -> (default_model, 'minimal')
    """
    if ':' in spec:
        model_nick, prompt_name = spec.split(':', 1)
        return (resolve_model(model_nick), prompt_name)
    return (get_default_model(), spec)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an emergent swarm of LLM agents"
    )

    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=None,
        help="Maximum number of cycles to run (default: from config)"
    )

    parser.add_argument(
        "--agents", "-a",
        nargs="+",
        default=None,
        metavar="SPEC",
        help="Agent specs as MODEL:PROMPT or PROMPT. MODEL uses nicknames from "
             "MODEL_* env vars. Available prompts: curious_explorer, coordinator, "
             "specialist, observer, minimal. Examples: claude:minimal, gemini:coordinator, minimal. "
             "Cannot be used with --resume (agents are restored from JSON)."
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file prefix (default: swarm_TIMESTAMP)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from SWARM_OUTPUT_DIR env or ./output)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show cycle summaries"
    )

    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Show real-time agent activity (tool calls and messages)"
    )

    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        metavar="FILE",
        help="Resume from a previous run's JSON state file. Restores agents, "
             "storage, messages, and telemetry. Cycle numbers continue from "
             "where the previous run left off. Example: output/swarm_20250116.json"
    )

    # Entropy settings
    parser.add_argument(
        "--entropy-tipping-point",
        type=float,
        default=None,
        metavar="CYCLES",
        help="Cycles before storage decay accelerates (default: 15)"
    )

    parser.add_argument(
        "--entropy-steepness",
        type=float,
        default=None,
        metavar="VALUE",
        help="Decay curve sharpness - higher = sharper (default: 0.5)"
    )

    parser.add_argument(
        "--no-entropy",
        action="store_true",
        help="Disable storage entropy (text decay)"
    )

    parser.add_argument(
        "--char-budget", "-b",
        type=int,
        default=None,
        metavar="CHARS",
        help="Character budget for write ops per agent turn (default: from SWARM_CHAR_BUDGET env var, unlimited if unset)"
    )

    return parser.parse_args()


# ANSI colors for watch mode
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
}

# Color palette for agents (bright, distinguishable colors)
AGENT_PALETTE = [
    "\033[36m",  # cyan
    "\033[33m",  # yellow
    "\033[35m",  # magenta
    "\033[34m",  # blue
    "\033[32m",  # green
    "\033[91m",  # bright red
    "\033[96m",  # bright cyan
    "\033[93m",  # bright yellow
    "\033[95m",  # bright magenta
    "\033[94m",  # bright blue
]

# Agent to color mapping
_agent_colors: Dict[str, str] = {}


def get_agent_color(agent_id: str) -> str:
    """Get a consistent color for an agent."""
    if agent_id not in _agent_colors:
        color_idx = len(_agent_colors) % len(AGENT_PALETTE)
        _agent_colors[agent_id] = AGENT_PALETTE[color_idx]
    return _agent_colors[agent_id]


def format_agent_id(agent_id: str) -> str:
    """Format agent ID with agent-specific color."""
    color = get_agent_color(agent_id)
    return f"{color}{COLORS['bold']}[{agent_id}]{COLORS['reset']}"


def format_wrapped_text(text: str, wrap_width: int = 100, indent: str = "    ") -> str:
    """Format text with word wrapping, respecting newlines."""
    lines = []
    for line in text.split('\n'):
        if line.strip():
            wrapped = textwrap.fill(
                line,
                width=wrap_width,
                initial_indent=indent,
                subsequent_indent=indent
            )
            lines.append(wrapped)
        else:
            lines.append("")  # Preserve blank lines
    return '\n'.join(lines)


def on_agent_thinking(agent_id: str, thinking: str):
    """Display agent thinking with word wrap."""
    if thinking.strip():
        print(f"{format_agent_id(agent_id)} {COLORS['dim']}(thinking){COLORS['reset']}")
        print(format_wrapped_text(thinking))


def on_agent_content(agent_id: str, content: str):
    """Display agent content output with word wrap."""
    if content.strip():
        print(format_agent_id(agent_id))
        print(format_wrapped_text(content))


def on_agent_tool_call(agent_id: str, tool_name: str, args: dict):
    """Display agent tool call (placeholder - actual output in on_agent_tool_result)."""
    pass


def on_agent_tool_result(agent_id: str, tool_name: str, args: dict, result: dict):
    """Display agent tool call and result on one line."""
    success = result.get("success", False)
    status = f"{COLORS['green']}OK{COLORS['reset']}" if success else f"{COLORS['red']}FAIL{COLORS['reset']}"

    # Build detail string based on tool type
    if not success:
        error = result.get("error", "unknown error")
        detail = f" - {error}"
    elif tool_name == "read_inbox":
        count = len(result.get("messages", []))
        detail = f" ({count} msgs)"
    elif tool_name == "send_message":
        to = args.get("to", "?")
        detail = f" -> {to}"
        # Add budget info if present
        if "budget_chars_used" in result:
            chars = result["budget_chars_used"]
            remaining = result["budget_chars_remaining"]
            detail += f" chars={chars} remaining={remaining}"
    elif tool_name == "list_agents":
        agents = result.get("agents", [])
        detail = f" ({len(agents)} agents)"
    elif tool_name == "list_storage_keys":
        keys = result.get("keys", [])
        detail = f" ({len(keys)} keys)"
    elif tool_name == "read_storage":
        key = args.get("key", "?")
        detail = f" key={key}"
    elif tool_name == "write_storage":
        key = args.get("key", "?")
        detail = f" key={key}"
        # Add budget info if present
        if "budget_chars_used" in result:
            chars = result["budget_chars_used"]
            remaining = result["budget_chars_remaining"]
            detail += f" chars={chars} remaining={remaining}"
    elif tool_name == "append_storage":
        key = args.get("key", "?")
        detail = f" key={key}"
        # Add budget info if present
        if "budget_chars_used" in result:
            chars = result["budget_chars_used"]
            remaining = result["budget_chars_remaining"]
            detail += f" chars={chars} remaining={remaining}"
    elif tool_name == "get_my_id":
        detail = f" -> {result.get('agent_id', '?')}"
    else:
        detail = ""

    print(f"{format_agent_id(agent_id)} {tool_name}: {status}{detail}")

    # For send_message, show the full message content with word wrap
    if tool_name == "send_message" and success:
        content = args.get("content", "")
        if content:
            print(format_wrapped_text(content))

    # For write_storage/append_storage, show the value with word wrap
    if tool_name in ("write_storage", "append_storage") and success:
        value = args.get("value", "")
        if value:
            # Handle non-string values (e.g., lists)
            if not isinstance(value, str):
                value = json.dumps(value)
            print(format_wrapped_text(value))


def on_agent_turn_end(agent_id: str):
    """Display when an agent finishes its turn."""
    print(f"{format_agent_id(agent_id)} {COLORS['dim']}-- end turn --{COLORS['reset']}")


def on_cycle_complete(result: CycleResult):
    """Callback after each cycle completes."""
    print(
        f"\n[Cycle {result.cycle}] "
        f"{result.duration_ms:.0f}ms | "
        f"agents: {result.agents_active} | "
        f"tools: {result.total_tool_calls} | "
        f"msgs: {result.messages_sent} | "
        f"tokens: {result.total_tokens}"
    )

    # Show per-agent summaries
    for agent_result in result.agent_results:
        status = "OK" if agent_result.success else "FAIL"
        print(
            f"  [{agent_result.agent_id}] {status} - "
            f"{agent_result.tool_calls} tools, "
            f"{agent_result.tokens_used} tokens"
        )


async def run_swarm(args, output_dir: Path, logger: logging.Logger):
    """Run the swarm with given configuration."""
    config = SwarmConfig.from_env()

    # Override config with CLI args
    if args.cycles:
        config.max_cycles = args.cycles

    # Create LLM client pool
    llm_pool = LLMClientPool()

    # Calculate entropy settings
    entropy_enabled = not args.no_entropy and config.entropy_enabled
    entropy_tipping_point = args.entropy_tipping_point if args.entropy_tipping_point is not None else config.entropy_tipping_point
    entropy_steepness = args.entropy_steepness if args.entropy_steepness is not None else config.entropy_steepness

    # Character budget (CLI overrides env var)
    char_budget = args.char_budget if args.char_budget is not None else config.char_budget_per_turn

    # Check if resuming from a previous run
    if args.resume:
        # Agents must come from JSON when resuming - reject -a flag
        if args.agents is not None:
            print("Error: Cannot specify --agents/-a when resuming. Agents are restored from the JSON file.")
            logger.error("Cannot specify --agents when resuming")
            return

        print(f"\n=== Resuming from: {args.resume} ===")
        logger.info(f"Resuming swarm from: {args.resume}")

        try:
            runtime = SwarmRuntime.restore_from_file(
                json_path=args.resume,
                llm_pool=llm_pool,
                max_concurrent_agents=config.max_concurrent_agents,
                max_tool_iterations=config.max_tool_iterations,
                char_budget_per_turn=char_budget,
                entropy_enabled=entropy_enabled,
                entropy_tipping_point=entropy_tipping_point,
                entropy_steepness=entropy_steepness
            )
            print(f"Restored from cycle {runtime._current_cycle}")
            restored_agents = runtime.list_agents()
            print(f"Restored agents: {[a.id for a in restored_agents]}")
            for agent in restored_agents:
                model_short = agent.model.split('/')[-1][:20]
                print(f"  {agent.id} ({model_short})")

        except (ValueError, FileNotFoundError) as e:
            print(f"Error resuming: {e}")
            logger.error(f"Failed to resume: {e}")
            return

    else:
        # Use default agents if none specified
        if args.agents is None:
            args.agents = ["curious_explorer", "coordinator"]
        # Create fresh runtime
        runtime = SwarmRuntime(
            llm_pool=llm_pool,
            max_concurrent_agents=config.max_concurrent_agents,
            max_tool_iterations=config.max_tool_iterations,
            char_budget_per_turn=char_budget,
            entropy_enabled=entropy_enabled,
            entropy_tipping_point=entropy_tipping_point,
            entropy_steepness=entropy_steepness
        )

        # Spawn agents with sequential minimal_N naming
        print("\n=== Spawning Agents ===")
        agent_count = 0
        for agent_spec in args.agents:
            model, prompt_name = parse_agent_spec(agent_spec)

            if prompt_name not in AGENT_PROMPTS:
                logger.warning(f"Unknown prompt type: {prompt_name}")
                continue

            # Simple sequential numbering for all agents
            agent_count += 1
            agent_id = f"minimal_{agent_count}"

            agent = runtime.spawn_agent(
                system_prompt=AGENT_PROMPTS[prompt_name],
                model=model,
                agent_id=agent_id
            )
            # Show short model name for display
            model_short = model.split('/')[-1][:20]
            print(f"  Spawned: {agent.id} ({model_short})")
            logger.info(f"Spawned agent: {agent.id} with model: {model}")

    # Set up watch mode callbacks
    if args.watch:
        runtime.set_activity_callbacks(
            on_content=on_agent_content,
            on_thinking=on_agent_thinking,
            on_tool_call=on_agent_tool_call,
            on_tool_result=on_agent_tool_result,
            on_turn_end=on_agent_turn_end
        )
        print("\nWatch mode enabled - showing real-time agent activity\n")

    if not runtime.get_active_agents():
        print("No agents spawned. Exiting.")
        return

    # Set up signal handler for graceful stop
    def signal_handler(sig, frame):
        print("\n\nStop requested (Ctrl+C). Finishing current cycle...")
        logger.info("Stop requested via signal")
        runtime.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Run the swarm
    if args.resume:
        print(f"\n=== Resuming Swarm ===")
        print(f"Starting from cycle: {runtime._current_cycle}")
    else:
        print(f"\n=== Starting Swarm ===")
    print(f"Max cycles: {config.max_cycles}")
    print(f"Active agents: {len(runtime.get_active_agents())}")
    print(f"Output directory: {output_dir}")
    print(f"Press Ctrl+C to stop gracefully\n")

    logger.info(f"{'Resuming' if args.resume else 'Starting'} swarm: max_cycles={config.max_cycles}, agents={len(runtime.get_active_agents())}, current_cycle={runtime._current_cycle}")

    start_time = datetime.now()

    result = await runtime.run(
        max_cycles=config.max_cycles,
        on_cycle_complete=on_cycle_complete if not args.quiet else None
    )

    # Generate output prefix
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_prefix = args.output or f"swarm_{timestamp}"
    output_base = output_dir / output_prefix

    # Print summary
    print("\n" + "=" * 50)
    print("SWARM EXECUTION COMPLETE")
    print("=" * 50)
    print(f"Total cycles: {result.total_cycles}")
    print(f"Total duration: {result.duration_s:.1f}s")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total tool calls: {result.total_tool_calls}")
    print(f"Total messages: {result.total_messages}")
    print(f"Stop reason: {result.stop_reason}")

    logger.info(f"Swarm complete: cycles={result.total_cycles}, tokens={result.total_tokens}, reason={result.stop_reason}")

    # Export telemetry
    json_path = f"{output_base}.json"
    graphml_path = f"{output_base}.graphml"

    runtime.export_telemetry_json(json_path)
    runtime.export_telemetry_graphml(graphml_path)

    print(f"\nTelemetry exported:")
    print(f"  JSON: {json_path}")
    print(f"  GraphML: {graphml_path}")

    logger.info(f"Telemetry exported: {json_path}, {graphml_path}")

    # Show storage summary
    print("\n=== Final Storage State ===")
    storage_snapshot = runtime.storage.get_snapshot()
    for key, value in storage_snapshot.items():
        if not key.startswith("_"):
            value_preview = str(value)[:100]
            if len(str(value)) > 100:
                value_preview += "..."
            print(f"  {key}: {value_preview}")

    # Show message graph
    print("\n=== Message Flow ===")
    msg_graph = runtime.messaging.get_message_graph()
    for edge in msg_graph["edges"]:
        print(f"  {edge['from']} -> {edge['to']}: {edge['count']} messages")

    # Write summary to file
    summary_path = f"{output_base}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Swarm Execution Summary\n")
        f.write(f"=======================\n\n")
        if args.resume:
            f.write(f"Resumed from: {args.resume}\n")
        f.write(f"Start time: {start_time.isoformat()}\n")
        f.write(f"End time: {datetime.now().isoformat()}\n")
        f.write(f"Models: {', '.join(llm_pool.get_models())}\n")
        agent_ids = [a.id for a in runtime.list_agents()]
        f.write(f"Agents: {', '.join(agent_ids)}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total cycles: {result.total_cycles}\n")
        f.write(f"  Total duration: {result.duration_s:.1f}s\n")
        f.write(f"  Total tokens: {result.total_tokens}\n")
        f.write(f"  Total tool calls: {result.total_tool_calls}\n")
        f.write(f"  Total messages: {result.total_messages}\n")
        f.write(f"  Stop reason: {result.stop_reason}\n\n")
        f.write(f"Storage Keys:\n")
        for key in storage_snapshot.keys():
            if not key.startswith("_"):
                f.write(f"  - {key}\n")
        f.write(f"\nMessage Flow:\n")
        for edge in msg_graph["edges"]:
            f.write(f"  {edge['from']} -> {edge['to']}: {edge['count']} messages\n")

    print(f"\nSummary saved: {summary_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_output_dir()

    # Set up logging
    log_file = setup_logging(output_dir, args.verbose, args.quiet)
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 50)
    print("SWARM LLM - Emergent Multi-Agent Coordination")
    print("=" * 50)
    print(f"Log file: {log_file}")

    logger.info("=" * 50)
    logger.info("SWARM LLM Starting")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Output directory: {output_dir}")

    try:
        asyncio.run(run_swarm(args, output_dir, logger))
    except KeyboardInterrupt:
        print("\nExecution interrupted.")
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Swarm execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
