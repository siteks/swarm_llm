#!/usr/bin/env python3
"""
Fiction generator — single-agent baseline for comparison with swarm outputs.

Repeatedly prompts an LLM to continue writing fiction, managing context
window limits by dropping oldest turns. No tools, no amnesia — just
iterative continuation.

Usage:
    python run_fiction.py --model claude
    python run_fiction.py --model gemini --chars 500000
    python run_fiction.py --model kimi --system "You are a novelist." --first "Write a dark fantasy story."
    python run_fiction.py --model claude -o custom_output.txt
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm

from config import resolve_model, get_default_model, load_model_nicknames
from core.llm_client import LLMClient


def resolve_model_and_nickname(model_arg: str | None) -> tuple[str, str]:
    """Resolve --model arg to (full_model_id, nickname).

    If model_arg is a known nickname (from MODEL_* env vars), use it directly.
    If model_arg is a full model ID, derive a short name from it.
    If model_arg is None, use the default model and nickname 'default'.
    """
    if model_arg is None:
        return get_default_model(), "default"

    nicknames = load_model_nicknames()
    if model_arg.lower() in nicknames:
        return nicknames[model_arg.lower()], model_arg.lower()

    # Not a known nickname — use as full model ID, derive a short name
    name = model_arg.split("/")[-1]
    # Drop date suffixes like -20250929
    parts = name.split("-")
    nick = "-".join(p for p in parts if not (len(p) == 8 and p.isdigit()))
    return model_arg, nick


def default_output_path(nickname: str) -> str:
    """Generate output/fiction_YYYYmmdd_HHMMss_<nickname>.txt"""
    output_dir = Path(os.getenv("SWARM_OUTPUT_DIR", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(output_dir / f"fiction_{timestamp}_{nickname}.txt")


def get_context_window(model: str) -> int:
    """Get the model's max input tokens from litellm model info."""
    try:
        info = litellm.get_model_info(model=model)
        return info.get("max_input_tokens") or info.get("max_tokens", 128000)
    except Exception:
        return 128000  # safe fallback


def trim_messages(messages, max_prompt_tokens, context_window):
    """Drop oldest user/assistant pairs (keep system prompt at index 0)
    until prompt_tokens is estimated to be under budget."""
    budget = int(context_window * max_prompt_tokens)
    # Each trim removes indices 1 and 2 (oldest user/assistant pair)
    while len(messages) > 3 and litellm.token_counter(model="gpt-4", messages=messages) > budget:
        del messages[1:3]
    return messages


async def run(args):
    model, nickname = resolve_model_and_nickname(args.model)
    output_path = args.output or default_output_path(nickname)
    context_window = get_context_window(model)

    client = LLMClient(model_override=model)
    if args.max_tokens:
        client.max_tokens = args.max_tokens

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.first},
    ]

    total_chars = 0
    turn = 0
    low_output_streak = 0
    LOW_OUTPUT_THRESHOLD = 100
    LOW_OUTPUT_MAX_STREAK = 5

    print(f"Model: {model}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    print(f"Context window: {context_window}", file=sys.stderr)
    print(f"Target chars: {args.chars}", file=sys.stderr)
    print(f"Max tokens/completion: {client.max_tokens}", file=sys.stderr)
    print(file=sys.stderr)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "w")

    try:
        while total_chars < args.chars:
            turn += 1
            content, usage = await client.simple_completion(messages)

            if not content:
                print(f"Turn {turn}: empty response, stopping.", file=sys.stderr)
                break

            total_chars += len(content)

            if len(content) < LOW_OUTPUT_THRESHOLD:
                low_output_streak += 1
                if low_output_streak >= LOW_OUTPUT_MAX_STREAK:
                    print(
                        f"Turn {turn}: {LOW_OUTPUT_MAX_STREAK} consecutive turns "
                        f"< {LOW_OUTPUT_THRESHOLD} chars, stopping.",
                        file=sys.stderr,
                    )
                    break
            else:
                low_output_streak = 0

            print(
                f"Turn {turn}: +{len(content)} chars, total={total_chars}, "
                f"prompt_tokens={usage.prompt_tokens}",
                file=sys.stderr,
            )

            if args.verbose:
                print(content[:200] + ("..." if len(content) > 200 else ""), file=sys.stderr)

            if turn > 1:
                out_file.write("\n\n")
            out_file.write(content)
            out_file.flush()

            # Build next turn
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": args.continue_prompt})

            # Trim if over context budget
            budget = int(args.context_fraction * context_window)
            if usage.prompt_tokens > budget:
                before = len(messages)
                messages = trim_messages(messages, args.context_fraction, context_window)
                after = len(messages)
                if before != after:
                    print(
                        f"  Trimmed {(before - after) // 2} turn(s) "
                        f"({before} -> {after} messages)",
                        file=sys.stderr,
                    )
    finally:
        out_file.close()

    print(f"\nDone: {turn} turns, {total_chars} chars -> {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate long-form fiction via iterative LLM continuation"
    )
    parser.add_argument("-o", "--output", default=None, help="Output text file (default: output/fiction_<timestamp>_<model>.txt)")
    parser.add_argument("--model", default=None, help="Model nickname or full ID")
    parser.add_argument("--chars", type=int, default=700000, help="Target total characters (default: 100000)")
    parser.add_argument("--system", default="You are a fiction writer. Write vivid, engaging prose.", help="System prompt")
    parser.add_argument("--first", default="Write a novel length story.", help="First user prompt")
    parser.add_argument("--continue-prompt", default="Continue your work.", help="Continuation prompt")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens per completion")
    parser.add_argument("--context-fraction", type=float, default=0.8, help="Fraction of context window before trimming (default: 0.8)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each turn's content preview to stderr")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
