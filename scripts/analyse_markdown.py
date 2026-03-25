#!/usr/bin/env python3
"""Analyze file sections using LLM completions."""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dotenv(env_path: Path) -> None:
    """Load environment variables from .env file.

    Args:
        env_path: Path to .env file

    Raises:
        FileNotFoundError: If .env file doesn't exist
    """
    if not env_path.exists():
        raise FileNotFoundError(
            f".env file not found at {env_path}\n"
            "Copy .env.example to .env and configure your API keys."
        )

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse KEY=value
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                # Remove surrounding quotes if present
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                os.environ[key] = value


# Load .env before importing modules that use environment variables
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / '.env')

from config import resolve_model
from core.llm_client import LLMClient


def parse_sections(content: str) -> list[tuple[str, str]]:
    """Parse file into (header, content) tuples."""
    pattern = r'=+\n([cC][yY][cC][lL][eE] \d+)\n=+\n'
    parts = re.split(pattern, content)
    # parts[0] is before first header (skip if empty)
    # then alternates: header, content, header, content...
    sections = []
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i+1] if i+1 < len(parts) else ""
        sections.append((header, body.strip()))
    return sections


def load_prompts(system_path: Path, user_path: Path) -> tuple[str, str]:
    """Load system and user prompts from files."""
    system_prompt = system_path.read_text().strip()
    user_template = user_path.read_text()

    # Validate user template has [DATA]...[/DATA] marker
    if '[DATA]' not in user_template or '[/DATA]' not in user_template:
        raise ValueError("User prompt must contain [DATA]...[/DATA] markers")
    # if '[RUN]' not in user_template or '[/RUN]' not in user_template:
    #     raise ValueError("User prompt must contain [RUN]...[/RUN] markers")

    return system_prompt, user_template


def build_user_prompt(template: str, run_name: str, content: str) -> str:
    """Insert content into user prompt template."""
    # Replace everything between [DATA] and [/DATA] with the content
    # Use lambda to avoid backslash interpretation in replacement string
    pattern = r'\[DATA\].*?\[/DATA\]'
    t = re.sub(pattern, lambda m: f'[DATA]\n{content}\n[/DATA]', template, flags=re.DOTALL)
    pattern = r'\[RUN\].*?\[/RUN\]'
    return re.sub(pattern, lambda m: run_name, t, flags=re.DOTALL)


async def analyze_section(client: LLMClient, system_prompt: str,
                          user_template: str, run_name: str, content: str,
                          verbose: bool = False) -> str:
    """Send section to LLM and return response."""
    user_prompt = build_user_prompt(user_template, run_name, content)

    if verbose:
        print(f"    Model: {client.model}")
        print(f"    System prompt: {system_prompt[:300]}{'...' if len(system_prompt) > 300 else ''}")
        print(f"    User prompt: {user_prompt[:2000]}{'...' if len(user_prompt) > 2000 else ''}")
        print(f"    User prompt length: {len(user_prompt)} chars")

    response, usage = await client.simple_completion(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt
    )

    if verbose:
        print(f"    Tokens: {usage.prompt_tokens} prompt, {usage.completion_tokens} completion")
        print(f"    Response length: {len(response)} chars")
        print(f"    Response preview: {response[:5000]}{'...' if len(response) > 5000 else ''}")

    return response


async def process_file(client: LLMClient, filepath: Path,
                       system_prompt: str, user_template: str,
                       split: bool = False, verbose: bool = False,
                       suffix: str = "_analysis") -> None:
    """Process a single file and write output."""
    content = filepath.read_text()

    # Extract a run name. Filename will always in the form swarm_20260119_145545_*
    fields = filepath.stem.split('_')
    if len(fields) > 1 and fields[0] == 'swarm':
        run_name = 'run_' + fields[1][4:] + '_' + fields[2][:4]
    else:
        run_name = 'synthesis'

    if split:
        sections = parse_sections(content)
        if not sections:
            print(f"  Warning: No sections found, analyzing entire file")
            sections = [("Full File", content.strip())]
    else:
        sections = [("Full File", content.strip())]

    results = []
    for header, body in sections:
        print(f"  Analyzing {header}...")
        analysis = await analyze_section(client, system_prompt, user_template, run_name, body, verbose)
        results.append((header, analysis))

    # Write output
    output_path = filepath.with_stem(filepath.stem + suffix).with_suffix(".md")
    with open(output_path, 'w') as f:
        for header, analysis in results:
            if split:
                f.write(f"===\n{header}\n===\n{analysis}\n\n")
            else:
                f.write(f"{analysis}\n\n")

    print(f"  Written to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description='Analyze file sections with LLM')
    parser.add_argument('files', nargs='+', help='Input files to process')
    parser.add_argument('-s', '--system', required=True,
                        help='Path to system prompt file')
    parser.add_argument('-u', '--user', required=True,
                        help='Path to user prompt file (must contain [DATA]...[/DATA])')
    parser.add_argument('-m', '--model',
                        help='Model nickname (e.g., claude, opus) or full model ID')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show model calls and response data')
    parser.add_argument('-sp', '--split', action='store_true',
                        help='Split file into sections (delimited by ===\\nCycle N\\n===)')
    parser.add_argument('--suffix', default='_analysis',
                        help='Output file suffix (default: _analysis)')
    args = parser.parse_args()

    # Load prompts from files
    system_prompt, user_template = load_prompts(Path(args.system), Path(args.user))

    # Resolve model nickname to full model ID
    model = resolve_model(args.model) if args.model else None
    client = LLMClient(model_override=model)

    for filepath in args.files:
        path = Path(filepath)
        print(f"Processing {path}...")
        await process_file(client, path, system_prompt, user_template, args.split, args.verbose, args.suffix)


if __name__ == '__main__':
    asyncio.run(main())
