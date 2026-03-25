#!/usr/bin/env python3
"""
Generate a markdown report combining summary, stats, and analysis.

Creates <stem>_report.md containing:
- Overview from summary.csv
- Per-cycle stats table from <stem>_stats.csv
- Cultural analysis from <stem>_storage_analysis.md

With -n flag, creates <stem>_report_full.md also containing:
- Narrative analysis from <stem>_storage_narrative.md

Usage:
    python scripts/generate_report.py swarm_*.json
    python scripts/generate_report.py swarm_*.json --summary summary.csv
    python scripts/generate_report.py swarm_*.json -n  # includes narrative
"""

import argparse
import csv
import sys
from pathlib import Path


def read_summary_row(summary_path: Path, run_name: str) -> dict:
    """Read the row for this run from summary.csv."""
    if not summary_path.exists():
        return {}

    with open(summary_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('run_name') == run_name:
                return row
    return {}


def read_stats_csv(stats_path: Path) -> tuple[list, list]:
    """Read stats CSV and return (headers, rows)."""
    if not stats_path.exists():
        return [], []

    with open(stats_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows


def format_stats_table(headers: list, rows: list) -> str:
    """Format stats as a markdown table."""
    if not headers or not rows:
        return "*No stats data available.*\n"

    # Calculate column widths
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ''))))

    # Build table
    lines = []

    # Header row
    header_cells = [h.ljust(widths[h]) for h in headers]
    lines.append('| ' + ' | '.join(header_cells) + ' |')

    # Separator row
    sep_cells = ['-' * widths[h] for h in headers]
    lines.append('| ' + ' | '.join(sep_cells) + ' |')

    # Data rows
    for row in rows:
        cells = [str(row.get(h, '')).ljust(widths[h]) for h in headers]
        lines.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines) + '\n'


def read_analysis(analysis_path: Path) -> str:
    """Read the storage analysis markdown."""
    if not analysis_path.exists():
        return "*No cultural analysis available.*\n"

    return analysis_path.read_text(encoding='utf-8')


def format_overview(summary: dict) -> str:
    """Format the overview section from summary data."""
    if not summary:
        return "*No summary data available.*\n"

    lines = []

    # Key metrics in a readable format
    fields = [
        ('combination', 'Model Combination'),
        ('agents', 'Agents'),
        ('cycles', 'Total Cycles'),
        ('total_words', 'Total Words'),
        ('total_tokens', 'Total Tokens'),
        ('total_tool_calls', 'Total Tool Calls'),
        ('llm_errors', 'LLM Errors'),
        ('run_time_sec', 'Run Time (seconds)'),
        ('broad_category', 'Broad Category'),
        ('specific_archetype', 'Specific Archetype'),
    ]

    for key, label in fields:
        if key in summary and summary[key]:
            lines.append(f"- **{label}:** {summary[key]}")


    return '\n'.join(lines) + '\n'


def generate_report(json_path: Path, summary_path: Path, include_narrative: bool = False) -> None:
    """Generate the report markdown file."""
    stem = json_path.stem
    run_name = stem

    # Paths to related files
    analysis_path = json_path.with_stem(stem + '_storage_raw_analysis').with_suffix('.md')
    narrative_path = json_path.with_stem(stem + '_storage_raw_narrative').with_suffix('.md')
    output_suffix = '_report_full' if include_narrative else '_report'
    output_path = json_path.with_stem(stem + output_suffix).with_suffix('.md')

    # Gather content
    summary = read_summary_row(summary_path, run_name)
    analysis = read_analysis(analysis_path)

    # Build report
    lines = []

    # Title
    lines.append(f"# Report on run {run_name}")
    lines.append('')

    # Overview section
    lines.append('## Overview')
    lines.append('')
    lines.append(format_overview(summary))
    lines.append('')

    # Per-cycle section
    # lines.append('## Per Cycle Statistics')
    # lines.append('')
    # lines.append(format_stats_table(stats_headers, stats_rows))
    # lines.append('')

    # Cultural analysis section
    lines.append('## Cultural Analysis')
    lines.append('')
    lines.append(analysis)

    # Narrative analysis section (optional)
    if include_narrative:
        lines.append('')
        lines.append('## Narrative Analysis')
        lines.append('')
        lines.append(read_analysis(narrative_path))

    # Write report
    report_content = '\n'.join(lines)
    output_path.write_text(report_content, encoding='utf-8')
    print(f"Written {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate markdown report from swarm run data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'json_files',
        nargs='+',
        help='JSON files to process'
    )
    parser.add_argument(
        '-s', '--summary',
        default='summary.csv',
        help='Path to summary.csv (default: summary.csv in current directory)'
    )
    parser.add_argument(
        '-n', '--narrative',
        action='store_true',
        help='Include narrative analysis section (outputs to <stem>_report_full.md)'
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)

    for json_file in args.json_files:
        path = Path(json_file)
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        if path.suffix != '.json':
            print(f"Warning: {path} is not a JSON file, skipping", file=sys.stderr)
            continue

        try:
            generate_report(path, summary_path, args.narrative)
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            continue


if __name__ == '__main__':
    main()
