#!/usr/bin/env python3
"""
Example: using llmbox_lib for process automation.

This script demonstrates running an agent programmatically to:
1. Analyze a directory of log files for errors
2. Generate a summary report
3. Optionally run a fix command if issues are found

Usage:
    python examples/process_automation.py /var/log/myapp
    python examples/process_automation.py .  --fix
"""

import argparse
import sys
import os

# Add parent directory to path so we can import llmbox_lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmbox_lib import Agent


def on_tool(name, args):
    """Log each tool call as it happens."""
    args_short = {k: (v[:60] + "...") if isinstance(v, str) and len(v) > 60 else v
                  for k, v in args.items()}
    print(f"  [tool] {name}({args_short})")


def on_turn(turn_num, result):
    """Log progress after each turn."""
    tool_count = len(result.tool_results)
    errors = sum(1 for t in result.tool_results if t.is_error)
    print(f"  [turn {turn_num}] {tool_count} tool calls, {errors} errors")


def analyze_logs(target_dir, apply_fix=False):
    agent = Agent(
        config={
            "cycle": {"max_turns": 20},
        },
        system_prompt=(
            "You are a log analysis assistant. Be concise and actionable. "
            "When analyzing logs, focus on errors, warnings, and patterns. "
            "Output a structured report with: ERRORS, WARNINGS, RECOMMENDATIONS."
        ),
        on_tool=on_tool,
        on_turn=on_turn,
    )

    if not agent.health():
        print("ERROR: API is not reachable. Check BEDROCK_API_URL and BEDROCK_API_KEY.")
        sys.exit(1)

    print(f"Analyzing logs in: {target_dir}")
    print(f"Model: {agent.model}")
    print()

    # Step 1: Analyze
    result = agent.run(
        f"List the files in '{target_dir}', then read any log files and analyze them "
        f"for errors, warnings, and unusual patterns. Give me a structured report.",
        max_turns=15,
    )

    print("\n" + "=" * 60)
    print("ANALYSIS REPORT")
    print("=" * 60)
    print(result.text)
    print(f"\n[completed in {result.total_turns} turns, status: {result.status}]")

    # Step 2: Optionally run a fix
    if apply_fix and "error" in result.text.lower():
        print("\n" + "=" * 60)
        print("APPLYING FIXES")
        print("=" * 60)

        fix_result = agent.run(
            "Based on the errors you found, suggest and apply safe fixes. "
            "Do NOT delete any files. Only fix configuration issues or permissions.",
            max_turns=10,
        )
        print(fix_result.text)
        print(f"\n[completed in {fix_result.total_turns} turns, status: {fix_result.status}]")

    return result


def batch_process(directories):
    """Run analysis across multiple directories, reusing the same agent."""
    agent = Agent(
        config={"cycle": {"max_turns": 10}},
        system_prompt="You are a log analysis assistant. Be concise.",
        on_tool=on_tool,
    )

    results = {}
    for d in directories:
        print(f"\n--- Processing: {d} ---")
        agent.reset()  # fresh conversation for each directory
        result = agent.run(f"Analyze log files in '{d}' for errors. Brief summary only.")
        results[d] = result
        print(f"  Status: {result.status} ({result.total_turns} turns)")
        print(f"  Summary: {result.text[:200]}")

    return results


def simple_question():
    """Simplest possible usage — one prompt, one answer."""
    agent = Agent()
    result = agent.run("What files are in the current directory?", max_turns=5)
    print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze logs with llmbox agent")
    parser.add_argument("directory", nargs="?", default=".",
                        help="Directory to analyze (default: current)")
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to apply safe fixes for issues found")
    args = parser.parse_args()

    analyze_logs(args.directory, apply_fix=args.fix)
