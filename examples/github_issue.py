"""Drill into a labeled issue on GitHub, several clicks through nested navigation.

Repo -> Issues tab -> label filter -> a specific issue: each step's candidate set
looks completely different (top nav, a filter dropdown, a results list), a good
exercise for the per-step element table rather than one repeated UI shape.

Usage:
    uv run --env-file .env python examples/github_issue.py
"""

from __future__ import annotations

from ts_browser_agent import Agent

_GOAL = (
    "Open the Issues tab for this repository, filter the issue list to issues "
    "labeled 'good first issue', and open the first matching issue. Stop once that "
    "issue's page is visible."
)


def main() -> None:
    with Agent("https://github.com/langchain-ai/langchain", _GOAL, max_steps=40) as agent:
        for state in agent.run():
            print(f"{state['elapsed_ms']:>6} ms  {state['operation']:<10} {state['target'] or ''}")
        print(f"Final status: {agent.status}")


if __name__ == "__main__":
    main()
