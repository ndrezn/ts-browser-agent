"""Drill into a labeled issue on GitHub, several clicks through nested navigation.

Repo -> Issues tab -> label filter -> a specific issue: each step's candidate set looks
completely different (top nav, a filter dropdown, a results list), a good exercise for
the per-step element table rather than one repeated UI shape.

Usage:
    uv run --env-file .env python examples/github_issue.py
"""

from __future__ import annotations

import asyncio

from _trace import run_and_print

from ts_browser_agent import build_browser_agent

_GOAL = (
    "Open the Issues tab of this repository, filter the issue list to issues labeled "
    "'good first issue', and open the first matching issue. Stop once that issue's page "
    "is visible.\n\nStart at https://github.com/langchain-ai/langchain"
)


async def main() -> None:
    await run_and_print(build_browser_agent(), _GOAL)


if __name__ == "__main__":
    asyncio.run(main())
