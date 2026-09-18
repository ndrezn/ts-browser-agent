"""A deep agent that delegates page interaction to `browse_fast`.

The deep agent's model decides *when* to browse and *what page-scoped goal* to give;
it never decides individual clicks or keystrokes — those stay with the classifier
inside `browse_fast`.

Usage:
    uv run --env-file .env python examples/deep_agent.py
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent

from ts_browser_agent import make_browse_fast_tool
from ts_browser_agent.model import message_text

_SYSTEM_PROMPT = (
    "You plan and verify browsing tasks. Delegate every page interaction to "
    "`browse_fast` with one concrete, page-scoped goal per call. Its status report is "
    "the only evidence a step worked; do not assume success beyond what it says. If a "
    "call reports `blocked` or `stalled`, change the goal or starting page rather than "
    "repeating the same call."
)


async def main() -> None:
    agent = create_deep_agent(
        model="openai:gpt-5.5",
        tools=[make_browse_fast_tool(headless=False)],
        system_prompt=_SYSTEM_PROMPT,
    )
    goal = "Go to https://en.wikipedia.org/wiki/Main_Page and open today's featured article."
    result = await agent.ainvoke({"messages": [("user", goal)]})
    print(message_text(result["messages"][-1].content))


if __name__ == "__main__":
    asyncio.run(main())
