"""A deep agent that plans a multi-step browsing task and delegates each page to
`browse_fast`.

The deep agent's model decides *when* to browse and *what page-scoped goal* to give it;
it never decides individual clicks or keystrokes — those stay inside `browse_fast`'s
fast, TypeSafe-classified step loop.

Usage:
    uv run --env-file .env python examples/deep_agent.py
"""

from __future__ import annotations

from deepagents import create_deep_agent

from ts_browser_agent import make_browse_fast_tool

_SYSTEM_PROMPT = (
    "You plan and verify browsing tasks. Delegate every page interaction to "
    "`browse_fast` with one concrete, page-scoped goal per call. Its status report is "
    "the only evidence a step worked; do not assume success beyond what it says. If a "
    "call reports `blocked` or `stalled`, change the goal or starting page rather than "
    "repeating the same call."
)


def main() -> None:
    browse_fast = make_browse_fast_tool(headless=False)
    agent = create_deep_agent(
        model="openai:gpt-5.5",
        tools=[browse_fast],
        system_prompt=_SYSTEM_PROMPT,
    )
    goal = "Go to https://en.wikipedia.org/wiki/Main_Page and open today's featured article."
    result = agent.invoke({"messages": [("user", goal)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
