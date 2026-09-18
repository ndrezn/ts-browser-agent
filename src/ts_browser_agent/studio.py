"""Expose the deep agent as a LangGraph graph for LangSmith Studio.

`langgraph dev` loads `graph` from here (see `langgraph.json` at the project root), so
goals can be typed into the Studio UI instead of hardcoded in an example script. Same
shape as the examples: the deep agent plans and delegates each page to `browse_fast`.

Usage:
    uv run langgraph dev
"""

from __future__ import annotations

from datetime import UTC, datetime

from deepagents import create_deep_agent

from ts_browser_agent.tool import make_browse_fast_tool

_SYSTEM_PROMPT = (
    f"Today's date is {datetime.now(tz=UTC).date().isoformat()}. You accomplish web tasks "
    "by delegating each page interaction to `browse_fast` with one concrete, page-scoped "
    "goal per call. Its classifier matches your goal text against what is on the page; it "
    "cannot compute relative dates or interpret vague places. Resolve any date to "
    "YYYY-MM-DD and any location to a specific city or airport yourself before sending "
    "a goal — a field typed with 'anywhere warm' matches no suggestion and the call will "
    "only stall. Each call's report includes the visible text of the page it ended on; "
    "read results from that text, and never assume a call found something just because "
    "it reports `done`. A `blocked` or `stalled` status needs a different goal or "
    "starting point, not the same call again. `browse_fast` only navigates and searches; "
    "it never purchases, submits payment, or enters personal information."
)

graph = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[make_browse_fast_tool(headless=False)],
    system_prompt=_SYSTEM_PROMPT,
)

__all__ = ["graph"]
