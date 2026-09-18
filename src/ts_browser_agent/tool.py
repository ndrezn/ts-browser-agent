"""Expose the browser agent as a tool for another agent to call.

A deep agent's own model already plans across many tool calls; it should not also decide
each individual click. `browse_fast` runs a whole browser agent inside one call: the
caller supplies a page and a page-scoped goal and gets back a status line plus the
visible text of the page the run ended on.

```python
from deepagents import create_deep_agent
from ts_browser_agent import make_browse_fast_tool

agent = create_deep_agent(model="openai:gpt-5.5", tools=[make_browse_fast_tool()])
```
"""

from __future__ import annotations

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ts_browser_agent.agent import build_browser_agent
from ts_browser_agent.model import message_text
from ts_browser_agent.snapshot import Snapshot
from ts_browser_agent.tools import BrowserSession

_MAX_REPORTED_TEXT = 3000

_TOOL_DESCRIPTION = (
    "Drive a live web page toward one concrete, page-scoped goal (click, type, select, "
    "or scroll) using a fast, classifier-driven step loop. Returns a status line plus "
    "the visible text of the page the run ended on, so you can read whatever result — "
    "a price, a title, a confirmation — the goal produced. Give it one goal per call; "
    "for a task spanning several pages or several independent searches, call it again "
    "with the next URL and the next goal. A `blocked` or `stalled` status means it "
    "could not make progress and needs a different goal or starting point, not a retry "
    "of the same call."
)


class BrowseFastInput(BaseModel):
    """Arguments the calling agent supplies to `browse_fast`."""

    url: str = Field(description="Page to start from.")
    goal: str = Field(
        description="One concrete, page-scoped goal, e.g. 'submit the search form for wireless mice'."
    )
    max_steps: int = Field(
        default=40, ge=1, le=200, description="Upper bound on attempts before giving up, at most 200."
    )


def _status(outcome: str) -> str:
    if outcome == "DONE":
        return "done"
    if outcome.startswith("STALLED"):
        return "stalled"
    return "blocked"


def _report(messages: list[BaseMessage]) -> str:
    outcome = next(
        (message_text(m.content) for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls),
        "",
    )
    snapshot = next(
        (m.artifact for m in reversed(messages) if isinstance(m, ToolMessage) and isinstance(m.artifact, Snapshot)),
        None,
    )
    status = f"status={_status(outcome)} outcome={outcome!r}"
    if snapshot is None:
        return status
    return f"{status} final_url={snapshot.url}\n\nVisible page text at the end of the run:\n{snapshot.text[:_MAX_REPORTED_TEXT]}"


def make_browse_fast_tool(
    *,
    text_model: str | BaseChatModel = "openai:gpt-5-mini",
    headless: bool = True,
    allow_private: bool = False,
) -> StructuredTool:
    """Build a `browse_fast` tool bound to one text model, headless setting, and URL policy.

    `allow_private` and `headless` are fixed when the tool is built, not exposed to the
    calling model — a prompt-injected page must not be able to talk the caller into
    loosening the URL policy through tool-call arguments. Each call builds its own
    browser agent, so parallel calls do not share a browser.

    Args:
        text_model: Chat model used only for `TYPE_TEXT` field values.
        headless: Whether the browser runs headless. Defaults to `True`, unlike
            `build_browser_agent`, since calling agents typically run unattended.
        allow_private: Allow navigation to loopback and private addresses. Leave this
            `False` unless every caller of the resulting tool is trusted.
    """

    async def _arun(url: str, goal: str, max_steps: int = 40) -> str:
        session = BrowserSession(headless=headless, allow_private=allow_private)
        agent = build_browser_agent(text_model=text_model, max_steps=max_steps, session=session)
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(f"{goal}\n\nStart at {url}")]})
        finally:
            await session.close()
        return _report(result["messages"])

    def _run(url: str, goal: str, max_steps: int = 40) -> str:
        return asyncio.run(_arun(url, goal, max_steps))

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name="browse_fast",
        description=_TOOL_DESCRIPTION,
        args_schema=BrowseFastInput,
    )


__all__ = ["BrowseFastInput", "make_browse_fast_tool"]
