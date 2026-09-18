"""Expose `Agent` as a tool for a deep agent to call.

A deep agent's own model already plans across many tool calls and tracks progress
toward a larger goal; it should not also decide each individual click. This tool keeps
`Agent`'s fast, TypeSafe-classified step loop entirely inside one call: the deep agent
supplies a page and a single page-scoped goal, and gets back a status report plus the
visible text of the page the run ended on — enough to read a result (a price, a title,
a confirmation message) without the deep agent ever touching the browser itself.

```python
from deepagents import create_deep_agent
from ts_browser_agent.tool import make_browse_fast_tool

browse_fast = make_browse_fast_tool()
agent = create_deep_agent(model="openai:gpt-5.5", tools=[browse_fast])
```
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ts_browser_agent.agent import Agent

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
    """Arguments a deep agent supplies to the `browse_fast` tool."""

    url: str = Field(description="Page to start from.")
    goal: str = Field(
        description="One concrete, page-scoped goal, e.g. 'submit the search form for wireless mice'."
    )
    max_steps: int = Field(
        default=40, ge=1, le=200, description="Upper bound on attempts before giving up, at most 200."
    )


def make_browse_fast_tool(
    *,
    text_model: str | BaseChatModel = "openai:gpt-5-mini",
    headless: bool = True,
    allow_private: bool = False,
) -> StructuredTool:
    """Build a `browse_fast` tool bound to one text model, headless setting, and URL policy.

    `allow_private` and `headless` are fixed when the tool is built, not exposed to the
    calling model — a prompt-injected page must not be able to talk the deep agent into
    loosening the URL policy through tool-call arguments.

    Args:
        text_model: Chat model used only for `TYPE_TEXT` field values.
        headless: Whether the underlying browser runs headless. Defaults to `True`,
            unlike `Agent` itself, since deep agents typically run unattended.
        allow_private: Allow navigation to loopback and private addresses. Leave this
            `False` unless every caller of the resulting tool is trusted, since it
            widens what a prompt-injected goal could reach.

    Returns:
        A `StructuredTool` suitable for `create_agent`/`create_deep_agent`'s `tools`.
    """

    def _run(url: str, goal: str, max_steps: int = 40) -> str:
        with Agent(
            url,
            goal,
            text_model=text_model,
            max_steps=max_steps,
            headless=headless,
            allow_private=allow_private,
        ) as agent:
            steps = list(agent.run())
            page_text = agent.browser.observe().text
        last = steps[-1] if steps else None
        status_line = (
            f"status={agent.status} steps_taken={len(steps)} "
            f"final_url={last['url'] if last else url!r} "
            f"last_operation={last['operation'] if last else None}"
        )
        return f"{status_line}\n\nVisible page text at the end of the run:\n{page_text[:_MAX_REPORTED_TEXT]}"

    return StructuredTool.from_function(
        func=_run,
        name="browse_fast",
        description=_TOOL_DESCRIPTION,
        args_schema=BrowseFastInput,
    )


__all__ = ["BrowseFastInput", "make_browse_fast_tool"]
