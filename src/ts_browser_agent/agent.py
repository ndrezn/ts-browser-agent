"""Build the browser agent: LangChain's `create_agent`, with TypeSafe as the model.

```python
import asyncio
from ts_browser_agent import build_browser_agent

agent = build_browser_agent()
goal = "Find the pricing page.\\n\\nStart at https://example.com"
result = asyncio.run(agent.ainvoke({"messages": [("user", goal)]}))
print(result["messages"][-1].content)  # "DONE", "BLOCKED", or "STALLED: ..."
```

There is no loop code here. The model (`TypeSafeBrowserModel`) decides one tool call per
turn from the page it is shown; the tools (`browser_tools`) act and return the next page;
`create_agent` runs that until the model stops calling tools. `DONE` is the classifier's
judgment that the goal is visibly satisfied, not a guarantee — verify before acting on it.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from ts_browser_agent.model import TypeSafeBrowserModel
from ts_browser_agent.tools import (
    BrowserFactory,
    BrowserSession,
    CloseBrowser,
    SnapshotFilter,
    browser_tools,
)


def build_browser_agent(
    *,
    text_model: str | BaseChatModel = "openai:gpt-5-mini",
    max_steps: int = 40,
    headless: bool = False,
    allow_private: bool = False,
    snapshot_filter: SnapshotFilter | None = None,
    browser_factory: BrowserFactory | None = None,
    session: BrowserSession | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile a browser agent. Async-only: drive it with `ainvoke` or `astream`.

    The goal message must name the URL to start from; the model's first tool call opens
    it. Put the URL at the end (`"...\\n\\nStart at https://..."`, as `browse_fast` does):
    the classifier weighs the goal's opening heavily, and a goal that opens with the
    start page reads as anchored to it once the agent has moved on. One browser per
    graph instance, so run one goal at a time per instance.

    Args:
        text_model: Chat model used only for `TYPE_TEXT` values.
        max_steps: Actions allowed per run before it ends `BLOCKED`.
        headless: Whether to launch Chromium headless.
        allow_private: Allow the start URL to resolve to a private or loopback address.
        snapshot_filter: Applied to every observation before the model sees it.
        browser_factory: Replaces `AsyncBrowser.create`; tests pass a fake.
        session: Supply your own session to control the browser's lifecycle; the other
            browser options are then ignored.
    """
    if session is None:
        session = BrowserSession(
            headless=headless,
            allow_private=allow_private,
            snapshot_filter=snapshot_filter,
            browser_factory=browser_factory,
        )
    return create_agent(
        model=TypeSafeBrowserModel(text_model=text_model, max_steps=max_steps),
        tools=browser_tools(session),
        middleware=[CloseBrowser(session)],
    )


__all__ = ["build_browser_agent"]
