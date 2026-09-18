"""A fast browser agent: LangChain's `create_agent` with `langchain-typesafe` as the model.

Every turn, `TypeSafeBrowserModel` asks TypeSafe one `Choice` question for the next
operation and one speculative `Choice` question per candidate target, batched into a
single request, and answers with one tool call. The tools act on the page and return the
next snapshot. A small chat model runs only when the operation is `TYPE_TEXT`.
"""

from ts_browser_agent.agent import build_browser_agent
from ts_browser_agent.model import TypeSafeBrowserModel
from ts_browser_agent.snapshot import Element, Snapshot
from ts_browser_agent.tool import make_browse_fast_tool

__all__ = ["Element", "Snapshot", "TypeSafeBrowserModel", "build_browser_agent", "make_browse_fast_tool"]
