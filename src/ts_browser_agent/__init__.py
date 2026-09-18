"""A fast browser agent built on `langchain-typesafe` and the LangChain SDK.

Every step asks TypeSafe one `Choice` question for the next operation and one
speculative `Choice` question per candidate target, batched into a single
`TypeSafeClassifier` request. A small chat model only runs when the resolved
operation is `TYPE_TEXT`.
"""

from ts_browser_agent.agent import Agent
from ts_browser_agent.snapshot import Element, Snapshot
from ts_browser_agent.tool import make_browse_fast_tool

__all__ = ["Agent", "Element", "Snapshot", "make_browse_fast_tool"]
