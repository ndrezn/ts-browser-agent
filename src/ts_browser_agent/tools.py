"""Browser actions as tools, and the one browser they act on.

Each tool performs its action and then observes, returning a short status line as the
tool result's content and the new `Snapshot` as its artifact. That artifact is how the
model sees the page on its next turn — the ordinary tool-calling contract, with a
structured observation instead of prose.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.tools import BaseTool, tool
from langgraph.runtime import Runtime

from ts_browser_agent.browser import ActionKind, AsyncBrowser, StalePage
from ts_browser_agent.snapshot import Snapshot

BrowserFactory = Callable[..., Awaitable[AsyncBrowser]]
SnapshotFilter = Callable[[Snapshot], Snapshot]


class BrowserSession:
    """The browser a graph instance acts on, opened by the `open` tool.

    It lives here rather than in graph state because a Playwright handle cannot be
    checkpointed. One session means one run at a time per graph instance;
    `browse_fast` builds a fresh graph per call for that reason.

    Args:
        headless: Whether to launch Chromium headless.
        allow_private: Allow the opened URL to resolve to a private or loopback address;
            see `ts_browser_agent.safety.ensure_navigable`. Fixed here, never a tool
            argument.
        snapshot_filter: Optional hook applied to every observation before the model
            sees it, for callers that want to drop or relabel candidates (the Wikipedia
            game uses it to enforce its rules).
        browser_factory: Replaces `AsyncBrowser.create`; tests pass a fake.
    """

    def __init__(
        self,
        *,
        headless: bool = False,
        allow_private: bool = False,
        snapshot_filter: SnapshotFilter | None = None,
        browser_factory: BrowserFactory | None = None,
    ) -> None:
        self.headless = headless
        self.allow_private = allow_private
        self.snapshot_filter = snapshot_filter
        self._factory: BrowserFactory = browser_factory or AsyncBrowser.create
        self.browser: AsyncBrowser | None = None

    async def open(self, url: str) -> None:
        await self.close()
        self.browser = await self._factory(url, headless=self.headless, allow_private=self.allow_private)

    def require(self) -> AsyncBrowser:
        if self.browser is None:
            message = "No page is open; the `open` tool must run first."
            raise RuntimeError(message)
        return self.browser

    async def observe(self) -> Snapshot:
        snapshot = await self.require().observe()
        return self.snapshot_filter(snapshot) if self.snapshot_filter else snapshot

    async def close(self) -> None:
        if self.browser is not None:
            await self.browser.close()
            self.browser = None


def browser_tools(session: BrowserSession) -> list[BaseTool]:
    """The action tools for one session. Every tool returns `(status, Snapshot)`."""

    async def report(status: str) -> tuple[str, Snapshot]:
        snapshot = await session.observe()
        return f"{status} — {snapshot.title} — {snapshot.url}", snapshot

    async def act(kind: ActionKind, node_id: int | None = None, value: str | None = None) -> tuple[str, Snapshot]:
        try:
            await session.require().act(kind=kind, node_id=node_id, value=value)
        except StalePage as error:
            return await report(f"failed: {error}")
        return await report("ok")

    @tool("open", response_format="content_and_artifact")
    async def open_page(url: str) -> tuple[str, Snapshot]:
        """Open a page in a fresh browser. Always the first action of a run."""
        await session.open(url)
        return await report("ok")

    @tool(response_format="content_and_artifact")
    async def click(node_id: int) -> tuple[str, Snapshot]:
        """Click the element with this id from the current page's element table."""
        return await act("click", node_id)

    @tool(response_format="content_and_artifact")
    async def type_text(node_id: int, value: str) -> tuple[str, Snapshot]:
        """Replace the text in the field with this id."""
        return await act("fill", node_id, value)

    @tool(response_format="content_and_artifact")
    async def select_option(node_id: int, value: str) -> tuple[str, Snapshot]:
        """Choose one option of the dropdown with this id."""
        return await act("select", node_id, value)

    @tool(response_format="content_and_artifact")
    async def scroll_up() -> tuple[str, Snapshot]:
        """Scroll up to reveal content above the current view."""
        return await act("scroll_up")

    @tool(response_format="content_and_artifact")
    async def scroll_down() -> tuple[str, Snapshot]:
        """Scroll down to reveal content below the current view."""
        return await act("scroll_down")

    @tool(response_format="content_and_artifact")
    async def wait() -> tuple[str, Snapshot]:
        """Wait briefly for the page to update."""
        return await act("wait")

    return [open_page, click, type_text, select_option, scroll_up, scroll_down, wait]


class CloseBrowser(AgentMiddleware[AgentState[Any], Any, Any]):
    """Close the session's browser when the run ends, on every exit path."""

    def __init__(self, session: BrowserSession) -> None:
        super().__init__()
        self._session = session

    async def aafter_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> None:
        await self._session.close()


__all__ = ["BrowserFactory", "BrowserSession", "CloseBrowser", "SnapshotFilter", "browser_tools"]
