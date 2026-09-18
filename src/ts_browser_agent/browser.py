"""Playwright-backed execution of resolved decisions.

Every action re-resolves its element from the snapshot's node-identity map and rechecks
visibility and occlusion immediately before acting, rather than trusting geometry read
during the snapshot. A `StalePage` means the page changed since the decision was made
and the caller should re-observe instead of retrying blindly.

Async throughout: the loop runs under `create_agent`, whose tool node awaits async tools
on the event loop, and Playwright's sync API cannot be driven across threads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ts_browser_agent.safety import ensure_navigable
from ts_browser_agent.snapshot import Snapshot, aread_snapshot

if TYPE_CHECKING:
    from playwright.async_api import Browser as PlaywrightBrowser
    from playwright.async_api import Page, Playwright

ActionKind = Literal["click", "fill", "select", "scroll_down", "scroll_up", "wait"]

# Re-resolves a live node by the id the snapshot assigned it, then rejects the action if
# the node detached, became hidden or disabled, or something else now occupies the click
# point (a modal, a menu, a moved element) since the snapshot was read.
#
# The click point is the center of the element's *largest rendered fragment*, not of
# `getBoundingClientRect()`. An inline link that wraps across two lines has two
# fragments, and the center of their union box can land on plain paragraph text between
# them — which `elementFromPoint` correctly reports as not the link, producing a false
# "occluded" on every wrapped link in an article.
_RESOLVE_AND_ACT_JS = r"""
({ id, kind, value }) => {
  const el = window.__tsFastAgent?.nodes.get(id);
  if (!el || !el.isConnected) return { ok: false, reason: "detached" };
  if (el.disabled || !el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) {
    return { ok: false, reason: "not_visible" };
  }
  const fragments = [...el.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
  const rect = fragments.length
    ? fragments.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b))
    : el.getBoundingClientRect();
  const x = rect.x + rect.width / 2;
  const y = rect.y + rect.height / 2;
  if (rect.width <= 0 || rect.height <= 0 || x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) {
    return { ok: false, reason: "off_screen" };
  }
  const atPoint = document.elementFromPoint(x, y);
  if (!atPoint || !el.contains(atPoint)) return { ok: false, reason: "occluded" };
  if (kind === "select") {
    const hasOption = [...el.options].some((o) => o.value === value && !o.disabled);
    if (el.tagName !== "SELECT" || !hasOption) return { ok: false, reason: "option_missing" };
    el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return { ok: true };
  }
  return { ok: true, x, y };
}
"""


class StalePage(RuntimeError):
    """The page changed since the decision was made; observe again before acting."""


def _check(result: dict[str, Any]) -> None:
    if not result["ok"]:
        raise StalePage(f"Element is no longer actionable: {result['reason']}.")


class AsyncBrowser:
    """Owns one Playwright page for the lifetime of a run. Build it with `create`."""

    def __init__(self, playwright: Playwright, browser: PlaywrightBrowser, page: Page) -> None:
        self._playwright = playwright
        self._browser = browser
        self.page = page

    @classmethod
    async def create(cls, url: str, *, headless: bool = False, allow_private: bool = False) -> AsyncBrowser:
        """Launch Chromium and open `url`, after the URL passes `ensure_navigable`."""
        from playwright.async_api import async_playwright

        ensure_navigable(url, allow_private=allow_private)
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=headless)
        page = await browser.new_page()
        await page.goto(url, wait_until="load")
        return cls(playwright, browser, page)

    async def observe(self) -> Snapshot:
        """Read the current page into an indexed snapshot.

        Retries briefly on a transient Playwright error, since the previous action (a
        submit click, following a link) may have just triggered a navigation that is
        still settling when this is called.
        """
        from playwright.async_api import Error as PlaywrightError

        last_error: PlaywrightError | None = None
        for _ in range(10):
            try:
                return await aread_snapshot(self.page)
            except PlaywrightError as error:
                last_error = error
                await self.page.wait_for_timeout(50)
        assert last_error is not None
        raise last_error

    async def act(self, *, kind: ActionKind, node_id: int | None = None, value: str | None = None) -> None:
        """Execute one resolved action, re-validating the target immediately before it runs.

        Raises:
            StalePage: If the target element is no longer actionable.
        """
        # Scroll the document explicitly rather than via a wheel event at the mouse's
        # last position: after clicking into a sidebar (Wikipedia's table of contents,
        # say) the wheel scrolls that panel and the page never moves.
        if kind == "scroll_down":
            await self.page.evaluate("window.scrollBy(0, 600)")
            return
        if kind == "scroll_up":
            await self.page.evaluate("window.scrollBy(0, -600)")
            return
        if kind == "wait":
            await self.page.wait_for_timeout(300)
            return
        if node_id is None:  # pragma: no cover - the tools always pass one
            message = f"Action kind {kind!r} requires a target element."
            raise ValueError(message)
        if kind == "select":
            result = await self.page.evaluate(_RESOLVE_AND_ACT_JS, {"id": node_id, "kind": kind, "value": value})
            _check(result)
            return
        result = await self.page.evaluate(_RESOLVE_AND_ACT_JS, {"id": node_id, "kind": kind, "value": None})
        _check(result)
        await self.page.mouse.click(result["x"], result["y"])
        if kind == "fill" and value is not None:
            await self.page.keyboard.press("ControlOrMeta+A")
            await self.page.keyboard.type(value)
            # A combobox's suggestion list is a network round trip away; without this,
            # the very next snapshot can miss it entirely and the loop just retypes.
            await self.page.wait_for_timeout(250)

    async def close(self) -> None:
        await self._browser.close()
        await self._playwright.stop()


__all__ = ["ActionKind", "AsyncBrowser", "StalePage"]
