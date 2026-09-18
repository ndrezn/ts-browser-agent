"""Play "Getting to Philosophy": repeatedly click the first link in the article body.

Wikipedia's well-known phenomenon: clicking the first link in an article's main text,
ignoring parentheses and italics, and repeating, eventually reaches Philosophy from most
articles. Every step looks like the same UI (a wall of article text and links), so this
is less about varied navigation and more a stress test of one decision made correctly,
many times in a row, without drifting to a different link kind (citations, infobox,
"disambiguation" notices) along the way.

Usage:
    uv run --env-file .env python examples/wiki_hop.py
"""

from __future__ import annotations

import asyncio

from _trace import run_and_print

from ts_browser_agent import build_browser_agent

_GOAL = (
    "Reach the Wikipedia article about Philosophy by repeatedly clicking the first link "
    "in the main article text of each page you land on. Ignore navigation bars, "
    "infoboxes, citations/footnotes, italicized or parenthetical text, and links to "
    "disambiguation pages. Stop once the current page is the Philosophy article."
    "\n\nStart at https://en.wikipedia.org/wiki/Noemvriana"
)


async def main() -> None:
    await run_and_print(build_browser_agent(max_steps=30), _GOAL)


if __name__ == "__main__":
    asyncio.run(main())
