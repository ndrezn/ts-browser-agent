"""Play "Getting to Philosophy": repeatedly click the first link in the article body.

Wikipedia's well-known phenomenon: clicking the first link in an article's main text,
ignoring parentheses and italics, and repeating, eventually reaches Philosophy from
most articles. Every step looks like the same UI (a wall of article text and links),
so this is less about varied navigation and more a stress test of one decision made
correctly, many times in a row, without drifting to a different link kind (citations,
infobox, "disambiguation" notices) along the way.

Usage:
    uv run --env-file .env python examples/wiki_hop.py
"""

from __future__ import annotations

from ts_browser_agent import Agent

_START_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
_GOAL = (
    "Reach the Wikipedia article about Philosophy by repeatedly clicking the first "
    "link in the main article text of each page you land on. Ignore navigation bars, "
    "infoboxes, citations/footnotes, italicized or parenthetical text, and links to "
    "disambiguation pages. Stop once the current page is the Philosophy article."
)


def main() -> None:
    with Agent(_START_URL, _GOAL, max_steps=30) as agent:
        for state in agent.run():
            print(f"{state['step']:>3}  {state['operation']:<10} {state['target'] or ''}  {state['url']}")
        print(f"Final status: {agent.status}")


if __name__ == "__main__":
    main()
