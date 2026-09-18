"""Play the Wikipedia game: reach one article from another by clicking article links only.

Usage:
    uv run --env-file .env python examples/wiki_game.py --start "Jimmy Page" --end Microphone
"""

from __future__ import annotations

import argparse
import asyncio
from urllib.parse import quote, urldefrag, urljoin, urlparse

from _trace import run_and_print

from ts_browser_agent import Snapshot, build_browser_agent
from ts_browser_agent.tools import SnapshotFilter


def _article_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def _goal(start: str, end: str) -> str:
    # The start URL goes last. The classifier weighs the goal's opening heavily, and a
    # goal that opens with the start page reads as anchored to it once the agent has
    # moved on — two runs ended BLOCKED after one click with the URL first.
    return (
        f"Navigate to the Wikipedia article for {end} by repeatedly clicking links, "
        f"starting from the {start} article. You win by making the fewest clicks, so make "
        f"clicks that you think are more likely to get you to {end} faster. Do not use "
        "search, the only way to win is by clicking links to new articles in the body of "
        "each article as you progress. You progress by moving from article to article. "
        f"You are not allowed to go back.\n\nStart at {_article_url(start)}"
    )


def _is_article_link(href: str, page_url: str) -> bool:
    """Same host, under `/wiki/`, no namespace colon, no query string.

    That excludes `Special:`, `Talk:`, `File:`, `Category:` and friends, edit and history
    links, and the `index.php?...&redirect=no` stubs behind "(Redirected from ...)"
    hatnotes, which have a single link back to the article you just left.
    """
    page = urlparse(page_url)
    link = urlparse(urljoin(page_url, href))
    if link.netloc != page.netloc or link.query:
        return False
    prefix = "/wiki/"
    return link.path.startswith(prefix) and ":" not in link.path[len(prefix) :]


def game_rules() -> tuple[SnapshotFilter, set[str]]:
    """A snapshot filter enforcing the rules by removing options, not by asking.

    The rules belong to the game, not the engine, so they live here and are applied to
    every observation before the model sees it. Telling the model "do not go back" and
    "do not use search" in the goal did not work; removing the options does. Kept:
    buttons (subsection toggles) and links to articles not yet visited. Dropped: text
    fields, non-article links, and anything resolving to a visited page, which includes
    same-page `#anchor` links.
    """
    visited: set[str] = set()

    def apply(snapshot: Snapshot) -> Snapshot:
        visited.add(urldefrag(snapshot.url).url)
        kept = []
        for element in snapshot.elements:
            if element.kind != "click":
                continue
            if element.href is None:
                kept.append(element)
                continue
            if not _is_article_link(element.href, snapshot.url):
                continue
            if urldefrag(urljoin(snapshot.url, element.href)).url in visited:
                continue
            kept.append(element)
        return snapshot.model_copy(update={"elements": kept})

    return apply, visited


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="LangChain", help="Title of the article to start from.")
    parser.add_argument("--end", default="Microphone", help="Title of the article to reach.")
    args = parser.parse_args()

    rules, visited = game_rules()
    agent = build_browser_agent(max_steps=30, snapshot_filter=rules)
    await run_and_print(agent, _goal(args.start, args.end))
    print(f"Distinct articles visited: {len(visited)}")


if __name__ == "__main__":
    asyncio.run(main())
