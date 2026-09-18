"""Answer an open-ended question by comparing several concrete flight searches.

`Agent` alone can't answer "what's the cheapest option... which week do you
recommend?" — it only clicks through UI toward one page-scoped goal, with no
mechanism to extract, compare, or reason about prices across searches. That's
exactly the shape a deep agent sits above `browse_fast` for: it plans which weeks
to check, delegates each search wholesale, reads the prices back from each call's
report, and synthesizes a recommendation.

`browse_fast`'s classifier matches goal text against the calendar's own day-cell
labels — it has no date arithmetic. A goal like "in the next two weeks" gives it
nothing concrete to match, and it can click through months indefinitely without
ever converging (confirmed while building this example). The deep agent's model,
unlike the fast loop, can compute real dates, so the system prompt requires every
`browse_fast` goal to carry an absolute date it already resolved from "this winter."

Usage:
    uv run --env-file .env python examples/flight_search_generic.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage

from ts_browser_agent import make_browse_fast_tool

_REPORT_PREVIEW_CHARS = 300

_SYSTEM_PROMPT = (
    f"Today's date is {datetime.now(tz=UTC).date().isoformat()}. You research flight prices by "
    "delegating individual searches to `browse_fast`. That tool's classifier matches "
    "your goal text against a calendar's own day labels; it cannot compute relative "
    "dates like 'next month' or 'in two weeks'. Always resolve a concrete YYYY-MM-DD "
    "departure and return date yourself before calling it, and put those exact dates "
    "in the goal you send it. Pass max_steps=60 on every call — a full search takes "
    "several clicks to navigate the calendar to the right month. Each call's report "
    "includes the visible text of the page it ended on — read prices directly from "
    "that text; do not assume a call found a price just because it reports `done`. "
    "`browse_fast` only searches; it never selects or books a flight."
)

_GOAL = (
    "What's the cheapest option for flights from Montreal (YUL) to Cancun (CUN) this "
    "winter? Check every different week spread across December, January, "
    "and February — one round-trip search per week, 7 nights each, one adult in "
    "economy, on https://www.google.com/travel/flights?hl=en — then tell me which "
    "week you recommend the cheapest price you found for it, and the booking link."
)


def _text(content: object) -> str:
    """Flatten message content to its text; gpt-5.5 returns a block list with reasoning."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def main() -> None:
    browse_fast = make_browse_fast_tool(headless=False)
    agent = create_deep_agent(model="openai:gpt-5.5", tools=[browse_fast], system_prompt=_SYSTEM_PROMPT)

    # Stream so every browse_fast call and its report are visible as they happen — how
    # many searches ran, what dates were sent, whether any came back blocked and were
    # retried, and that each quoted price actually appears in a page-text report.
    final_answer = ""
    for update in agent.stream({"messages": [("user", _GOAL)]}, stream_mode="updates"):
        for node_output in update.values():
            if not isinstance(node_output, dict):
                continue
            for message in node_output.get("messages", []):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        print(f"-> {call['name']}({json.dumps(call['args'])})")
                    if text := _text(message.content).strip():
                        final_answer = text
                elif isinstance(message, ToolMessage):
                    status_line, _, page_text = _text(message.content).partition("\n")
                    print(f"<- {status_line}")
                    preview = " ".join(page_text.split())[:_REPORT_PREVIEW_CHARS]
                    if preview:
                        print(f"   {preview}...")

    print(f"\n{final_answer}")


if __name__ == "__main__":
    main()
