"""Search Google Flights for a Montreal-to-tropics trip, without booking.

A long chain of clicks and selections on real, dynamic UI: origin and destination
autocomplete, a date picker navigated several months forward, then search. The goal
stops once results are visible — this project does not automate payment or ticket
purchase. `DONE` here means "results are visible," never "booked."

Usage:
    uv run --env-file .env python examples/flight_search.py
"""

from __future__ import annotations

import asyncio

from _trace import run_and_print

from ts_browser_agent import build_browser_agent

_GOAL = (
    "Search for round-trip flights from Montreal (YUL) to Cancun (CUN), departing "
    "2027-01-15 and returning 2027-01-22, for three adults in economy. Stop once matching "
    "flight options are visible in the results list. Do not select a flight, do not "
    "proceed past the results page, and never enter payment or personal information."
    "\n\nStart at https://www.google.com/travel/flights?hl=en"
)


async def main() -> None:
    await run_and_print(build_browser_agent(max_steps=60), _GOAL)
    print("This example only searches; it never selects a flight or enters payment details.")


if __name__ == "__main__":
    asyncio.run(main())
