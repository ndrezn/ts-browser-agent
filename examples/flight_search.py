"""Search Google Flights for a Montreal-to-tropics trip, without booking.

Exercises a long chain of clicks and selections on real, dynamic UI: origin field,
destination field, a date-picker (multiple clicks), then search. Mirrors the flight
search jev-ultrafast itself demos, but the goal explicitly stops once results are
visible — this project does not automate payment or ticket purchase. `agent.status ==
"done"` here means "results are visible," never "booked."

Usage:
    uv run --env-file .env python examples/flight_search.py
"""

from __future__ import annotations

from ts_browser_agent import Agent

_GOAL = (
    "Search for round-trip flights from Montreal (YUL) to Cancun (CUN), departing "
    "2027-01-15 and returning 2027-01-22, for three adults in economy. Stop once matching "
    "flight options are visible in the results list. Do not select a flight, do not "
    "proceed past the results page, and never enter payment or personal information."
)


def main() -> None:
    with Agent("https://www.google.com/travel/flights?hl=en", _GOAL, max_steps=60) as agent:
        for state in agent.run():
            print(f"{state['elapsed_ms']:>6} ms  {state['operation']:<10} {state['target'] or ''}")
        print(f"Final status: {agent.status}")
    print("This example only searches; it never selects a flight or enters payment details.")


if __name__ == "__main__":
    main()
