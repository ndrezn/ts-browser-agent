"""Run the agent against one URL and goal.

Usage:
    uv run --env-file .env python examples/run.py \
        --url https://en.wikipedia.org/wiki/Main_Page \
        --goal "Open the article about the Rosetta Stone."
"""

from __future__ import annotations

import argparse

from ts_browser_agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()

    with Agent(args.url, args.goal, max_steps=args.max_steps, headless=args.headless) as agent:
        for state in agent.run():
            print(f"[{state['elapsed_ms']:>6} ms] {state['operation']:<10} {state['target'] or ''}")
        print(f"Final status: {agent.status}")


if __name__ == "__main__":
    main()
