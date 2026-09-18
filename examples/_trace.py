"""Shared by the examples: stream a run and print one line per decision and tool result."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ts_browser_agent.model import message_text


async def run_and_print(agent: Any, goal: str) -> str:
    """Stream `goal` through `agent`, printing each step; return the final outcome text."""
    outcome = ""
    async for update in agent.astream({"messages": [HumanMessage(goal)]}, stream_mode="updates"):
        for node, payload in update.items():
            if not isinstance(payload, dict):
                continue
            for message in payload.get("messages", []):
                if node == "model" and isinstance(message, AIMessage):
                    text = message_text(message.content)
                    print(f"-> {text}")
                    if not message.tool_calls:
                        outcome = text
                elif node == "tools" and isinstance(message, ToolMessage):
                    print(f"   {message_text(message.content)}")
    print(f"Outcome: {outcome}")
    return outcome
