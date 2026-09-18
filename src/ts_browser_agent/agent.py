"""The complete step loop: observe, decide with TypeSafe, act, repeat.

```python
from ts_browser_agent import Agent

with Agent("https://example.com", "Find the pricing page and open it.") as agent:
    for state in agent.run():
        print(state["step"], state["status"], state["operation"])
```

`DONE` still needs independent verification by the caller; like the fan-out pattern this
mirrors, the loop only reports that TypeSafe believes the goal is visibly satisfied.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Self

from langchain_core.language_models import BaseChatModel

from ts_browser_agent.browser import ActionKind, Browser, StalePage
from ts_browser_agent.decision import Decision, decide
from ts_browser_agent.snapshot import Snapshot
from ts_browser_agent.text import generate_field_text

Status = Literal["running", "done", "blocked", "stalled"]

_OPERATION_TO_ACTION: dict[str, ActionKind] = {
    "CLICK": "click",
    "TYPE_TEXT": "fill",
    "SELECT": "select",
    "SCROLL_UP": "scroll_up",
    "SCROLL_DOWN": "scroll_down",
    "WAIT": "wait",
}
_STALL_WINDOW = 3


class Agent:
    """Drives one Playwright page toward `goal` using TypeSafe-classified steps.

    Args:
        url: Starting page. Validated by `ensure_navigable` before anything loads,
            since this value may originate from an LLM rather than a person once
            `Agent` is wrapped as a tool.
        goal: Natural-language description of what a completed run looks like.
        text_model: Chat model (or model string for `init_chat_model`) used only to
            generate the value for `TYPE_TEXT` steps. Defaults to a small OpenAI model;
            pick something cheap, since it sits on the per-step critical path.
        max_steps: Upper bound on attempts, successful or not, before the run stops as
            `blocked`. A repeatedly failing action still spends budget.
        headless: Whether to launch Chromium headless.
        allow_private: Allow `url` to resolve to a loopback or private address, for
            testing against a local dev server. Never set this from untrusted input;
            see `ts_browser_agent.safety.ensure_navigable`.
    """

    def __init__(
        self,
        url: str,
        goal: str,
        *,
        text_model: str | BaseChatModel = "openai:gpt-5-mini",
        max_steps: int = 40,
        headless: bool = False,
        allow_private: bool = False,
    ) -> None:
        goal = goal.strip()
        if not goal:
            message = "`goal` must not be empty."
            raise ValueError(message)
        self.goal = goal
        self.text_model = text_model
        self.max_steps = max_steps
        self.browser = Browser(url, headless=headless, allow_private=allow_private)
        self.history: list[str] = []
        self.status: Status = "running"
        self._ticks = 0
        self._consecutive_failures = 0
        self._last_failed_target: str | None = None
        self._unchanged_streak = 0

    def _resolve_and_act(self, snapshot: Snapshot, decision: Decision) -> str:
        """Execute the decision and return its human-readable description."""
        action = _OPERATION_TO_ACTION[decision.operation]
        if decision.target is None:
            self.browser.act(kind=action)
            return decision.operation
        element = decision.target
        value = element.option_value
        if decision.operation == "TYPE_TEXT":
            value = generate_field_text(self.text_model, self.goal, element, snapshot, self.history)
        self.browser.act(kind=action, node_id=element.id, value=value)
        return f"{decision.operation} '{element.label}'" + (f" = {value!r}" if value else "")

    def _step(self, started_at: float) -> dict[str, Any]:
        self._ticks += 1
        snapshot = self.browser.observe()
        decision = decide(snapshot, self.goal, self.history)
        page_changed: bool | None = None
        if decision.operation in {"DONE", "BLOCKED"}:
            self.status = "done" if decision.operation == "DONE" else "blocked"
        else:
            target_signature = f"{decision.operation}:{decision.target.id if decision.target else None}"
            try:
                description = self._resolve_and_act(snapshot, decision)
                self._consecutive_failures = 0
            except StalePage:
                # A repeated failure on the exact same target never grows `history`, so
                # without this the loop could retry an unresolvably occluded/stale
                # element forever within its `max_steps` budget instead of giving up.
                self._consecutive_failures = (
                    self._consecutive_failures + 1 if target_signature == self._last_failed_target else 1
                )
                self._last_failed_target = target_signature
                if self._consecutive_failures >= _STALL_WINDOW:
                    self.status = "stalled"
            else:
                # Stall on repeated *ineffective* actions, never on repeated actions as
                # such: advancing a date picker four months is four identical `Next`
                # clicks that each change the page, while re-clicking a link that
                # redirects to the current page changes nothing. The marker in the
                # history string lets the classifier see the no-op in `recent_actions`.
                page_changed = self.browser.observe().fingerprint != snapshot.fingerprint
                self.history.append(description if page_changed else f"{description} (page did not change)")
                if page_changed or decision.operation == "WAIT":
                    self._unchanged_streak = 0
                else:
                    self._unchanged_streak += 1
                    if self._unchanged_streak >= _STALL_WINDOW:
                        self.status = "stalled"
        return {
            "step": len(self.history),
            "operation": decision.operation,
            "target": decision.target.label if decision.target else None,
            "confidence": decision.confidence,
            "page_changed": page_changed,
            "status": self.status,
            "url": snapshot.url,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
        }

    def run(self) -> Any:
        """Run steps until the goal is satisfied, blocked, stalled, or `max_steps` is hit.

        `max_steps` bounds total attempts, not just successful actions — a repeatedly
        failing action (see `_consecutive_failures`) still counts against it.
        """
        started_at = time.perf_counter()
        while self.status == "running" and self._ticks < self.max_steps:
            yield self._step(started_at)
        if self.status == "running":
            self.status = "blocked"

    def close(self) -> None:
        self.browser.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["Agent"]
