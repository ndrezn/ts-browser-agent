"""The step loop's stall and budget rules, driven against a fake browser.

`decide` is scripted and `Browser` is replaced, so no classifier request and no
Chromium are involved; what's under test is purely how `Agent` turns a sequence of
decisions and page outcomes into history, status, and spent budget.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from typing import Any

import pytest

from ts_browser_agent import agent as agent_module
from ts_browser_agent.agent import Agent
from ts_browser_agent.browser import StalePage
from ts_browser_agent.decision import Decision
from ts_browser_agent.snapshot import Element, Snapshot

ONE = Element(id=1, role="button", kind="click", label="One", current_value="")
TWO = Element(id=2, role="button", kind="click", label="Two", current_value="")
FIELD = Element(id=3, role="textbox", kind="fill", label="Query", current_value="")

CLICK_ONE = Decision(operation="CLICK", target=ONE, confidence=0.9)
CLICK_TWO = Decision(operation="CLICK", target=TWO, confidence=0.9)
TYPE_FIELD = Decision(operation="TYPE_TEXT", target=FIELD, confidence=0.9)
WAIT = Decision(operation="WAIT", target=None, confidence=0.5)
DONE = Decision(operation="DONE", target=None, confidence=0.9)

STALL_WINDOW = 3


class FakeBrowser:
    """A page whose fingerprint advances on every action unless it is `static`."""

    def __init__(self, url: str, *, headless: bool = False, allow_private: bool = False) -> None:
        self.url = url
        self.version = 0
        self.acts: list[tuple[str, int | None, str | None]] = []
        self.fail_targets: set[int] = set()
        self.static = False
        self.closed = False

    def observe(self) -> Snapshot:
        return Snapshot(
            url=self.url,
            title="t",
            text="x",
            elements=[ONE, TWO, FIELD],
            can_scroll_down=True,
            can_scroll_up=False,
            fingerprint=f"fp{self.version}",
        )

    def act(self, *, kind: str, node_id: int | None = None, value: str | None = None) -> None:
        if node_id in self.fail_targets:
            message = "Element is no longer actionable: occluded."
            raise StalePage(message)
        self.acts.append((kind, node_id, value))
        if not self.static:
            self.version += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_browser(monkeypatch: pytest.MonkeyPatch) -> type[FakeBrowser]:
    monkeypatch.setattr(agent_module, "Browser", FakeBrowser)
    return FakeBrowser


def _script(monkeypatch: pytest.MonkeyPatch, decisions: Iterable[Decision]) -> None:
    scripted: Iterator[Decision] = iter(decisions)
    monkeypatch.setattr(agent_module, "decide", lambda snapshot, goal, history: next(scripted))


def _run(max_steps: int = 40, **kwargs: Any) -> tuple[Agent, FakeBrowser, list[dict[str, Any]]]:
    agent = Agent("https://example.com/", "goal", max_steps=max_steps, **kwargs)
    browser: FakeBrowser = agent.browser  # type: ignore[assignment]
    steps = list(agent.run())
    return agent, browser, steps


def test_repeated_effective_actions_never_stall(fake_browser: type[FakeBrowser], monkeypatch: pytest.MonkeyPatch) -> None:
    # Four identical clicks that each change the page — a date picker's "Next" —
    # must run to completion rather than be mistaken for a loop.
    _script(monkeypatch, [CLICK_ONE] * 5 + [DONE])
    agent, browser, steps = _run()
    assert agent.status == "done"
    assert len(browser.acts) == 5
    assert all(not entry.endswith("(page did not change)") for entry in agent.history)
    assert all(step["page_changed"] for step in steps[:-1])


def test_three_ineffective_actions_stall(fake_browser: type[FakeBrowser], monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, itertools.repeat(CLICK_ONE))

    def _static(url: str, **kwargs: Any) -> FakeBrowser:
        browser = FakeBrowser(url, **kwargs)
        browser.static = True
        return browser

    monkeypatch.setattr(agent_module, "Browser", _static)
    agent, browser, steps = _run()
    assert agent.status == "stalled"
    assert len(browser.acts) == STALL_WINDOW
    assert all(entry.endswith("(page did not change)") for entry in agent.history)
    assert steps[-1]["page_changed"] is False


def test_wait_resets_the_ineffective_streak(fake_browser: type[FakeBrowser], monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [CLICK_ONE, CLICK_ONE, WAIT, CLICK_ONE, CLICK_ONE, DONE])

    def _static(url: str, **kwargs: Any) -> FakeBrowser:
        browser = FakeBrowser(url, **kwargs)
        browser.static = True
        return browser

    monkeypatch.setattr(agent_module, "Browser", _static)
    agent, browser, _ = _run()
    assert agent.status == "done"
    assert [kind for kind, _, _ in browser.acts] == ["click", "click", "wait", "click", "click"]


def test_repeated_failures_on_one_target_stall_early(fake_browser: type[FakeBrowser], monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, itertools.repeat(CLICK_ONE))

    def _failing(url: str, **kwargs: Any) -> FakeBrowser:
        browser = FakeBrowser(url, **kwargs)
        browser.fail_targets = {ONE.id}
        return browser

    monkeypatch.setattr(agent_module, "Browser", _failing)
    agent, browser, steps = _run(max_steps=40)
    assert agent.status == "stalled"
    assert browser.acts == []
    assert agent.history == []
    assert len(steps) == STALL_WINDOW


def test_failures_across_targets_still_spend_the_budget(fake_browser: type[FakeBrowser], monkeypatch: pytest.MonkeyPatch) -> None:
    # Alternating targets never trip the same-target rule, and never grow history —
    # `max_steps` must bound attempts, or this would spin until something changed.
    _script(monkeypatch, itertools.cycle([CLICK_ONE, CLICK_TWO]))

    def _failing(url: str, **kwargs: Any) -> FakeBrowser:
        browser = FakeBrowser(url, **kwargs)
        browser.fail_targets = {ONE.id, TWO.id}
        return browser

    monkeypatch.setattr(agent_module, "Browser", _failing)
    budget = 6
    agent, browser, steps = _run(max_steps=budget)
    assert agent.status == "blocked"
    assert len(steps) == budget
    assert browser.acts == []


def test_done_ends_the_run_without_acting(fake_browser: type[FakeBrowser], monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [DONE])
    agent, browser, steps = _run()
    assert agent.status == "done"
    assert browser.acts == []
    assert steps[-1]["operation"] == "DONE"
    assert steps[-1]["page_changed"] is None


def test_type_text_types_the_generated_value(fake_browser: type[FakeBrowser], monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [TYPE_FIELD, DONE])
    monkeypatch.setattr(agent_module, "generate_field_text", lambda *args, **kwargs: "hello")
    agent, browser, _ = _run()
    assert browser.acts == [("fill", FIELD.id, "hello")]
    assert agent.history == ["TYPE_TEXT 'Query' = 'hello'"]


def test_context_manager_closes_the_browser(fake_browser: type[FakeBrowser], monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [DONE])
    with Agent("https://example.com/", "goal") as agent:
        browser: FakeBrowser = agent.browser  # type: ignore[assignment]
        list(agent.run())
    assert browser.closed


def test_empty_goal_rejected(fake_browser: type[FakeBrowser]) -> None:
    with pytest.raises(ValueError, match="goal"):
        Agent("https://example.com/", "   ")
