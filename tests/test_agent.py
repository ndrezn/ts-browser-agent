"""The compiled agent end to end, against a fake browser and a scripted classifier."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Iterable
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ts_browser_agent import model as model_module
from ts_browser_agent.agent import build_browser_agent
from ts_browser_agent.browser import StalePage
from ts_browser_agent.decision import Decision
from ts_browser_agent.snapshot import Element, Snapshot

ONE = Element(id=1, role="button", kind="click", label="One", current_value="")
TWO = Element(id=2, role="button", kind="click", label="Two", current_value="")
FIELD = Element(id=3, role="textbox", kind="fill", label="Query", current_value="")

CLICK_ONE = Decision(operation="CLICK", target=ONE, confidence=0.9)
CLICK_TWO = Decision(operation="CLICK", target=TWO, confidence=0.9)
TYPE_FIELD = Decision(operation="TYPE_TEXT", target=FIELD, confidence=0.9)
DONE = Decision(operation="DONE", target=None, confidence=0.9)

GOAL = "Go to https://example.com/start and click One."
STALL_WINDOW = 3


class FakeBrowser:
    """A page whose fingerprint advances on every action unless it is `static`."""

    created: ClassVar[list[FakeBrowser]] = []

    def __init__(self, url: str, *, static: bool = False, fail_targets: Iterable[int] = ()) -> None:
        self.url = url
        self.version = 0
        self.static = static
        self.fail_targets = set(fail_targets)
        self.acts: list[tuple[str, int | None, str | None]] = []
        self.closed = False
        FakeBrowser.created.append(self)

    async def observe(self) -> Snapshot:
        return Snapshot(
            url=self.url,
            title="t",
            text="x",
            elements=[ONE, TWO, FIELD],
            can_scroll_down=True,
            can_scroll_up=False,
            fingerprint=f"fp{self.version}",
        )

    async def act(self, *, kind: str, node_id: int | None = None, value: str | None = None) -> None:
        if node_id in self.fail_targets:
            message = "Element is no longer actionable: occluded."
            raise StalePage(message)
        self.acts.append((kind, node_id, value))
        if not self.static:
            self.version += 1

    async def close(self) -> None:
        self.closed = True


def _factory(calls: list[dict[str, Any]] | None = None, **options: Any) -> Any:
    async def create(url: str, *, headless: bool = False, allow_private: bool = False) -> FakeBrowser:
        if calls is not None:
            calls.append({"url": url, "headless": headless, "allow_private": allow_private})
        return FakeBrowser(url, **options)

    return create


def _script(monkeypatch: pytest.MonkeyPatch, decisions: Iterable[Decision]) -> None:
    scripted = iter(decisions)

    async def fake(snapshot: Snapshot, goal: str, history: list[str]) -> Decision:
        return next(scripted)

    monkeypatch.setattr(model_module, "adecide", fake)


def _run(agent: Any, goal: str = GOAL) -> list[Any]:
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(goal)]}))
    return list(result["messages"])


@pytest.fixture(autouse=True)
def _reset_browsers() -> None:
    FakeBrowser.created.clear()


def test_open_uses_the_goal_url_and_the_build_time_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [DONE])
    calls: list[dict[str, Any]] = []
    agent = build_browser_agent(browser_factory=_factory(calls), headless=True, allow_private=True)
    messages = _run(agent)
    assert calls == [{"url": "https://example.com/start", "headless": True, "allow_private": True}]
    assert messages[-1].content == "DONE"


def test_actions_reach_the_browser_and_results_carry_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [CLICK_ONE] * 5 + [DONE])
    agent = build_browser_agent(browser_factory=_factory())
    messages = _run(agent)
    browser = FakeBrowser.created[0]
    assert browser.acts == [("click", 1, None)] * 5
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 6  # open + five clicks
    assert all(isinstance(m.artifact, Snapshot) for m in tool_messages)
    assert messages[-1].content == "DONE"


def test_browser_is_closed_when_the_run_ends_done(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [DONE])
    _run(build_browser_agent(browser_factory=_factory()))
    assert FakeBrowser.created[0].closed


def test_no_change_actions_stall_and_close_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, itertools.repeat(CLICK_ONE))
    agent = build_browser_agent(browser_factory=_factory(static=True))
    messages = _run(agent)
    browser = FakeBrowser.created[0]
    assert len(browser.acts) == STALL_WINDOW
    assert messages[-1].content.startswith("STALLED")
    assert browser.closed


def test_same_target_failures_stall_early(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, itertools.repeat(CLICK_ONE))
    agent = build_browser_agent(browser_factory=_factory(fail_targets=[1]))
    messages = _run(agent)
    browser = FakeBrowser.created[0]
    assert browser.acts == []
    failed = [m for m in messages if isinstance(m, ToolMessage) and str(m.content).startswith("failed:")]
    assert len(failed) == STALL_WINDOW
    assert messages[-1].content.startswith("STALLED")
    assert browser.closed


def test_failures_across_targets_spend_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, itertools.cycle([CLICK_ONE, CLICK_TWO]))
    agent = build_browser_agent(browser_factory=_factory(fail_targets=[1, 2]), max_steps=6)
    messages = _run(agent)
    assert "budget" in messages[-1].content
    assert FakeBrowser.created[0].closed


def test_type_text_types_the_generated_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [TYPE_FIELD, DONE])

    async def fake_text(*args: Any, **kwargs: Any) -> str:
        return "hello"

    monkeypatch.setattr(model_module, "agenerate_field_text", fake_text)
    _run(build_browser_agent(browser_factory=_factory()))
    assert FakeBrowser.created[0].acts == [("fill", 3, "hello")]


def test_model_messages_describe_each_action(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [CLICK_ONE, DONE])
    messages = _run(build_browser_agent(browser_factory=_factory()))
    descriptions = [m.content for m in messages if isinstance(m, AIMessage)]
    assert descriptions == ["OPEN https://example.com/start", "CLICK 'One'", "DONE"]
