"""`TypeSafeBrowserModel` decides from messages alone; drive it with hand-built histories."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ts_browser_agent import model as model_module
from ts_browser_agent.decision import Decision
from ts_browser_agent.model import TypeSafeBrowserModel
from ts_browser_agent.snapshot import Element, Snapshot

ONE = Element(id=1, role="button", kind="click", label="One", current_value="")
FIELD = Element(id=2, role="textbox", kind="fill", label="Query", current_value="")
SIZE = Element(id=3, role="combobox", kind="select", label="Size -> L", current_value="M", option_value="L")

GOAL = HumanMessage("Go to https://example.com/start. Then click One.")


def _snapshot(fingerprint: str) -> Snapshot:
    return Snapshot(
        url="https://example.com/start",
        title="t",
        text="x",
        elements=[ONE, FIELD, SIZE],
        can_scroll_down=True,
        can_scroll_up=False,
        fingerprint=fingerprint,
    )


def _step(tool: str, args: dict[str, Any], snapshot: Snapshot, *, content: str = "ok", description: str | None = None) -> list[BaseMessage]:
    call_id = f"call_{tool}_{snapshot.fingerprint}"
    return [
        AIMessage(content=description or tool, tool_calls=[{"name": tool, "args": args, "id": call_id, "type": "tool_call"}]),
        ToolMessage(content=content, tool_call_id=call_id, artifact=snapshot),
    ]


def _opened(fingerprint: str = "fp0") -> list[BaseMessage]:
    return [GOAL, *_step("open", {"url": "https://example.com/start"}, _snapshot(fingerprint))]


def _decide(*decisions: Decision, monkeypatch: pytest.MonkeyPatch, seen: list[list[str]] | None = None) -> None:
    scripted = iter(decisions)

    async def fake(snapshot: Snapshot, goal: str, history: list[str]) -> Decision:
        if seen is not None:
            seen.append(history)
        return next(scripted)

    monkeypatch.setattr(model_module, "adecide", fake)


def _run(model: TypeSafeBrowserModel, messages: list[BaseMessage]) -> AIMessage:
    result = asyncio.run(model.ainvoke(messages))
    assert isinstance(result, AIMessage)
    return result


def test_first_turn_opens_the_url_from_the_goal() -> None:
    message = _run(TypeSafeBrowserModel(), [GOAL])
    assert message.tool_calls == [
        {"name": "open", "args": {"url": "https://example.com/start"}, "id": message.tool_calls[0]["id"], "type": "tool_call"}
    ]
    assert message.content == "OPEN https://example.com/start"


def test_goal_without_url_is_blocked() -> None:
    message = _run(TypeSafeBrowserModel(), [HumanMessage("Click One.")])
    assert message.tool_calls == []
    assert message.content.startswith("BLOCKED")


def test_done_ends_without_a_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(operation="DONE", target=None, confidence=0.9), monkeypatch=monkeypatch)
    message = _run(TypeSafeBrowserModel(), _opened())
    assert message.tool_calls == []
    assert message.content == "DONE"


def test_click_becomes_a_click_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(operation="CLICK", target=ONE, confidence=0.9), monkeypatch=monkeypatch)
    message = _run(TypeSafeBrowserModel(), _opened())
    assert message.tool_calls[0]["name"] == "click"
    assert message.tool_calls[0]["args"] == {"node_id": 1}
    assert message.content == "CLICK 'One'"


def test_type_text_uses_the_generated_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(operation="TYPE_TEXT", target=FIELD, confidence=0.9), monkeypatch=monkeypatch)

    async def fake_text(*args: Any, **kwargs: Any) -> str:
        return "hello"

    monkeypatch.setattr(model_module, "agenerate_field_text", fake_text)
    message = _run(TypeSafeBrowserModel(text_model="openai:gpt-5-mini"), _opened())
    assert message.tool_calls[0]["name"] == "type_text"
    assert message.tool_calls[0]["args"] == {"node_id": 2, "value": "hello"}
    assert message.content == "TYPE_TEXT 'Query' = 'hello'"


def test_select_carries_the_option_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(operation="SELECT", target=SIZE, confidence=0.9), monkeypatch=monkeypatch)
    message = _run(TypeSafeBrowserModel(), _opened())
    assert message.tool_calls[0]["name"] == "select_option"
    assert message.tool_calls[0]["args"] == {"node_id": 3, "value": "L"}


def test_targetless_operation_calls_its_tool_with_no_args(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(operation="SCROLL_DOWN", target=None, confidence=0.9), monkeypatch=monkeypatch)
    message = _run(TypeSafeBrowserModel(), _opened())
    assert message.tool_calls[0]["name"] == "scroll_down"
    assert message.tool_calls[0]["args"] == {}


def test_history_marks_actions_that_changed_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    _decide(Decision(operation="DONE", target=None, confidence=0.9), monkeypatch=monkeypatch, seen=seen)
    messages = [
        *_opened("fp0"),
        *_step("click", {"node_id": 1}, _snapshot("fp1"), description="CLICK 'One'"),
        *_step("click", {"node_id": 1}, _snapshot("fp1"), description="CLICK 'One'"),
    ]
    _run(TypeSafeBrowserModel(), messages)
    assert seen == [["CLICK 'One'", "CLICK 'One' (page did not change)"]]


def test_three_actions_that_change_nothing_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)  # any classifier call would exhaust the empty script
    messages = [*_opened("fp0")]
    for _ in range(3):
        messages += _step("click", {"node_id": 1}, _snapshot("fp0"), description="CLICK 'One'")
    message = _run(TypeSafeBrowserModel(), messages)
    assert message.tool_calls == []
    assert message.content.startswith("STALLED")


def test_wait_breaks_a_no_change_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(operation="CLICK", target=ONE, confidence=0.9), monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    messages += _step("click", {"node_id": 1}, _snapshot("fp0"))
    messages += _step("wait", {}, _snapshot("fp0"))
    messages += _step("click", {"node_id": 1}, _snapshot("fp0"))
    message = _run(TypeSafeBrowserModel(), messages)
    assert message.tool_calls[0]["name"] == "click"


def test_repeated_effective_actions_never_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(operation="CLICK", target=ONE, confidence=0.9), monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    for i in range(1, 6):
        messages += _step("click", {"node_id": 1}, _snapshot(f"fp{i}"))
    message = _run(TypeSafeBrowserModel(), messages)
    assert message.tool_calls[0]["name"] == "click"


def test_same_target_failures_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    for _ in range(3):
        messages += _step("click", {"node_id": 1}, _snapshot("fp0"), content="failed: occluded")
    message = _run(TypeSafeBrowserModel(), messages)
    assert message.content.startswith("STALLED")


def test_step_budget_ends_the_run_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    for i in range(1, 3):
        messages += _step("click", {"node_id": 1}, _snapshot(f"fp{i}"))
    message = _run(TypeSafeBrowserModel(max_steps=2), messages)
    assert message.tool_calls == []
    assert "budget" in message.content


def test_bind_tools_is_accepted_and_sync_generate_is_refused() -> None:
    model = TypeSafeBrowserModel()
    assert model.bind_tools([]) is not None
    with pytest.raises(NotImplementedError, match="async-only"):
        model.invoke([GOAL])
