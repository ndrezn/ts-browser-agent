"""Per-step classifier construction and answer resolution, with no request sent."""

from __future__ import annotations

from langchain_typesafe.types import ChoiceAnswer

from ts_browser_agent.decision import build_classifier, resolve_decision
from ts_browser_agent.snapshot import Element, Snapshot

CLICK = Element(id=1, role="button", kind="click", label="Submit", current_value="")
FILL = Element(id=2, role="textbox", kind="fill", label="Query", current_value="abc")
SELECT = Element(id=3, role="combobox", kind="select", label="Size -> L", current_value="M", option_value="L")


def _snapshot(elements: list[Element], *, down: bool = False, up: bool = False) -> Snapshot:
    return Snapshot(
        url="https://example.com/",
        title="Example",
        text="hello",
        elements=elements,
        can_scroll_down=down,
        can_scroll_up=up,
        fingerprint="fp",
    )


def _answer(choice: str, *, confidence: float = 0.9) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=choice, probabilities={choice: 1.0}, confidence=confidence)


def test_only_operations_with_candidates_are_offered() -> None:
    classifier, _ = build_classifier(_snapshot([CLICK]))
    operations = classifier.questions["operation"]
    assert set(operations.criteria) == {"CLICK", "WAIT", "DONE", "BLOCKED"}  # type: ignore[union-attr]
    assert set(classifier.questions) == {"operation", "click_target"}


def test_scroll_operations_follow_scroll_flags() -> None:
    classifier, _ = build_classifier(_snapshot([], down=True, up=True))
    operations = classifier.questions["operation"]
    assert set(operations.criteria) == {"SCROLL_UP", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED"}  # type: ignore[union-attr]
    assert set(classifier.questions) == {"operation"}


def test_every_present_kind_gets_a_speculative_target_question() -> None:
    classifier, targets = build_classifier(_snapshot([CLICK, FILL, SELECT]))
    assert set(classifier.questions) == {"operation", "click_target", "type_text_target", "select_target"}
    assert classifier.questions["type_text_target"].criteria == {  # type: ignore[union-attr]
        "2": {"label": "Query", "current_value": "abc"}
    }
    assert set(targets["SELECT"]) == {"3:L"}


def test_resolve_reads_only_the_head_matching_the_operation() -> None:
    _, targets = build_classifier(_snapshot([CLICK, FILL]))
    answers = {
        "operation": _answer("CLICK", confidence=0.4),
        "click_target": _answer("1", confidence=0.95),
        "type_text_target": _answer("2", confidence=0.99),
    }
    decision = resolve_decision(answers, targets)
    assert decision.operation == "CLICK"
    assert decision.target is CLICK
    assert decision.confidence == 0.95


def test_resolve_targetless_operation_uses_operation_confidence() -> None:
    _, targets = build_classifier(_snapshot([CLICK], down=True))
    decision = resolve_decision({"operation": _answer("SCROLL_DOWN", confidence=0.7)}, targets)
    assert decision.operation == "SCROLL_DOWN"
    assert decision.target is None
    assert decision.confidence == 0.7


def test_resolve_select_target_carries_option_value() -> None:
    _, targets = build_classifier(_snapshot([SELECT]))
    answers = {"operation": _answer("SELECT"), "select_target": _answer("3:L")}
    decision = resolve_decision(answers, targets)
    assert decision.target is not None
    assert decision.target.option_value == "L"


def test_link_href_is_offered_and_non_links_carry_none() -> None:
    link = Element(id=4, role="button", kind="click", label="Microphone", current_value="", href="/wiki/Microphone")
    anchor = Element(id=5, role="button", kind="click", label="5 Equipment", current_value="", href="#Equipment")
    classifier, _ = build_classifier(_snapshot([CLICK, link, anchor]))
    criteria = classifier.questions["click_target"].criteria  # type: ignore[union-attr]
    assert criteria["4"] == {"label": "Microphone", "current_value": "", "href": "/wiki/Microphone"}
    assert criteria["5"]["href"] == "#Equipment"  # type: ignore[index]
    assert "href" not in criteria["1"]  # type: ignore[operator]
