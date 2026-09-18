"""`Element`/`Snapshot` indexing: target keys and per-kind grouping."""

from __future__ import annotations

from ts_browser_agent.snapshot import Element, Snapshot


def _snapshot(elements: list[Element]) -> Snapshot:
    return Snapshot(
        url="https://example.com/",
        title="Example",
        text="hello",
        elements=elements,
        can_scroll_down=False,
        can_scroll_up=False,
        fingerprint="fp",
    )


def test_click_target_key_is_node_id() -> None:
    element = Element(id=7, role="button", kind="click", label="Go", current_value="")
    assert element.target_key == "7"


def test_select_target_key_includes_option_value() -> None:
    element = Element(
        id=3, role="combobox", kind="select", label="Size -> L", current_value="M", option_value="L"
    )
    assert element.target_key == "3:L"


def test_targets_group_candidates_by_kind() -> None:
    snapshot = _snapshot(
        [
            Element(id=1, role="button", kind="click", label="Go", current_value=""),
            Element(id=2, role="textbox", kind="fill", label="Query", current_value=""),
            Element(id=3, role="combobox", kind="select", label="Size -> L", current_value="", option_value="L"),
            Element(id=3, role="combobox", kind="select", label="Size -> M", current_value="", option_value="M"),
        ]
    )
    assert set(snapshot.targets("click")) == {"1"}
    assert set(snapshot.targets("fill")) == {"2"}
    assert set(snapshot.targets("select")) == {"3:L", "3:M"}
    assert snapshot.targets("select")["3:M"].option_value == "M"
