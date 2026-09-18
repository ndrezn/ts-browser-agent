"""`browse_fast`'s model-facing schema: bounded budget, no policy knobs exposed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ts_browser_agent.tool import BrowseFastInput, make_browse_fast_tool

MAX_STEPS_CAP = 200
DEFAULT_MAX_STEPS = 40


def test_default_max_steps() -> None:
    assert BrowseFastInput(url="https://example.com/", goal="g").max_steps == DEFAULT_MAX_STEPS


def test_max_steps_at_cap_is_accepted() -> None:
    assert BrowseFastInput(url="https://example.com/", goal="g", max_steps=MAX_STEPS_CAP).max_steps == MAX_STEPS_CAP


@pytest.mark.parametrize("max_steps", [0, -1, MAX_STEPS_CAP + 1, 10_000])
def test_max_steps_outside_bounds_rejected(max_steps: int) -> None:
    with pytest.raises(ValidationError):
        BrowseFastInput(url="https://example.com/", goal="g", max_steps=max_steps)


def test_schema_exposes_only_url_goal_and_budget() -> None:
    tool = make_browse_fast_tool()
    assert tool.name == "browse_fast"
    assert tool.args_schema is not None
    assert set(tool.args_schema.model_fields) == {"url", "goal", "max_steps"}  # type: ignore[union-attr]
