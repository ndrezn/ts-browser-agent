"""TypeSafe as the model in a stock `create_agent` loop.

`create_agent` expects a chat model that emits tool calls and tools that return
observations. `TypeSafeBrowserModel` is that model, and it never generates text. Each
turn it reads the goal from the first human message and the current page from the last
tool result's `Snapshot` artifact, asks TypeSafe which operation and which element, and
returns an `AIMessage` with exactly one tool call — or none, which ends the run. The one
chat model in the loop is used only to produce the value for `TYPE_TEXT`.

Everything the model needs is derived from the messages it is given, so it holds no
per-run state and one instance can serve any number of runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from ts_browser_agent.decision import adecide
from ts_browser_agent.snapshot import Snapshot
from ts_browser_agent.text import agenerate_field_text, resolve_text_model

_TOOL_FOR = {
    "CLICK": "click",
    "TYPE_TEXT": "type_text",
    "SELECT": "select_option",
    "SCROLL_UP": "scroll_up",
    "SCROLL_DOWN": "scroll_down",
    "WAIT": "wait",
}
_URL = re.compile(r"https?://\S+")
_FAILED = "failed:"
_UNCHANGED = " (page did not change)"


def message_text(content: object) -> str:
    """Flatten message content to its text; some providers return a block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


@dataclass(frozen=True)
class _Step:
    """One executed action and the page it produced."""

    description: str
    tool: str
    signature: str
    failed: bool
    snapshot: Snapshot


def _as_snapshot(artifact: object) -> Snapshot | None:
    if isinstance(artifact, Snapshot):
        return artifact
    if isinstance(artifact, dict):  # a checkpointer may hand the artifact back as plain data
        try:
            return Snapshot.model_validate(artifact)
        except ValueError:
            return None
    return None


def _steps(messages: Sequence[BaseMessage]) -> list[_Step]:
    """Pair each tool-calling `AIMessage` with the `ToolMessage` that answered it."""
    steps: list[_Step] = []
    pending: AIMessage | None = None
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            pending = message
            continue
        if isinstance(message, ToolMessage) and pending is not None:
            snapshot = _as_snapshot(message.artifact)
            if snapshot is not None:
                call = pending.tool_calls[0]
                steps.append(
                    _Step(
                        description=message_text(pending.content),
                        tool=call["name"],
                        signature=f"{call['name']}:{call['args'].get('node_id')}",
                        failed=message_text(message.content).startswith(_FAILED),
                        snapshot=snapshot,
                    )
                )
            pending = None
    return steps


def _changed(steps: list[_Step], index: int) -> bool:
    return steps[index].snapshot.fingerprint != steps[index - 1].snapshot.fingerprint


def _history(steps: list[_Step]) -> list[str]:
    """Descriptions of the actions after `open`, marked when an action changed nothing."""
    return [step.description + ("" if _changed(steps, i) else _UNCHANGED) for i, step in enumerate(steps) if i > 0]


def _stall(steps: list[_Step], limit: int) -> str | None:
    """Why the run should stop, or `None` while it is still making progress.

    Repeated *effective* actions are fine — advancing a date picker four months is four
    identical clicks that each change the page. What stops a run is a tail of actions
    that changed nothing, or of failures on the same target. `WAIT` breaks a no-change
    streak, since waiting for a page to load is not a stuck loop.
    """
    if len(steps) - 1 < limit:
        return None
    tail = list(range(len(steps) - limit, len(steps)))
    if all(steps[i].failed for i in tail) and len({steps[i].signature for i in tail}) == 1:
        return f"{limit} failed attempts on the same target"
    if all(steps[i].tool != "wait" and not steps[i].failed and not _changed(steps, i) for i in tail):
        return f"{limit} actions in a row changed nothing"
    return None


def _call(name: str, args: dict[str, Any], *, content: str) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": f"call_{uuid4().hex[:12]}", "type": "tool_call"}],
    )


class TypeSafeBrowserModel(BaseChatModel):
    """The browser-driving policy, packaged as a chat model so `create_agent` runs it as is.

    Args:
        text_model: Chat model, or `init_chat_model` string, used only for `TYPE_TEXT`
            values; see `resolve_text_model`.
        max_steps: Actions allowed after the page is opened. Past it the run ends
            `BLOCKED`, through the normal exit so the browser is still closed.
        no_progress_limit: Consecutive no-change actions, or same-target failures,
            before the run ends `STALLED`.
    """

    text_model: str | BaseChatModel = "openai:gpt-5-mini"
    max_steps: int = Field(default=40, ge=1)
    no_progress_limit: int = Field(default=3, ge=1)
    _resolved_text_model: BaseChatModel | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "typesafe-browser"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Accept the agent's tools.

        Operations map to a fixed tool set, so the schemas are not consulted, but
        `create_agent` binds tools on every call and the base implementation raises.
        """
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = "TypeSafeBrowserModel is async-only; run the agent with `ainvoke` or `astream`."
        raise NotImplementedError(message)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = await self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _next(self, messages: list[BaseMessage]) -> AIMessage:
        goal = next((message_text(m.content) for m in messages if isinstance(m, HumanMessage)), "")
        steps = _steps(messages)
        if not steps:
            match = _URL.search(goal)
            if match is None:
                return AIMessage(content="BLOCKED: the goal names no URL to open.")
            url = match.group(0).rstrip(".,;:)")
            return _call("open", {"url": url}, content=f"OPEN {url}")
        if len(steps) - 1 >= self.max_steps:
            return AIMessage(content=f"BLOCKED: step budget of {self.max_steps} exhausted.")
        stalled = _stall(steps, self.no_progress_limit)
        if stalled is not None:
            return AIMessage(content=f"STALLED: {stalled}.")

        snapshot = steps[-1].snapshot
        history = _history(steps)
        decision = await adecide(snapshot, goal, history)
        if decision.operation in ("DONE", "BLOCKED"):
            return AIMessage(content=decision.operation)
        tool = _TOOL_FOR[decision.operation]
        if decision.target is None:
            return _call(tool, {}, content=decision.operation)

        element = decision.target
        value = element.option_value
        if decision.operation == "TYPE_TEXT":
            value = await agenerate_field_text(self._text_chat_model(), goal, element, snapshot, history)
        args: dict[str, Any] = {"node_id": element.id}
        if value is not None:
            args["value"] = value
        description = f"{decision.operation} '{element.label}'" + (f" = {value!r}" if value else "")
        return _call(tool, args, content=description)

    def _text_chat_model(self) -> BaseChatModel:
        if self._resolved_text_model is None:
            self._resolved_text_model = resolve_text_model(self.text_model)
        return self._resolved_text_model


__all__ = ["TypeSafeBrowserModel", "message_text"]
