# ts-browser-agent

A fast browser agent built on
[`langchain-typesafe`](https://docs.langchain.com/oss/python/integrations/providers/typesafe)
and the LangChain SDK, with no dependency on any third-party browser-agent package.

Each step reads the page into a numbered table of interactive elements and sends
TypeSafe one request: a `Choice` question for the next operation
(`CLICK`/`TYPE_TEXT`/`SELECT`/`SCROLL_UP`/`SCROLL_DOWN`/`WAIT`/`DONE`/`BLOCKED`), plus a
speculative `Choice` question per operation for its candidate targets. Only the target
answer matching the chosen operation is used, but all arrive in one round trip. A small
chat model runs only for `TYPE_TEXT`, to produce that field's value.

## Quickstart: run it in LangSmith Studio

Type a goal into Studio; the built-in deep agent plans, hands each page to the
`browse_fast` tool, and reports back.

```bash
uv sync
uv run playwright install chromium
cp .env.example .env
uv run langgraph dev --allow-blocking
```

In `.env`, `TYPESAFE_API_KEY` is required; TypeSafe is separate from LangSmith and has no
gateway route, so a LangSmith key will not work for it. `LANGSMITH_API_KEY` with
`LANGSMITH_GATEWAY=true` routes the `TYPE_TEXT` chat model through the
[LangSmith gateway](https://docs.langchain.com/langsmith/llm-gateway), so no
`OPENAI_API_KEY` is needed.

`langgraph dev` serves the graph in `studio.py` (see `langgraph.json`) on
`127.0.0.1:2024`, or another port if that one is taken, and opens Studio at the URL it
prints. Try:

> Go to https://en.wikipedia.org/wiki/Main_Page and open the article about the Rosetta
> Stone. Tell me its first sentence.

> Search Google Flights for round-trip flights from Montreal to Cancún, departing
> 2027-01-15 and returning 2027-01-22, one adult in economy. Tell me the cheapest price
> shown.

Every `browse_fast` call opens a visible Chromium. Use concrete dates and places; the
classifier matches your words against the page and cannot resolve "next month" or
"somewhere warm". Goals naming private, loopback, or cloud-metadata addresses are
blocked before any navigation (URL policy below).

`--allow-blocking` is required because the dev server rejects synchronous I/O on its
event loop, and both Playwright's sync API and `TypeSafeClassifier`'s HTTP client are
synchronous. It is a dev-only override; a deployment would need an async tool path,
which this project does not have yet.

## Why `langchain-typesafe`

`TypeSafeClassifier` answers categorical questions about a JSON payload in one request
and returns a probability over the answers you offered, not generated text. A browser
step has that shape: which operation, and which element on the page. Trimmed from what
`decision.py` builds each step:

```python
from langchain_typesafe import Choice, TypeSafeClassifier

classifier = TypeSafeClassifier(
    questions={
        "operation": Choice(
            instructions="Advance the goal from the current page using one operation.",
            criteria={"CLICK": "Click an element.", "TYPE_TEXT": "Type into a field.", "DONE": "..."},
        ),
        "click_target": Choice(
            instructions="Choose the best target if the next operation is CLICK.",
            criteria={"14": {"label": "Where to?"}, "21": {"label": "Cancún, Mexico"}},
        ),
    }
)
response = classifier.invoke({"goal": goal, "page": page, "elements": elements})
response.choices["operation"].choice            # "CLICK"
response.choices["click_target"].probabilities  # {"21": 0.91, "14": 0.09}
```

Criteria are rebuilt from the live DOM every step (up to 255 per question), and the
target questions ride along with the operation question, so a whole decision is one
round trip: about 300ms here, against 1.2–2.5s for the loop's one chat-model call.
Because an answer is an index into a list the code built, model output never becomes a
selector or an action that is not on the page. `Choice` is all this project uses; the
package also has `Noul` (yes/no) and `Score` (rubric). The classifier is a normal
`Runnable`, so `ainvoke`, `batch`, callbacks, and LangSmith tracing work as usual.

```bash
uv add langchain-typesafe
```

Docs: [provider guide](https://docs.langchain.com/oss/python/integrations/providers/typesafe)
· [API reference](https://reference.langchain.com/python/integrations/langchain_typesafe/)
· [TypeSafe docs](https://docs.typesafe.ai/introduction)
· [speculative fan-out pattern](https://docs.typesafe.ai/patterns/fan-out)
· [PyPI](https://pypi.org/project/langchain-typesafe/)

## Use it as a utility

The Studio graph is two importable pieces assembled.

### Wrap `browse_fast` in your own deep agent

`make_browse_fast_tool()` returns a LangChain tool for `create_agent` or
`create_deep_agent`. The outer model decides when to browse and with what goal; every
click inside a call stays with the classifier loop.

```python
from deepagents import create_deep_agent
from ts_browser_agent import make_browse_fast_tool

browse_fast = make_browse_fast_tool()  # headless=True, allow_private=False by default
agent = create_deep_agent(model="openai:gpt-5.5", tools=[browse_fast])
agent.invoke({"messages": [("user", "Find the pricing page and summarize the tiers.")]})
```

A call returns a status line plus the visible text of the final page, so the caller can
read a price or a title from it. `headless`, `allow_private`, and `text_model` are set
when the tool is built and are not tool arguments. Since `url` comes from a model,
`ensure_navigable` checks it before any browser launches. Private and loopback
addresses are blocked unless `allow_private=True`; link-local addresses, which include
cloud metadata endpoints, are always blocked. A prompt-injected page cannot loosen this,
because `allow_private` is not an argument.

`examples/deep_agent.py` and `flight_search_generic.py` use this path.

### Drive `Agent` directly

You get the per-step trace and the final status:

```python
from ts_browser_agent import Agent

with Agent(
    "https://en.wikipedia.org/wiki/Main_Page",
    "Open the article about the Rosetta Stone.",
) as agent:
    for state in agent.run():
        print(state["step"], state["operation"], state["target"])
    print(agent.status)
```

`DONE` is the classifier's judgment that the goal is visibly satisfied, not a
guarantee; verify before acting on it. Pass `allow_private=True` to test against a
local dev server. `examples/run.py`, `local_form.py`, `flight_search.py`,
`github_issue.py`, and `wiki_hop.py` use this path.

## Examples

All but `run.py` and `local_form.py` open a visible browser.

| Example | Path | What it exercises |
| --- | --- | --- |
| `run.py` | `Agent` | Generic CLI: `--url` and `--goal` in, the step trace out |
| `local_form.py` | `Agent` | `TYPE_TEXT` and `allow_private=True`, against a throwaway local server |
| `flight_search.py` | `Agent` | A long click/select chain on real dynamic UI (Google Flights). Search only; it never books |
| `github_issue.py` | `Agent` | Structurally different UI at each step (tab, filter, list, issue) |
| `wiki_hop.py` | `Agent` | The same decision (first link in body text) made correctly many times in a row |
| `deep_agent.py` | deep agent | The minimal shape: one `browse_fast` call, one page-scoped goal |
| `flight_search_generic.py` | deep agent | An open-ended question ("cheapest week this winter?") one run cannot answer. The deep agent runs several concrete searches, reads prices from each report, and recommends a week; every call is streamed so retries are visible |

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal "Open the article about the Rosetta Stone."
uv run --env-file .env python examples/flight_search_generic.py
```

## Layout

| File | Job |
| --- | --- |
| `snapshot.py` | Indexed DOM read: assigns stable ids to interactive elements |
| `decision.py` | Builds the per-step `TypeSafeClassifier` and resolves its answer |
| `text.py` | Small-model text generation for `TYPE_TEXT`, kept out of the decision path |
| `browser.py` | Playwright execution, re-validating each target immediately before acting |
| `safety.py` | SSRF guard applied to every navigation, since `url` may come from an LLM |
| `agent.py` | The step loop tying the above together |
| `tool.py` | Wraps `Agent` as a `browse_fast` tool for `create_agent`/`create_deep_agent` |
| `studio.py` | Exposes the deep agent as a `graph` for `langgraph dev` / LangSmith Studio |

## Tests

```bash
uv run pytest
```

Offline, with dummy credentials and no browser or network. They cover the SSRF guard,
classifier construction and answer resolution, snapshot indexing, the step loop's stall
and budget rules (against a fake browser), and the tool's input bounds.

## Credits

The design follows [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) by
Browser Use — one TypeSafe request per step decides operation and target, a small model
produces only typed text, and no LLM reasons about individual clicks. The instruction
text in `decision.py` is adapted from it under the MIT license. The implementation is
independent: Playwright rather than Browser Harness, `langchain-typesafe` rather than a
custom client.
