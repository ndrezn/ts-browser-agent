# ts-browser-agent

A fast browser agent built on
[`langchain-typesafe`](https://docs.langchain.com/oss/python/integrations/providers/typesafe)
and the LangChain SDK. No third-party browser-agent package.

Each step turns the page into a numbered table of interactive elements and sends
TypeSafe one request. It asks which operation comes next
(`CLICK`/`TYPE_TEXT`/`SELECT`/`SCROLL_UP`/`SCROLL_DOWN`/`WAIT`/`DONE`/`BLOCKED`) and,
speculatively, which element each operation would target. One round trip, one decision.
A small chat model runs only for `TYPE_TEXT`, to produce the value to type.

<a href="docs/wiki_game.mp4"><img src="docs/wiki_game.gif" alt="A recorded run of examples/wiki_game.py" width="100%" /></a>

A recorded run of `examples/wiki_game.py`: reach one Wikipedia article from another by
clicking article links only. [Watch the video](docs/wiki_game.mp4).

## Quickstart: run it in LangSmith Studio

1. Install:

   ```bash
   uv sync
   uv run playwright install chromium
   ```

2. Add a `.env` based on `.env.example` and fill in your values.
3. Start the dev server:

   ```bash
   uv run langgraph dev --allow-blocking
   ```

4. Type a goal. The deep agent plans, hands each page to the `browse_fast` tool, and
   reports back. Each call opens a visible Chromium.

   > Go to https://en.wikipedia.org/wiki/Main_Page and open the article about the
   > Rosetta Stone. Tell me its first sentence.

   > Search Google Flights for round-trip flights from Montreal to Cancún, departing
   > 2027-01-15 and returning 2027-01-22, one adult in economy. Tell me the cheapest
   > price shown.

   Use concrete dates and places; the classifier cannot resolve "next month" or
   "somewhere warm". Private, loopback, and cloud-metadata addresses are blocked before
   any navigation.

## Why `langchain-typesafe`

`TypeSafeClassifier` answers categorical questions about a JSON payload in one request.
It returns probabilities over answers you offered, not generated text. A browser step
fits: which operation, and which element. Trimmed from `decision.py`:

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

Criteria are rebuilt from the live DOM each step, up to 255 per question. The target
questions ride along with the operation question, so a decision is one round trip:
about 300ms here, against 1.2–2.5s for the loop's single chat-model call.

Answers are indices into a list the code built, so model output never becomes a
selector. This project only uses `Choice`; the package also has `Noul` (yes/no) and
`Score` (rubric). The classifier is a normal `Runnable`, so `ainvoke`, `batch`,
callbacks, and LangSmith tracing all work.

```bash
uv add langchain-typesafe
```

Docs: [provider guide](https://docs.langchain.com/oss/python/integrations/providers/typesafe)
· [API reference](https://reference.langchain.com/python/integrations/langchain_typesafe/)
· [TypeSafe docs](https://docs.typesafe.ai/introduction)
· [speculative fan-out pattern](https://docs.typesafe.ai/patterns/fan-out)
· [PyPI](https://pypi.org/project/langchain-typesafe/)

## Use it as a utility

The Studio graph is two importable pieces.

### Wrap `browse_fast` in your own deep agent

`make_browse_fast_tool()` returns a LangChain tool for `create_agent` or
`create_deep_agent`. The outer model decides when to browse and with what goal; the
clicks stay with the classifier loop.

```python
from deepagents import create_deep_agent
from ts_browser_agent import make_browse_fast_tool

browse_fast = make_browse_fast_tool()  # headless=True, allow_private=False by default
agent = create_deep_agent(model="openai:gpt-5.5", tools=[browse_fast])
agent.invoke({"messages": [("user", "Find the pricing page and summarize the tiers.")]})
```

A call returns a status line plus the final page's visible text, so the caller can read
a price or a title from it. `headless`, `allow_private`, and `text_model` are fixed
when the tool is built; they are not tool arguments.

`url` comes from a model, so `ensure_navigable` checks it before any browser launches.
Private and loopback addresses are blocked unless `allow_private=True`. Link-local
addresses, which include cloud metadata endpoints, are always blocked. A prompt-injected
page cannot change this.

See `examples/deep_agent.py`.

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

`DONE` is the classifier's judgment, not a guarantee; verify before acting on it.
`allow_private=True` permits a local dev server. See `examples/flight_search.py`,
`github_issue.py`, `wiki_hop.py`, and `wiki_game.py`.

## Examples

Every example opens a visible browser so you can watch.

| Example | Path | What it exercises |
| --- | --- | --- |
| `flight_search.py` | `Agent` | A long click/select chain on real dynamic UI (Google Flights). Search only; it never books |
| `github_issue.py` | `Agent` | Structurally different UI at each step (tab, filter, list, issue) |
| `wiki_hop.py` | `Agent` | The same decision (first link in body text) made correctly many times in a row |
| `wiki_game.py` | `Agent` | Reach one article from another (`--start`, `--end`). The game's rules are enforced by filtering each snapshot in the example: no revisits, article links only, no search |
| `deep_agent.py` | deep agent | The minimal shape: one `browse_fast` call, one page-scoped goal |

```bash
uv run --env-file .env python examples/wiki_game.py --start "Jimmy Page" --end Microphone
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

Offline: dummy credentials, no browser, no network. They cover the SSRF guard,
classifier construction and resolution, snapshot indexing, the step loop's stall and
budget rules, and the tool's input bounds.

## Credits

The design follows [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) by
Browser Use: one TypeSafe request per step, a small model for typed text only, no LLM
reasoning per click. The instruction text in `decision.py` is adapted from it under the
MIT license. The implementation is independent — Playwright rather than Browser
Harness, `langchain-typesafe` rather than a custom client.
