"""Indexed DOM snapshots.

Reads the page atomically, once per step, and returns a stable, numbered table of
interactive elements. The step loop only ever refers to elements by that number;
`browser.py` re-resolves geometry and visibility immediately before every action.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel

# Assigns a persistent integer id to every interactive node the first time it is seen
# (via a WeakMap keyed by the DOM node) so an index survives re-snapshots as long as the
# node itself is still attached. `SELECT` options are exploded into one candidate per
# option, matching how `TYPE_TEXT`/`CLICK`/`SELECT` targets are offered independently.
_SNAPSHOT_JS = r"""
() => {
  const store = (window.__tsFastAgent ??= { ids: new WeakMap(), nodes: new Map(), next: 1 });
  const identify = (el) => {
    if (!store.ids.has(el)) store.ids.set(el, store.next++);
    const id = store.ids.get(el);
    store.nodes.set(id, el);
    return id;
  };
  const visible = (el) =>
    el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }) &&
    !el.closest("[aria-hidden='true'],[inert]");
  // Computes an element's accessible name the way assistive tech does: explicit
  // labelling first, then "name from content" — recursively concatenating child
  // text and any labelled descendants. A widget with no label of its own (a
  // calendar day cell wrapping a div whose *own* aria-label carries the real date)
  // only gets a meaningful name through this last step, not a single attribute
  // check — so this is general, not calendar-specific.
  const accessibleName = (el, seen = new Set()) => {
    if (!el || seen.has(el)) return "";
    seen.add(el);
    const labelledBy = (el.getAttribute("aria-labelledby") || "")
      .split(/\s+/)
      .filter(Boolean)
      .map((id) => accessibleName(document.getElementById(id), seen))
      .filter(Boolean)
      .join(" ");
    if (labelledBy) return labelledBy.trim();
    const ariaLabel = el.getAttribute("aria-label");
    if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();
    if (el.labels && el.labels.length) {
      const fromLabels = [...el.labels].map((l) => accessibleName(l, seen)).filter(Boolean).join(" ");
      if (fromLabels) return fromLabels;
    }
    if (el.tagName !== "INPUT") {
      const fromContent = [...el.childNodes]
        .map((node) => {
          if (node.nodeType === Node.TEXT_NODE) return node.textContent;
          if (node.nodeType === Node.ELEMENT_NODE && node.getAttribute("aria-hidden") !== "true") {
            return accessibleName(node, seen);
          }
          return "";
        })
        .join(" ")
        .trim();
      if (fromContent) return fromContent;
    }
    return (el.getAttribute("placeholder") || el.getAttribute("title") || el.getAttribute("alt") || "").trim();
  };
  const roleOf = (el) => {
    const tag = el.tagName;
    if (tag === "A" || tag === "BUTTON") return "button";
    if (tag === "SELECT") return "combobox";
    if (tag === "TEXTAREA") return "textbox";
    if (tag === "INPUT") {
      if (["submit", "button", "reset", "image"].includes(el.type)) return "button";
      if (["checkbox", "radio"].includes(el.type)) return el.type;
      return "textbox";
    }
    return el.getAttribute("role");
  };
  const editable = (el) =>
    (el.tagName === "TEXTAREA" ||
      (el.tagName === "INPUT" && !["submit", "button", "reset", "checkbox", "radio", "image"].includes(el.type))) &&
    !el.readOnly;

  // TypeSafe's `Choice` questions cap out at 255 candidates. Restricting to what is
  // actually within the viewport keeps candidate counts bounded on any real page and
  // matches the mental model: elements below the fold are reached by scrolling first,
  // not offered as an immediate target.
  // Autocomplete and menu widgets (a location suggestion, a label filter) render as
  // ARIA roles on plain <li>/<div> elements, not the native tags above — without
  // these, typing into a field like a destination search box produces a suggestion
  // list the agent can see but never has a way to click.
  const elements = [];
  const selector =
    "a[href],button,input,textarea,select," +
    "[role='button'],[role='link'],[role='option'],[role='menuitem']," +
    "[role='menuitemradio'],[role='menuitemcheckbox'],[role='tab']," +
    "[role='checkbox'],[role='radio'],[role='switch']";
  for (const el of document.querySelectorAll(selector)) {
    if (elements.length >= 200) break;
    if (el.disabled || !visible(el)) continue;
    // Largest rendered fragment, not the union box: a wrapped inline link's union
    // center can sit on text that isn't the link. Mirrors the click-point logic in
    // browser.py so the on-screen test here judges the same point that gets clicked.
    const fragments = [...el.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
    if (!fragments.length) continue;
    const rect = fragments.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b));
    const cx = rect.x + rect.width / 2;
    const cy = rect.y + rect.height / 2;
    if (cx < 0 || cy < 0 || cx >= innerWidth || cy >= innerHeight) continue;
    const id = identify(el);
    const label = (accessibleName(el) || roleOf(el) || "element").replace(/\s+/g, " ").trim();
    if (el.tagName === "SELECT") {
      for (const option of el.options) {
        if (elements.length >= 200) break;
        if (option.disabled) continue;
        elements.push({
          id,
          option_value: option.value,
          role: "combobox",
          kind: "select",
          label: `${label} -> ${option.label}`,
          current_value: el.selectedOptions[0]?.label || "",
        });
      }
      continue;
    }
    elements.push({
      id,
      role: roleOf(el) || "generic",
      kind: editable(el) ? "fill" : "click",
      label,
      current_value: "value" in el ? String(el.value) : "",
      // The raw attribute, not the resolved URL: "#Equipment" says same-page anchor and
      // "/wiki/Microphone" says where a link goes, which a label alone cannot.
      href: el.tagName === "A" ? el.getAttribute("href") : null,
    });
  }

  const bodyText = document.body.innerText.slice(0, 4000);
  const canScrollDown = window.scrollY + window.innerHeight < document.documentElement.scrollHeight - 2;
  const canScrollUp = window.scrollY > 0;
  return {
    url: location.href,
    title: document.title,
    text: bodyText,
    elements,
    can_scroll_down: canScrollDown,
    can_scroll_up: canScrollUp,
  };
}
"""


class Element(BaseModel):
    """One candidate the step loop can act on.

    `id` is the stable per-node identifier assigned by the injected snapshot script; it
    is not a DOM selector and is only ever resolved back to a live node inside the page,
    immediately before an action executes.
    """

    id: int
    role: str
    kind: Literal["click", "fill", "select"]
    label: str
    current_value: str
    option_value: str | None = None
    href: str | None = None

    @property
    def target_key(self) -> str:
        """The `Choice` candidate key offered to `TypeSafeClassifier`."""
        return str(self.id) if self.kind != "select" else f"{self.id}:{self.option_value}"


class Snapshot(BaseModel):
    """One atomic read of the page: visible text plus the current element table."""

    url: str
    title: str
    text: str
    elements: list[Element]
    can_scroll_down: bool
    can_scroll_up: bool
    fingerprint: str

    def targets(self, kind: Literal["click", "fill", "select"]) -> dict[str, Element]:
        """Return candidates of one action kind, keyed by their `Choice` target key."""
        return {el.target_key: el for el in self.elements if el.kind == kind}


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_snapshot(page: Any) -> Snapshot:
    """Read the page once and return its indexed element table.

    Args:
        page: A Playwright `Page` (or `async_api.Page`) that has finished loading.

    Returns:
        The current snapshot, including a fingerprint used to detect staleness.
    """
    raw = page.evaluate(_SNAPSHOT_JS)
    elements = [Element.model_validate(el) for el in raw["elements"]]
    fingerprint = _fingerprint(
        {"url": raw["url"], "text": raw["text"], "elements": raw["elements"]}
    )
    return Snapshot(
        url=raw["url"],
        title=raw["title"],
        text=raw["text"],
        elements=elements,
        can_scroll_down=raw["can_scroll_down"],
        can_scroll_up=raw["can_scroll_up"],
        fingerprint=fingerprint,
    )


__all__ = ["Element", "Snapshot", "read_snapshot"]
