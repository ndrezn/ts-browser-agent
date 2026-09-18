"""Shared fixtures: every test runs offline with dummy credentials."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _dummy_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # `TypeSafeClassifier` refuses to construct without a key; a dummy satisfies
    # validation and is never sent anywhere, since no test issues a request.
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("LANGSMITH_GATEWAY", raising=False)
