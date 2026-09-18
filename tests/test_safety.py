"""`ensure_navigable`: numeric addresses only, so nothing here touches DNS."""

from __future__ import annotations

import pytest

from ts_browser_agent.safety import UnsafeURLError, ensure_navigable


def test_public_address_passes() -> None:
    url = "https://93.184.216.34/path?q=1"
    assert ensure_navigable(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
    ],
)
def test_private_addresses_blocked_by_default(url: str) -> None:
    with pytest.raises(UnsafeURLError, match="private"):
        ensure_navigable(url)


def test_ipv6_loopback_blocked_by_default() -> None:
    with pytest.raises(UnsafeURLError):
        ensure_navigable("http://[::1]:3000/")


@pytest.mark.parametrize("url", ["http://127.0.0.1:3000/", "http://10.0.0.5/"])
def test_private_addresses_allowed_when_opted_in(url: str) -> None:
    assert ensure_navigable(url, allow_private=True) == url


@pytest.mark.parametrize("allow_private", [False, True])
def test_cloud_metadata_blocked_even_when_private_allowed(allow_private: bool) -> None:
    with pytest.raises(UnsafeURLError, match="disallowed"):
        ensure_navigable("http://169.254.169.254/latest/meta-data/", allow_private=allow_private)


@pytest.mark.parametrize("url", ["ftp://93.184.216.34/", "file:///etc/passwd", "javascript:alert(1)"])
def test_unsupported_schemes_rejected(url: str) -> None:
    with pytest.raises(UnsafeURLError, match="scheme"):
        ensure_navigable(url)


def test_missing_hostname_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="hostname"):
        ensure_navigable("http:///just-a-path")
