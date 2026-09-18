"""URL safety checks for browser navigation.

Once `Agent` is wrapped as a deep-agent tool, its `url` argument is chosen by an LLM
rather than typed by a person — a prompt-injected page could talk the outer agent into
re-invoking the tool against an internal service or a cloud metadata endpoint. This
module is the one checkpoint every `Browser` navigation passes through before it runs.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeURLError(ValueError):
    """`url` resolves to an address browser navigation must not reach."""


def ensure_navigable(url: str, *, allow_private: bool = False) -> str:
    """Validate `url` before a `Browser` navigates to it.

    Args:
        url: The URL to check.
        allow_private: Allow loopback and private-network addresses, for testing
            against a local dev server. Link-local addresses — which include cloud
            metadata endpoints such as `169.254.169.254` — are always blocked
            regardless of this flag.

    Returns:
        `url`, unchanged, once it passes every check.

    Raises:
        UnsafeURLError: If the scheme is unsupported, the hostname does not resolve,
            or any resolved address is disallowed.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        message = f"Unsupported URL scheme: {parsed.scheme!r}."
        raise UnsafeURLError(message)
    if not parsed.hostname:
        message = "URL has no hostname."
        raise UnsafeURLError(message)

    try:
        addr_info = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as error:
        message = f"Could not resolve hostname {parsed.hostname!r}."
        raise UnsafeURLError(message) from error

    for _family, _type, _proto, _canonname, sockaddr in addr_info:
        address = ipaddress.ip_address(sockaddr[0])
        if address.is_link_local or address.is_multicast or address.is_unspecified or address.is_reserved:
            message = f"URL for {parsed.hostname!r} resolves to a disallowed address: {address}."
            raise UnsafeURLError(message)
        if not allow_private and (address.is_private or address.is_loopback):
            message = (
                f"URL for {parsed.hostname!r} resolves to a private address: {address}. "
                "Pass allow_private=True to permit this, e.g. for a local dev server."
            )
            raise UnsafeURLError(message)
    return url


__all__ = ["UnsafeURLError", "ensure_navigable"]
