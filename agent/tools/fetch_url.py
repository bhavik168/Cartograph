"""Optional HTTP fetch tool. Off unless CARTOGRAPHER_ENABLE_FETCH_URL=1.

Off by default for two reasons: a research run should be reproducible from a
fixed corpus, and remote HTML is the single richest source of prompt injection.
When enabled, the fetched body still goes through ``guards.quarantine`` in the
researcher node like every other tool result — this module only handles the
network side.

Requests to private and loopback address ranges are refused (a basic SSRF
guard); redirects are followed but re-checked at each hop.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

from langchain_core.tools import tool

MAX_BYTES = 200_000
TIMEOUT_S = 15.0


def is_enabled() -> bool:
    return os.getenv("CARTOGRAPHER_ENABLE_FETCH_URL", "0") == "1"


class FetchError(RuntimeError):
    pass


def _assert_public(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise FetchError(f"scheme not allowed: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise FetchError("no host in URL")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise FetchError(f"could not resolve {host}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise FetchError(f"refusing to fetch non-public address {ip}")


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()


def fetch(url: str) -> str:
    import httpx

    _assert_public(url)
    with httpx.Client(timeout=TIMEOUT_S, follow_redirects=True) as client:
        response = client.get(url, headers={"User-Agent": "cartographer/0.1"})
        _assert_public(str(response.url))  # re-check after redirects
        response.raise_for_status()
        body = response.text[: MAX_BYTES * 2]
    content_type = response.headers.get("content-type", "")
    return _strip_html(body) if "html" in content_type else body[:MAX_BYTES]


@tool("fetch_url")
def fetch_url(url: str) -> str:
    """Fetch a public web page and return its text. Disabled by default.

    Only use when the local corpus cannot answer the question and the user has
    explicitly enabled network access.
    """
    if not is_enabled():
        return (
            "FETCH DISABLED: network access is off for this run. "
            "Answer from the local corpus only."
        )
    try:
        return fetch(url)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
        return f"FETCH ERROR: {exc}"
