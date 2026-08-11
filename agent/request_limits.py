"""Request-body size bound for the agent's ``POST /invocations`` endpoint.

``/invocations`` takes the raw Starlette ``Request`` (it must: the AgentCore
Runtime delivers the payload body directly, and the endpoint tolerates two
payload shapes), which means it bypasses the ``InvocationRequest`` pydantic
model and with it any declarative size bound. Nothing else in the stack supplies
one either: uvicorn has no request-body-size flag, and AgentCore's own platform
request limits are an unverified backstop rather than a control this repository
owns. So ``await request.json()`` buffered a body of any size in-container.

Reachability bounds the exposure -- the runtime carries no
``authorizer_configuration``, so only IAM principals allowed to call
``InvokeAgentRuntime`` reach this endpoint -- but an unbounded read is the kind of
gap that stops being theoretical the moment the runtime is fronted by something
broader. This module supplies the cap.

Two checks, because either alone is bypassable:

* ``check_content_length`` rejects a declared oversize body BEFORE any of it is
  read. Cheap, but trusts a header the client controls.
* ``read_bounded_body`` caps the actual bytes read, aborting as soon as the
  running total exceeds the limit. This is the load-bearing one: it holds when
  ``Content-Length`` is absent, unparseable or a lie (a chunked body carries no
  ``Content-Length`` at all).

Deliberately stdlib-only and free of any framework import: the agent's own
dependencies (fastapi, strands) are not part of the test toolchain, so keeping
this module import-light is what makes the bound directly testable. It consumes
any async iterable of ``bytes``, which ``Request.stream()`` satisfies.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterable

# 256 KiB. An invocation payload is a prompt, an optional model id and a Cognito
# access token -- kilobytes, not megabytes. Sized to leave generous headroom for a
# long prompt while still being orders of magnitude below what an unbounded read
# would accept.
DEFAULT_MAX_BODY_BYTES = 256 * 1024

# Optional deploy-time override, read from the environment like every other knob
# in this container (BEDROCK_MODEL_ID, AWS_REGION, GATEWAY_URL). Unset is the
# normal case and yields DEFAULT_MAX_BODY_BYTES.
MAX_BODY_BYTES_ENV_VAR = "MAX_REQUEST_BODY_BYTES"


class BodyTooLarge(Exception):
    """Raised when a request body exceeds the configured limit.

    Attributes:
        limit: The byte limit that was exceeded.
    """

    def __init__(self, limit: int) -> None:
        """Initialise with the limit that was exceeded.

        Args:
            limit: The byte limit in force when the body was rejected.
        """
        self.limit = limit
        # The message names the limit but NOT the observed size: the caller turns
        # this into a 413 detail, and echoing how much was received tells a
        # prober how far it got.
        super().__init__(f"Request body exceeds the {limit}-byte limit.")


def max_body_bytes() -> int:
    """Return the byte limit for a request body.

    Reads ``MAX_REQUEST_BODY_BYTES`` from the environment, falling back to
    ``DEFAULT_MAX_BODY_BYTES``. A value that is not a positive integer is
    IGNORED in favour of the default rather than honoured or raised on: a typo in
    a deploy-time variable must not be able to switch the bound off (``0``) or
    crash every invocation.

    Returns:
        The limit in bytes, always positive.
    """
    raw = os.environ.get(MAX_BODY_BYTES_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_MAX_BODY_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES
    if value <= 0:
        return DEFAULT_MAX_BODY_BYTES
    return value


def check_content_length(header_value: str | None, limit: int | None = None) -> None:
    """Reject a body whose declared ``Content-Length`` exceeds the limit.

    A fast pre-read rejection only. An absent, non-numeric or understated header
    passes here by design -- ``read_bounded_body`` is what actually enforces the
    bound on the bytes that arrive.

    Args:
        header_value: The raw ``Content-Length`` header value, or None.
        limit: Byte limit to apply. Defaults to ``max_body_bytes()``.

    Raises:
        BodyTooLarge: If the header declares more bytes than the limit allows.
    """
    effective = max_body_bytes() if limit is None else limit
    if not header_value:
        return
    try:
        declared = int(header_value)
    except ValueError:
        return
    if declared > effective:
        raise BodyTooLarge(effective)


class MalformedBody(Exception):
    """Raised when a bounded body is not a JSON object.

    Carries the message the caller should return to the client verbatim: the
    distinction between "not JSON" and "not an object" is useful to a legitimate
    caller and reveals nothing.
    """


async def read_bounded_body(
    chunks: AsyncIterable[bytes], limit: int | None = None
) -> bytes:
    """Read a request body, aborting once it exceeds the limit.

    The running total is checked per chunk and the read abandoned on the first
    chunk that crosses the limit, so an oversized body is never fully buffered --
    which is the whole point of the bound.

    Args:
        chunks: Async iterable of body chunks (``Request.stream()`` satisfies it).
        limit: Byte limit to apply. Defaults to ``max_body_bytes()``.

    Returns:
        The complete body as bytes, when it fits within the limit.

    Raises:
        BodyTooLarge: As soon as the accumulated body exceeds the limit.
    """
    effective = max_body_bytes() if limit is None else limit
    parts: list[bytes] = []
    total = 0
    async for chunk in chunks:
        if not chunk:
            continue
        total += len(chunk)
        if total > effective:
            raise BodyTooLarge(effective)
        parts.append(chunk)
    return b"".join(parts)


async def parse_bounded_json(
    content_length: str | None,
    chunks: AsyncIterable[bytes],
    limit: int | None = None,
) -> dict[str, Any]:
    """Read, bound and parse a JSON-object request body.

    The whole sequence the endpoint needs, in one framework-free call: reject a
    declared oversize body, cap the bytes actually read, parse, and require a
    JSON object. Keeping it here rather than inline in the endpoint is what makes
    the endpoint's behaviour testable -- ``main.py`` cannot be imported without
    fastapi, so logic left inline there is logic no test can reach. The endpoint
    is reduced to mapping these two exception types onto status codes.

    Args:
        content_length: The raw ``Content-Length`` header value, or None.
        chunks: Async iterable of body chunks (``Request.stream()`` satisfies it).
        limit: Byte limit to apply. Defaults to ``max_body_bytes()``.

    Returns:
        The parsed body as a dict.

    Raises:
        BodyTooLarge: If the body exceeds the limit (caller returns 413).
        MalformedBody: If the body is not valid JSON, or is valid JSON that is
            not an object (caller returns 400).
    """
    check_content_length(content_length, limit)
    raw = await read_bounded_body(chunks, limit)

    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        # Covers an empty body too: json.loads(b"") raises. An empty body is a
        # malformed request, not a request that happens to carry no fields.
        raise MalformedBody("Request body is not valid JSON.") from exc

    if not isinstance(parsed, dict):
        # Reading the body by hand means the framework no longer guarantees this:
        # a bare JSON list, string or number would otherwise reach .get().
        raise MalformedBody("Request body must be a JSON object.")

    return parsed
