"""
Fail-CLOSED regression tests for the RESPONSE interceptor's exception path.

The interceptor is the last line of defence against credential material leaving on
a tool reply, and its ``except Exception`` branch used to return the ORIGINAL,
unscrubbed body on every failure. Combined with an uncapped recursive scrub over an
uncapped ``copy.deepcopy``, that made the exception path a *scrubber bypass*: nest a
reply deeply enough, the scrub dies with ``RecursionError`` part-way through, and
the raw body — credentials intact — is handed back to the caller. The count-only
``response_scrub removed=0`` log made the bypass look exactly like a clean reply.

These tests pin the two halves of the fix so neither can regress:

* **Depth cap (``credential_scrubber``).** Nesting past ``_MAX_BODY_DEPTH`` raises
  ``DepthLimitExceeded`` deterministically, BEFORE ``copy.deepcopy`` gets a chance
  to exhaust the stack, and the exception message carries no body content. The
  budget spans the whole walk, so a chain of JSON-encoded-inside-JSON strings
  cannot buy fresh recursion allowance one decode at a time — and a payload the JSON
  DECODER itself cannot bottom out is refused rather than quietly downgraded to a
  text-only scan that never applies the credential-key-name rule.
* **Fail closed (``handler``).** When the scrub raises, the handler withholds the
  reply — a JSON-RPC error body, not the original — logs a distinguishable
  ``stage=scrub withheld=1`` line, and still never raises out to the gateway. A
  failure AFTER a successful scrub returns the scrubbed body instead, so a broken
  log handler cannot cost availability or leak the original. The handshake
  pass-through the fail-open behaviour was defending is verified to still work:
  ``initialize`` / ``tools/list`` replies round-trip deep-equal on both the
  method-gate and the scrub route, and a non-``tools/call`` method still passes
  through untouched.

The headline assertion is the one the old suite never made: after the exception
path runs, the credential material is NOT in the returned body.

Import resolution: ``from handler import handler`` and
``from credential_scrubber import ...`` resolve to the ``response_interceptor/``
modules, which the root ``conftest.py`` puts on ``sys.path`` (they ship flat at the
zip Lambda's archive root).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import tracemalloc
from collections.abc import Callable
from typing import Any

import pytest

import credential_scrubber
import handler as handler_module
from credential_scrubber import (
    DepthLimitExceeded,
    ScrubError,
    _MAX_BODY_DEPTH,
    scrub,
)
from handler import handler

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: The interceptor output envelope version the handler must always emit.
_OUTPUT_VERSION = "1.0"

#: JSON-RPC error code the handler reports for a withheld reply ("Internal error").
_WITHHELD_ERROR_CODE = -32603

#: Cap on the per-entry errors a withheld batch reply is answered with (mirrors the
#: handler's own bound, imported rather than duplicated so the two cannot drift).
_MAX_WITHHELD_BATCH_ITEMS = handler_module._MAX_WITHHELD_BATCH_ITEMS

#: Exception-path log line: `error=1`, the stage that failed, and the withheld flag —
#: no body content. The stage token is what tells an operator which of the three
#: exception paths ran, so a failed scrub can never be read as a clean reply.
_ERROR_LOG_PATTERN = re.compile(
    r"^response_scrub error=1 stage=(envelope|scrub|post_scrub) withheld=([01]) removed=(\d+)$"
)

#: Real-shaped credential material. If any of these strings appears in a handler
#: result, the scrubber was bypassed — which is precisely what this file forbids.
_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_SESSION_TOKEN = "FQoGZXIvYXdzE" + "AbCdEfGh1234567890" * 20

#: Nesting depth that trips the cap with room to spare, while staying well inside
#: what a caller could actually send.
_OVER_DEEP = _MAX_BODY_DEPTH + 25

#: Nesting depth past what the RECURSIVE machinery itself survives. Measured on
#: CPython 3.14 at the default 1000-frame limit, `copy.deepcopy` gives out from ~500
#: containers, so twice the recursion limit clears every recursive stage by a wide
#: margin on any interpreter. A body this deep is what proves the pre-copy check runs
#: BEFORE the copy: without it, `RecursionError` — not `DepthLimitExceeded` — is what
#: comes out.
_PAST_RECURSION_CEILING = sys.getrecursionlimit() * 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_dict(node: Any) -> dict[str, Any]:
    """Wrap ``node`` in one dict level."""
    return {"n": node}


def _wrap_list(node: Any) -> list[Any]:
    """Wrap ``node`` in one list level."""
    return [node]


#: Both container kinds, so every depth test covers the `_scrub_list` arm of the
#: guard as well as `_scrub_dict`. A dict-only test leaves the list arm and its
#: `depth + 1` threading unpinned.
_WRAPPERS = [pytest.param(_wrap_dict, id="dict"), pytest.param(_wrap_list, id="list")]


def _nest(leaf: Any, depth: int, wrap: Callable[[Any], Any] = _wrap_dict) -> Any:
    """Wrap ``leaf`` in ``depth`` nested containers.

    Args:
        leaf: The innermost value.
        depth: Number of enclosing containers to build.
        wrap: Container constructor — :func:`_wrap_dict` or :func:`_wrap_list`.

    Returns:
        The outermost container of the nested chain.
    """
    node = leaf
    for _ in range(depth):
        node = wrap(node)
    return node


def _credential_leaf() -> dict[str, Any]:
    """A leaf carrying all three credential shapes, under STS-style key names."""
    return {
        "AccessKeyId": _ACCESS_KEY_ID,
        "SecretAccessKey": _SECRET_ACCESS_KEY,
        "SessionToken": _SESSION_TOKEN,
    }


def _tools_call_event(body: Any, *, status_code: int = 200) -> dict[str, Any]:
    """Build a RESPONSE interceptor input envelope for a ``tools/call`` reply.

    The originating method is included so the method gate does NOT skip the scan —
    this is the credential-bearing path.

    Args:
        body: The reply body at ``mcp.gatewayResponse.body``.
        status_code: The gateway response status code to carry.

    Returns:
        A well-formed interceptor input event.
    """
    return {
        "mcp": {
            "gatewayRequest": {"body": {"jsonrpc": "2.0", "id": 7, "method": "tools/call"}},
            "gatewayResponse": {"statusCode": status_code, "body": body},
        }
    }


def _transformed(result: dict[str, Any]) -> dict[str, Any]:
    """Extract ``mcp.transformedGatewayResponse`` from a handler result."""
    return result["mcp"]["transformedGatewayResponse"]


def _assert_valid_envelope(result: dict[str, Any]) -> None:
    """Assert ``result`` is a well-formed ``interceptorOutputVersion: "1.0"`` envelope."""
    assert result["interceptorOutputVersion"] == _OUTPUT_VERSION
    assert set(_transformed(result).keys()) == {"statusCode", "body"}


def _assert_no_credentials(value: Any) -> None:
    """Assert no credential material survives anywhere in ``value``.

    Serialises the whole value so a credential is caught at ANY depth and under any
    key, rather than only where the test happened to look.

    Args:
        value: The body (or log text) to search.
    """
    rendered = value if isinstance(value, str) else json.dumps(value)
    for material in (_ACCESS_KEY_ID, _SECRET_ACCESS_KEY, _SESSION_TOKEN):
        assert material not in rendered, "credential material survived the scrubber"


def _error_log_flags(caplog: pytest.LogCaptureFixture) -> tuple[str, int, int]:
    """Return ``(stage, withheld, removed)`` from the single exception-path log line.

    Selects the ``error=1`` line specifically: the ``post_scrub`` path legitimately
    emits the success count first and then the error line, so filtering on the
    ``response_scrub`` prefix alone would see two records. Exactly one ERROR line must
    be present.

    Args:
        caplog: pytest log-capture fixture.

    Returns:
        The stage token and the two integers the detail-free error line carries.
    """
    matches = [
        match
        for match in (_ERROR_LOG_PATTERN.match(r.getMessage()) for r in caplog.records)
        if match is not None
    ]
    assert len(matches) == 1, (
        "expected exactly one detail-free error line, got "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    match = matches[0]
    return match.group(1), int(match.group(2)), int(match.group(3))


# ---------------------------------------------------------------------------
# Depth cap — the scrubber refuses instead of dying half-way
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrap", _WRAPPERS)
def test_scrub_accepts_a_body_exactly_at_the_depth_cap(wrap: Callable[[Any], Any]) -> None:
    """The deepest ACCEPTED node sits at exactly the cap — the boundary, not near it.

    ``_nest(leaf, _MAX_BODY_DEPTH)`` puts the leaf at depth ``_MAX_BODY_DEPTH``
    exactly, so this pairs with the rejection test below to pin the comparison as
    ``>`` and not ``>=``. Asserting only that "something deep-ish works" would let
    an off-by-one through.
    """
    scrubbed, removed = scrub(_nest("leaf", _MAX_BODY_DEPTH, wrap))

    assert removed == 0
    assert scrubbed == _nest("leaf", _MAX_BODY_DEPTH, wrap)


@pytest.mark.parametrize("wrap", _WRAPPERS)
def test_scrub_rejects_one_container_past_the_cap(wrap: Callable[[Any], Any]) -> None:
    """One container past the cap is refused — the other half of the boundary."""
    with pytest.raises(DepthLimitExceeded):
        scrub(_nest("leaf", _MAX_BODY_DEPTH + 1, wrap))


@pytest.mark.parametrize("wrap", _WRAPPERS)
def test_scrub_still_removes_credentials_just_under_the_cap(
    wrap: Callable[[Any], Any],
) -> None:
    """A legitimately deep body is scrubbed, not refused: the cap is not over-eager."""
    scrubbed, removed = scrub(_nest(_credential_leaf(), _MAX_BODY_DEPTH - 1, wrap))

    assert removed == 3
    _assert_no_credentials(scrubbed)


@pytest.mark.parametrize("wrap", _WRAPPERS)
def test_scrub_raises_depth_limit_instead_of_recursion_error(
    wrap: Callable[[Any], Any],
) -> None:
    """Past the recursive machinery's OWN ceiling, the typed error still comes out.

    This is the test that pins the pre-copy ordering. At ``_PAST_RECURSION_CEILING``
    the recursive ``copy.deepcopy`` cannot complete, so if the iterative check did
    not run BEFORE it, ``RecursionError`` would escape instead — non-deterministic,
    and dependent on how much stack the caller had left. ``pytest.raises`` here is
    exact: ``RecursionError`` is not a ``DepthLimitExceeded``, so the wrong exception
    fails the test rather than satisfying it.
    """
    body = _nest(_credential_leaf(), _PAST_RECURSION_CEILING, wrap)

    with pytest.raises(DepthLimitExceeded) as excinfo:
        scrub(body)

    # Catchable as the family the handler keys its fail-closed branch on.
    assert isinstance(excinfo.value, ScrubError)
    # The message names the limit and nothing else — no body content.
    _assert_no_credentials(str(excinfo.value))
    assert "AccessKeyId" not in str(excinfo.value)


@pytest.mark.parametrize("wrap", _WRAPPERS)
def test_scrub_depth_budget_spans_embedded_json_chains(
    wrap: Callable[[Any], Any],
) -> None:
    """A JSON-encoded payload cannot restart the depth budget.

    Rule 3 decodes a JSON-encoded string and keeps scrubbing the containers it
    finds. If each decode restarted the depth budget, an attacker-influenced string
    payload could re-create unbounded recursion by chaining decodes — each one
    individually shallow. The budget is therefore cumulative across the walk.

    The two halves are each ~55 containers: neither the outer nesting nor the decoded
    payload exceeds the cap of 100 alone, and only their SUM does. A per-structure
    budget would accept this body.
    """
    half = (_MAX_BODY_DEPTH // 2) + 5
    encoded_payload = json.dumps(_nest(_credential_leaf(), half, wrap))
    body = _nest({"text": encoded_payload}, half, wrap)

    with pytest.raises(DepthLimitExceeded):
        scrub(body)


def test_embedded_json_boundary_is_pinned_at_the_exact_container_count() -> None:
    """The IN-WALK cap boundary is pinned, not just the pre-copy one.

    The pre-copy check catches over-deep bodies before the walk, so for a plain
    structural body it — not ``_assert_depth_budget`` — decides the boundary. Only
    containers decoded out of an embedded JSON string reach the in-walk comparison, so
    that comparison has to be pinned HERE or a one-container shift in it goes
    unnoticed.

    ``k`` outer dicts plus the ``{"text": ...}`` wrapper plus the ``m + 1`` containers
    decoded out of the payload give ``k + 1 + m`` levels below the top: 99 is the
    deepest accepted body and 100 is refused.
    """
    def _chain(outer: int, payload_depth: int) -> dict[str, Any]:
        encoded = json.dumps(_nest(_credential_leaf(), payload_depth))
        return _nest({"text": encoded}, outer)

    # Exactly at the limit: scrubbed normally, credential removed.
    scrubbed, removed = scrub(_chain(49, 49))
    assert removed == 3
    _assert_no_credentials(scrubbed)

    # Exactly one container further: refused.
    with pytest.raises(DepthLimitExceeded):
        scrub(_chain(49, 50))


def test_embedded_json_list_chain_is_bounded_by_the_list_guard() -> None:
    """The LIST arm of the depth guard is load-bearing on its own.

    Every other depth case ends at a dict — the credential leaf — so the dict guard
    fires and the list guard could be deleted unnoticed. Here the decoded payload is
    pure list nesting with a STRING leaf, so no dict is ever entered: only
    ``_scrub_list``'s own check can stop the recursion.
    """
    half = (_MAX_BODY_DEPTH // 2) + 5
    encoded_payload = json.dumps(_nest(_SESSION_TOKEN, half, _wrap_list))
    body = _nest({"text": encoded_payload}, half, _wrap_list)

    with pytest.raises(DepthLimitExceeded):
        scrub(body)


def test_embedded_json_too_deep_for_the_decoder_is_refused_not_downgraded() -> None:
    """A payload the JSON DECODER cannot bottom out must NOT fall back to text rules.

    ``json.loads`` raises ``RecursionError`` from ~116k nesting levels — well under
    the 1 MiB embedded-JSON size gate, so a document body can carry one. Swallowing
    that error and applying only the plain-text rules would leave rule 1 (credential
    KEY names) unapplied while still reporting ``removed=0``: a reply that looks
    clean and was never structurally scrubbed. It must raise instead, which routes it
    to the handler's fail-closed branch.
    """
    depth = sys.getrecursionlimit() * 200  # ~200k: past the decoder, under the size gate
    payload = "[" * depth + '{"session_token": "unreachable"}' + "]" * depth
    assert len(payload) < 1_048_576, "payload must stay under _MAX_EMBEDDED_JSON_CHARS"

    with pytest.raises(DepthLimitExceeded):
        scrub({"result": {"content": [{"text": payload}]}})


def test_scrub_leaves_ordinary_nesting_untouched() -> None:
    """A realistic reply shape is unaffected: no false depth rejection, no-op preserved."""
    body = {
        "jsonrpc": "2.0",
        "id": 4,
        "result": {
            "content": [{"type": "text", "text": json.dumps({"doc_id": "d-1", "body": "text"})}],
            "isError": False,
        },
    }

    scrubbed, removed = scrub(body)

    assert removed == 0
    assert scrubbed == body


# ---------------------------------------------------------------------------
# Handler — fail CLOSED when the scrub cannot vouch for the body
# ---------------------------------------------------------------------------


def test_over_deep_tools_call_reply_is_withheld_not_passed_through(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """THE regression: an over-deep ``tools/call`` reply must not ship its credentials.

    Before the fix this exact input returned the raw body — ``AKIA...``,
    secret access key and session token intact — because the scrub's
    ``RecursionError`` was swallowed and the original echoed. The handler must now
    withhold the body, and must still not raise out to the gateway.
    """
    body = {"jsonrpc": "2.0", "id": 7, "result": _nest(_credential_leaf(), _OVER_DEEP)}

    with caplog.at_level(logging.INFO):
        result = handler(_tools_call_event(body), None)

    _assert_valid_envelope(result)
    returned = _transformed(result)["body"]

    # The bypass is closed: nothing credential-shaped comes back.
    _assert_no_credentials(returned)
    # And nothing of the original body comes back either — it was withheld whole.
    assert returned["error"]["code"] == _WITHHELD_ERROR_CODE
    assert "result" not in returned

    # The failure is visible in the logs instead of masquerading as a clean scrub.
    assert _error_log_flags(caplog) == ("scrub", 1, 0)
    _assert_no_credentials(caplog.text)


def test_withheld_reply_is_a_correlatable_json_rpc_error() -> None:
    """The withheld body is a usable JSON-RPC error carrying the original id.

    An MCP client is blocked on a specific request id. Returning ``{}`` would be a
    protocol violation that can leave it waiting, so the replacement body is a
    well-formed error object it can correlate and fail fast on.
    """
    body = {"jsonrpc": "2.0", "id": "req-42", "result": _nest(_credential_leaf(), _OVER_DEEP)}

    result = handler(_tools_call_event(body, status_code=200), None)
    returned = _transformed(result)["body"]

    assert returned["jsonrpc"] == "2.0"
    assert returned["id"] == "req-42"
    assert returned["error"]["code"] == _WITHHELD_ERROR_CODE
    assert isinstance(returned["error"]["message"], str)
    # The status code is echoed unchanged: a JSON-RPC-level error, not an HTTP one.
    assert _transformed(result)["statusCode"] == 200


def test_withheld_reply_tolerates_a_missing_or_illegal_id() -> None:
    """A body with no usable JSON-RPC id still yields a valid error (``id: null``).

    ``True`` is a ``bool``, hence an ``int`` subclass, but not a legal JSON-RPC id —
    so it must be dropped rather than echoed.
    """
    for original_id in ({}, {"id": True}, {"id": ["not", "scalar"]}):
        body = {"jsonrpc": "2.0", **original_id, "result": _nest(_credential_leaf(), _OVER_DEEP)}

        returned = _transformed(handler(_tools_call_event(body), None))["body"]

        assert returned["id"] is None
        _assert_no_credentials(returned)


def test_withheld_batch_reply_keeps_the_array_shape() -> None:
    """A withheld BATCH reply comes back as an array of per-entry errors.

    A JSON-RPC batch reply is a top-level array. Answering it with a single object
    would hand the client a shape it is not parsing and ids it cannot correlate —
    the very failure the withheld body exists to avoid.
    """
    batch = [
        {"jsonrpc": "2.0", "id": 1, "result": _nest(_credential_leaf(), _OVER_DEEP)},
        {"jsonrpc": "2.0", "id": "two", "result": {"ok": True}},
    ]

    returned = _transformed(handler(_tools_call_event(batch), None))["body"]

    assert isinstance(returned, list)
    assert [entry["id"] for entry in returned] == [1, "two"]
    assert all(entry["error"]["code"] == _WITHHELD_ERROR_CODE for entry in returned)
    _assert_no_credentials(returned)


def test_withheld_batch_reply_is_bounded() -> None:
    """A long batch is truncated instead of inflating into a much larger reply.

    Per-entry errors are ~120 bytes each. Emitting one per entry for an arbitrarily
    long batch of small entries would turn a withheld reply into an amplifier, so the
    array stops at the cap — exactly at it, so the truncation is pinned rather than
    merely bounded. Ids past the cap are not answered; that is the deliberate trade.
    """
    batch: list[Any] = [{"jsonrpc": "2.0", "id": index} for index in range(500)]
    batch[0] = {"jsonrpc": "2.0", "id": 0, "result": _nest(_credential_leaf(), _OVER_DEEP)}

    returned = _transformed(handler(_tools_call_event(batch), None))["body"]

    assert isinstance(returned, list)
    assert len(returned) == _MAX_WITHHELD_BATCH_ITEMS
    assert [entry["id"] for entry in returned] == list(range(_MAX_WITHHELD_BATCH_ITEMS))
    _assert_no_credentials(returned)


def test_a_late_failure_on_the_method_gate_path_still_returns_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fault after the method gate must not turn a handshake reply into ``{}``.

    The gate's pass-through body is cleared for return by design, so it is published
    before the gate returns. Otherwise a fault on this branch falls through to the
    envelope-read default and answers an ``initialize`` with an empty body — precisely
    the handshake-breaking shape the pass-through exists to prevent, on the branch
    the handshake travels through.
    """
    real_transformed = handler_module._transformed_response
    calls = {"n": 0}

    def _fail_first(status_code: Any, body: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated late failure on the pass-through path")
        return real_transformed(status_code, body)

    monkeypatch.setattr(handler_module, "_transformed_response", _fail_first)
    initialize_reply = {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
    event = {
        "mcp": {
            "gatewayRequest": {"body": {"jsonrpc": "2.0", "id": 1, "method": "initialize"}},
            "gatewayResponse": {"statusCode": 200, "body": initialize_reply},
        }
    }

    result = handler(event, None)

    assert _transformed(result)["body"] == initialize_reply


def test_withheld_message_does_not_advertise_the_scrubber() -> None:
    """The client-facing message stays generic; the specifics go to the logs.

    A caller acting on attacker-influenced document content should not be handed
    confirmation that a credential scrubber sits in the path, nor a signal it can
    bisect the depth cap against. The operator still gets the detail — the log line
    carries ``stage=scrub``.
    """
    body = {"jsonrpc": "2.0", "id": 1, "result": _nest(_credential_leaf(), _OVER_DEEP)}

    returned = _transformed(handler(_tools_call_event(body), None))["body"]
    message = returned["error"]["message"].lower()

    for internal in ("scrub", "credential", "depth", "recursion", "nest"):
        assert internal not in message, f"client-facing message leaks internals: {internal}"


def test_a_failure_after_a_successful_scrub_returns_the_scrubbed_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A post-scrub failure returns the SCRUBBED body — neither the original nor nothing.

    The scrub succeeded, so a clean body already exists; withholding it would be a
    needless outage, and falling back to ``original_body`` would be a leak. Simulated
    by failing the FIRST envelope build, i.e. a step after the scrub and before the
    return; the except path's own build then has to succeed.
    """
    real_transformed = handler_module._transformed_response
    calls = {"n": 0}

    def _fail_first(status_code: Any, body: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated post-scrub failure")
        return real_transformed(status_code, body)

    monkeypatch.setattr(handler_module, "_transformed_response", _fail_first)
    body = {"jsonrpc": "2.0", "id": 1, "result": _credential_leaf()}

    with caplog.at_level(logging.INFO):
        result = handler(_tools_call_event(body), None)

    returned = _transformed(result)["body"]
    # The scrubbed body came back: credentials gone, but the reply is still a reply.
    _assert_no_credentials(returned)
    assert "result" in returned
    assert _error_log_flags(caplog) == ("post_scrub", 0, 0)


def test_a_broken_log_handler_never_escapes_the_interceptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A logging failure must not turn a recoverable reply into a gateway 5xx.

    The recovery branches log too, so an unguarded log call there would be the one
    place an exception could still escape ``handler`` — after which the gateway
    returns 5xx instead of the response. Logging is observability, never a reason to
    fail a reply.
    """

    def _always_raise(self: logging.Logger, msg: str, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated log handler failure")

    monkeypatch.setattr(logging.Logger, "info", _always_raise)

    # Both a clean reply and one that trips the fail-closed path must survive.
    clean = handler(_tools_call_event({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}), None)
    assert _transformed(clean)["body"] == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}

    over_deep = {"jsonrpc": "2.0", "id": 2, "result": _nest(_credential_leaf(), _OVER_DEEP)}
    withheld = handler(_tools_call_event(over_deep), None)
    _assert_no_credentials(_transformed(withheld)["body"])


def test_a_credential_shaped_id_is_not_echoed_into_the_withheld_body() -> None:
    """The withheld body must not smuggle credentials out through the ``id`` field.

    The id is the only field copied out of a reply the scrubber never vouched for. A
    response is not obliged to carry a sane one, so an over-deep reply whose ``id``
    IS an access key would otherwise put that key straight into the body built to
    withhold it.
    """
    for hostile_id in (_ACCESS_KEY_ID, _SECRET_ACCESS_KEY, _SESSION_TOKEN):
        body = {
            "jsonrpc": "2.0",
            "id": hostile_id,
            "result": _nest(_credential_leaf(), _OVER_DEEP),
        }

        returned = _transformed(handler(_tools_call_event(body), None))["body"]

        assert returned["id"] is None, "a credential-shaped id must not be echoed"
        _assert_no_credentials(returned)


@pytest.mark.parametrize(
    "candidate,echoed",
    [
        ("req-42", True),
        ("7f3a9c2e-1b4d-4e8a-9f2c-3d5b6a7c8e9f", True),
        ("call:7", True),
        ("x" * 129, False),  # too long to be a correlation id
        ('id"><script>', False),  # outside the id charset
        ("id with spaces", False),
    ],
)
def test_only_vetted_string_ids_are_echoed(candidate: str, echoed: bool) -> None:
    """Realistic ids survive; anything that is not shaped like one becomes ``null``."""
    body = {"jsonrpc": "2.0", "id": candidate, "result": _nest(_credential_leaf(), _OVER_DEEP)}

    returned = _transformed(handler(_tools_call_event(body), None))["body"]

    assert returned["id"] == (candidate if echoed else None)


def test_the_depth_guard_memory_is_bounded_by_depth_not_by_payload_size() -> None:
    """The pre-copy guard must not allocate in proportion to a wide, shallow body.

    The guard runs BEFORE ``copy.deepcopy``, on an attacker-influenced payload. If it
    enqueued every scalar it saw, a long flat list would make it allocate
    proportionally to the reply before the copy even started — turning a depth guard
    into a memory amplifier. Retaining only the ancestor chain keeps its footprint
    bounded by the cap.
    """
    wide_body = {"result": [f"item-{index}" for index in range(200_000)]}

    tracemalloc.start()
    try:
        credential_scrubber._assert_within_depth(wide_body)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Ancestor-only bookkeeping is a handful of iterators; per-scalar bookkeeping for
    # 200k entries would be megabytes.
    assert peak < 100_000, f"depth guard allocated {peak} bytes on a wide body"


def test_unexpected_scrub_failure_also_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fail-closed is keyed on "the scrub failed", not on the depth cap specifically.

    The depth cap is the one failure mode we know about today. A future defect in
    the scrubber must not re-open the bypass, so the handler withholds on ANY
    exception raised from the scrub — simulated here with an unrelated
    ``ValueError``.
    """

    def _boom(_value: Any) -> tuple[Any, int]:
        raise ValueError("simulated scrubber defect")

    monkeypatch.setattr("handler.scrub", _boom)
    body = {"jsonrpc": "2.0", "id": 1, "result": _credential_leaf()}

    with caplog.at_level(logging.INFO):
        result = handler(_tools_call_event(body), None)

    _assert_valid_envelope(result)
    returned = _transformed(result)["body"]
    _assert_no_credentials(returned)
    assert returned["error"]["code"] == _WITHHELD_ERROR_CODE
    assert _error_log_flags(caplog) == ("scrub", 1, 0)


def test_a_malformed_gateway_request_does_not_disable_fail_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unusable ``gatewayRequest`` still scrubs — and still fails closed.

    The method gate is an optimization: when the method cannot be read it must fall
    back to scrubbing, not to skipping. So a garbled request shape around an
    over-deep reply must still end in a withheld body, never a pass-through.
    """
    over_deep_reply = {
        "jsonrpc": "2.0",
        "id": 9,
        "result": _nest(_credential_leaf(), _OVER_DEEP),
    }
    event = {
        "mcp": {
            "gatewayRequest": {"body": "not-a-dict"},
            "gatewayResponse": {"statusCode": 200, "body": over_deep_reply},
        }
    }

    with caplog.at_level(logging.INFO):
        result = handler(event, None)

    _assert_no_credentials(_transformed(result)["body"])
    assert _error_log_flags(caplog) == ("scrub", 1, 0)


# ---------------------------------------------------------------------------
# The handshake pass-through the fail-open branch was protecting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("include_request", [True, False], ids=["method-gate", "scrub-path"])
@pytest.mark.parametrize(
    "method,body",
    [
        (
            "initialize",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "support-tools-gateway", "version": "1.0.0"},
                },
            },
        ),
        (
            "tools/list",
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "ReadDocument"}]}},
        ),
        ("ping", {"jsonrpc": "2.0", "id": 3, "result": {}}),
    ],
)
def test_protocol_replies_still_round_trip_deep_equal(
    method: str, body: dict[str, Any], include_request: bool
) -> None:
    """Failing closed did not cost the MCP handshake.

    This is the regression the fail-open branch existed to prevent: a prior
    interceptor blocked ``initialize`` and broke the handshake. Protocol replies must
    still come back byte-identical on BOTH routes, which is why the request envelope
    is parametrized: with ``gatewayRequest`` present the method gate skips the scan,
    and with it omitted the scrub actually runs and has to be a structural no-op. A
    method-gate-only test could not fail if the no-op property broke.
    """
    mcp: dict[str, Any] = {"gatewayResponse": {"statusCode": 200, "body": body}}
    if include_request:
        mcp["gatewayRequest"] = {"body": {"jsonrpc": "2.0", "id": 1, "method": method}}

    result = handler({"mcp": mcp}, None)

    assert _transformed(result)["body"] == body


def test_non_tools_call_reply_is_never_withheld() -> None:
    """A non-``tools/call`` reply keeps its pass-through even when it is over-deep.

    The method gate returns before the scrub, so protocol traffic can never be
    withheld by the depth cap — fail-closed applies only where credentials would
    actually be.
    """
    body = {"jsonrpc": "2.0", "id": 5, "result": _nest({"deep": True}, _OVER_DEEP)}
    event = {
        "mcp": {
            "gatewayRequest": {"body": {"jsonrpc": "2.0", "id": 5, "method": "tools/list"}},
            "gatewayResponse": {"statusCode": 200, "body": body},
        }
    }

    result = handler(event, None)

    assert _transformed(result)["body"] == body
