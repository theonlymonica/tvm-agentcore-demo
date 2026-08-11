"""Property test — the REQUEST interceptor never mutates its input event.

The property: for ANY REQUEST-interceptor input event — a scoped
``tools/call``, an out-of-scope ``tools/call``, or a non-``tools/call``
protocol message (``initialize`` / ``tools/list`` /
``notifications/initialized`` / ``ping``), with arbitrary headers, params, and
arguments (optionally including a model-supplied ``arguments["context"]``) —
after ``interceptor.handler.handler(event, None)`` returns:

* (a) the input ``event`` is deep-equal to a ``copy.deepcopy`` snapshot taken
  BEFORE the call (the handler mutated no part of the inbound event at any
  depth, because it deep-copies the request body before every write); and
* (b) none of the credential values the interceptor vended for this call — the
  ``access_key_id`` / ``secret_access_key`` / ``session_token`` it generated for
  ``context.tenant_credentials`` — appear ANYWHERE in the input ``event`` at any
  depth (walked recursively over dict keys/values, list items, and strings, and
  never as a substring of any string value).

This property does NOT assert the absence of a model-supplied ``context`` key:
the model may itself place a value at ``arguments["context"]``, and the
deep-equal check in (a) PRESERVES it. The security-bearing invariant is
non-mutation of the inbound event plus the absence of the vended credential
values from it.

Only the two external boundaries are stubbed, following the monkeypatch pattern
used by the interceptor's context wire-contract test and
``tests/test_context_injection.py``:
``interceptor.handler.served_scope_from_authorization`` (returns a fixed
derivable scope so the scoped path proceeds to vend + inject) and
``interceptor.handler._vend_for_tool`` (returns a known single credentials dict
so the test knows exactly which credential strings to search for).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import interceptor.handler as interceptor_handler

# --- Fixed, derivable served scope. Its value is irrelevant to the property: it
# is injected into the interceptor's DEEP COPY of the body, never into the input
# event, so it is not searched for below. -------------------------------------
_SERVED_SCOPE = "payments-core"

# --- Frozen composite tool names: the three scoped tools, plus
# out-of-scope names, plus the non-``tools/call`` protocol methods. -----------
_SCOPED_COMPOSITE_NAMES = (
    "ReadDocument___read_document",
    "SearchDocuments___search_documents",
    "Reply___reply",
)
_OUT_OF_SCOPE_NAMES = (
    "OtherTarget___not_a_scoped_tool",
    "evil___fake_read_document",
    "NoDelimiterName",
)
_NON_CALL_METHODS = (
    "initialize",
    "tools/list",
    "notifications/initialized",
    "ping",
)

# --- Disjoint alphabets so a vended credential value can NEVER collide with
# arbitrary generated event content. Event content is lowercase-only; credential
# values carry an UPPERCASE sentinel prefix, so ``cred_value in event_string`` is
# always False for a genuinely unmutated event — a substring match can therefore
# only mean the handler leaked a credential into the input event. --------------
_EVENT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 -_/.:"
_CRED_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_event_text = st.text(alphabet=_EVENT_ALPHABET, max_size=24)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


def _json_values() -> st.SearchStrategy[Any]:
    """Arbitrary JSON-like values (str/int/bool/None and nested dict/list)."""
    return st.recursive(
        st.one_of(st.none(), st.booleans(), st.integers(), _event_text),
        lambda children: st.one_of(
            st.lists(children, max_size=3),
            st.dictionaries(_event_text, children, max_size=3),
        ),
        max_leaves=6,
    )


def _arguments() -> st.SearchStrategy[dict[str, Any]]:
    """Arbitrary model-supplied ``params.arguments`` maps."""
    return st.dictionaries(_event_text, _json_values(), max_size=4)


def _headers() -> st.SearchStrategy[dict[str, str]]:
    """Arbitrary request headers (string -> string)."""
    return st.dictionaries(_event_text, _event_text, max_size=3)


@st.composite
def _credential_values(draw: st.DrawFn) -> dict[str, str]:
    """Three known credential strings, each with an uppercase sentinel prefix.

    The uppercase prefix guarantees disjointness from the lowercase event
    content, so any occurrence of one of these strings inside the input event
    can only come from the handler leaking it — never from a generation
    collision.
    """

    def _cred(prefix: str) -> str:
        return prefix + draw(
            st.text(alphabet=_CRED_ALPHABET, min_size=8, max_size=20)
        )

    return {
        "access_key_id": _cred("CREDVALUEAKID"),
        "secret_access_key": _cred("CREDVALUESECRET"),
        "session_token": _cred("CREDVALUETOKEN"),
    }


@st.composite
def _interceptor_events(draw: st.DrawFn) -> dict[str, Any]:
    """Build an arbitrary REQUEST-interceptor input event across all paths.

    Covers the three routing categories the property must hold over: a scoped
    ``tools/call`` (credentials are vended + injected), an out-of-scope
    ``tools/call`` (pass-through), and a non-``tools/call`` protocol message
    (pass-through) — each with arbitrary headers, params, and arguments, and an
    optional model-supplied ``arguments["context"]``.
    """
    category = draw(st.sampled_from(("scoped", "out_of_scope", "non_call")))
    headers = draw(_headers())
    # Optionally carry an Authorization header (its value is irrelevant — the JWT
    # boundary is stubbed — but its presence exercises the header read path).
    if draw(st.booleans()):
        headers["Authorization"] = "bearer " + draw(_event_text)
    req_id = draw(st.one_of(st.integers(), _event_text, st.none()))

    if category == "non_call":
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": draw(st.sampled_from(_NON_CALL_METHODS)),
        }
        if draw(st.booleans()):
            body["params"] = draw(
                st.dictionaries(_event_text, _json_values(), max_size=3)
            )
    else:
        names = (
            _SCOPED_COMPOSITE_NAMES
            if category == "scoped"
            else _OUT_OF_SCOPE_NAMES
        )
        arguments = draw(_arguments())
        # Optionally include a model-supplied context: the deep-equal
        # check MUST preserve it (the handler overwrites context in the copy only).
        if draw(st.booleans()):
            arguments["context"] = draw(_json_values())
        body = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": draw(st.sampled_from(names)),
                "arguments": arguments,
            },
        }

    return {"mcp": {"gatewayRequest": {"headers": headers, "body": body}}}


# ---------------------------------------------------------------------------
# Recursive credential search
# ---------------------------------------------------------------------------


def _walk_strings(value: Any) -> Iterator[str]:
    """Yield every string reachable in a nested structure (dict keys + values,
    list/tuple items), so a leaked credential is caught at any depth."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


# ---------------------------------------------------------------------------
# The non-mutation property
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(event=_interceptor_events(), credentials=_credential_values())
def test_request_interceptor_no_event_mutation(
    event: dict[str, Any],
    credentials: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REQUEST interceptor never mutates its input event.

    For all REQUEST-interceptor input events (scoped / out-of-scope
    ``tools/call`` and non-``tools/call`` methods, arbitrary headers / params /
    arguments, optionally a model-supplied ``context``), after the handler
    returns the input ``event`` is deep-equal to its pre-call snapshot and none
    of the vended credential values appear anywhere in it at any depth.
    """
    # Stub the two external boundaries. Derive a fixed scope so the scoped path
    # proceeds to vend + inject, and vend the KNOWN credential values so the test
    # knows exactly which strings to search for. ``_vend_for_tool`` returns a
    # single credentials dict; a fresh dict is returned each call so the handler
    # cannot alias the test's copy.
    monkeypatch.setattr(
        interceptor_handler,
        "served_scope_from_authorization",
        lambda _authorization: _SERVED_SCOPE,
    )
    monkeypatch.setattr(
        interceptor_handler,
        "_vend_for_tool",
        lambda _tool, _scope: dict(credentials),
    )

    snapshot = copy.deepcopy(event)

    interceptor_handler.handler(event, None)

    # (a) The inbound event is untouched at any depth — the handler wrote only
    # into a deep copy of the body. A model-supplied arguments["context"] is
    # preserved by this check, not asserted away.
    assert event == snapshot

    # (b) No vended credential value appears anywhere in the input event, at any
    # depth, not even as a substring of any string value or key.
    event_strings = list(_walk_strings(event))
    for credential_value in credentials.values():
        for text in event_strings:
            assert credential_value not in text, (
                "vended credential value leaked into the input event"
            )
