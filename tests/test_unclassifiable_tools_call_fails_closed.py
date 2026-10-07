"""An unclassifiable ``tools/call`` must FAIL CLOSED, not pass through.

Why this file exists
--------------------
``classify_tool`` returns one of the three scoped tools or ``UNCLASSIFIABLE``,
and its docstring states that UNCLASSIFIABLE exists "so the caller can fail
closed". The handler did the opposite: it returned the body unchanged.

That was harmless while the vended credentials rode inside
``params.arguments`` — a pass-through added no credential, so there was nothing
to leak. The header channel inverted it. Every Lambda target now allowlists the
four ``x-tvm-*`` header names, and the interceptor's precedence over a
client-supplied header only applies to a header the interceptor WRITES. On a
pass-through it writes none, so an ``x-tvm-session-token`` supplied by the
CLIENT — the agent, i.e. the component this architecture assumes hostile — is
the only value present for an allowlisted name, and it propagates to the Lambda.

These assertions pin the refusal so the branch cannot drift back to
pass-through, and they check the two things that matter about it: that the
envelope is a short-circuit rather than a forwarded request, and that it carries
no credential header for the gateway to merge.

Note on what is NOT asserted here: whether the Gateway would route an
unclassifiable name to a Lambda at all. It answers ``-32602 Unknown tool`` for a
name it cannot resolve, so the branch should be unreachable in practice — but
that is the Gateway's behaviour, not this component's, and it stops holding the
moment a routable target is added that this classifier does not know. The point
of failing closed here is to not depend on it.
"""

from __future__ import annotations

from typing import Any

import pytest

import interceptor.handler as interceptor_handler

_CREDENTIAL_HEADER_PREFIX = "x-tvm-"


def _event(
    tool_name: str, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """Build a tools/call REQUEST-interceptor payload."""
    return {
        "mcp": {
            "gatewayRequest": {
                "headers": headers or {"Authorization": "Bearer token"},
                "body": {
                    "jsonrpc": "2.0",
                    "id": "probe-1",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": {"doc_id": "PAY-001"},
                    },
                },
            }
        }
    }


def _forwarded_headers(result: dict[str, Any]) -> dict[str, str]:
    """Headers the interceptor asked the gateway to forward, if any."""
    transformed = (result.get("mcp") or {}).get("transformedGatewayRequest") or {}
    return transformed.get("headers") or {}


def _is_short_circuit(result: dict[str, Any]) -> bool:
    """True when the envelope carries a response instead of a forwarded request."""
    return bool((result.get("mcp") or {}).get("transformedGatewayResponse"))


@pytest.mark.parametrize(
    "name",
    [
        "UnknownTarget___unknown_tool",  # routable shape, unknown trailing tool
        "evil___fake_read_document",  # crafted to end in a known tool name
        "read_document",  # no target delimiter at all
        "",  # empty name
    ],
)
def test_unclassifiable_tools_call_is_refused(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every unclassifiable name short-circuits instead of passing through."""
    vend_calls: list[Any] = []
    monkeypatch.setattr(
        interceptor_handler,
        "_vend_for_tool",
        lambda *a, **k: vend_calls.append((a, k)),
    )

    result = interceptor_handler.handler(_event(name), None)

    assert _is_short_circuit(result), (
        f"{name!r} was passed through instead of refused: the envelope carries a "
        "forwarded request, not a short-circuit response"
    )
    assert vend_calls == [], "no credential may be vended for an unclassifiable call"


@pytest.mark.parametrize(
    "name",
    ["UnknownTarget___unknown_tool", "evil___fake_read_document", ""],
)
def test_refusal_emits_no_credential_header(name: str) -> None:
    """The refusal envelope carries no ``x-tvm-*`` header for the gateway to merge."""
    result = interceptor_handler.handler(_event(name), None)

    emitted = _forwarded_headers(result)
    offenders = [h for h in emitted if h.lower().startswith(_CREDENTIAL_HEADER_PREFIX)]
    assert offenders == [], (
        f"refusal for {name!r} emitted credential headers {offenders}; a refused "
        "call must contribute no allowlisted header value"
    )


def test_a_client_supplied_credential_header_is_not_forwarded() -> None:
    """A client-sent ``x-tvm-*`` on an unclassifiable call is not forwarded.

    This is the path the refusal closes: the caller supplies the header, the
    interceptor writes none, and the target's allowlist would otherwise let the
    caller's value through. Refusing means there is no forwarded request at all.
    """
    attacker = {
        "Authorization": "Bearer token",
        "x-tvm-session-token": "attacker-chosen-session-token",
        "x-tvm-served-scope": "billing-internal",
    }

    result = interceptor_handler.handler(
        _event("UnknownTarget___unknown_tool", headers=attacker), None
    )

    forwarded = {h.lower() for h in _forwarded_headers(result)}
    assert "x-tvm-session-token" not in forwarded
    assert "x-tvm-served-scope" not in forwarded
    assert _is_short_circuit(result), (
        "the call must be refused, so the gateway never forwards the caller's "
        "allowlisted headers to a target"
    )


def test_guard_has_teeth_a_classified_tool_is_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal that fired on EVERY tools/call would pass the assertions above.

    Without this, the three tests above would measure nothing: they would be
    equally green on a handler that refuses all traffic. So a classified tool
    must get PAST the classification branch and reach the identity check.
    """
    reached: list[str] = []

    def _record_then_none(_header: Any) -> None:
        reached.append("identity")
        return None

    monkeypatch.setattr(
        interceptor_handler,
        "verified_identity_from_authorization",
        _record_then_none,
    )

    interceptor_handler.handler(_event("ReadDocument___read_document"), None)

    assert reached == ["identity"], (
        "a classified tool must get past the classification branch and reach the "
        "identity check; it was refused earlier instead"
    )
