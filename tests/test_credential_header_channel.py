"""The credential HEADER channel, end to end on both sides.

REPLACES ``tests/test_read_document_context_spike.py`` and
``tests/test_context_injection.py``, which proved the RETIRED in-body ``context``
wire contract. Those files were not deleted to reduce coverage: every property
they asserted is re-asserted here against the channel that now carries the
credentials. The mapping, so the trade is auditable:

===========================================  ==================================
retired assertion                             where it lives now
===========================================  ==================================
context shape is exactly scope + creds        test_exactly_four_headers_and_no_more
only the three named fields travel            test_exactly_four_headers_and_no_more
all three scoped tools get the handover       test_every_scoped_tool_gets_headers
model-supplied context is overwritten         test_body_is_forwarded_byte_identical
                                              (+ the Gateway's own precedence,
                                              which is why the body is irrelevant)
authorization header casing (3 cases)         test_authorization_header_casing
absent authorization still fails closed       test_absent_authorization_fails_closed
tool returns table + scope                    test_tool_reads_scope_from_headers
fail closed: missing/non-object/empty/partial test_tool_fails_closed_* (parametrized
                                              in test_tool_session_factory.py)
missing context -> generic error              test_tool_returns_generic_error
===========================================  ==================================

Two properties are NEW, and they are the ones the in-body design could not have:

* ``test_body_is_forwarded_byte_identical`` — the request body is now deep-equal
  to what arrived, so the credential is not merely absent from the tool's schema,
  it is absent from the wire the model's output travels on.
* ``test_oversize_value_fails_closed`` — the 4096-byte per-value limit. AWS does
  not document a maximum STS session-token size, so the guard is what stands
  between a future larger token and a silently truncated credential.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

import interceptor.handler as interceptor_handler
from common.credentials_context import (
    PROPAGATED_HEADERS_KEY,
    ScopedCredentialsError,
    served_scope_from_context,
)
from header_context import (
    ACCESS_KEY_ID_HEADER,
    SECRET_ACCESS_KEY_HEADER,
    SERVED_SCOPE_HEADER,
    SESSION_TOKEN_HEADER,
    credential_context,
    lambda_context,
)
from interceptor.credential_headers import (
    CREDENTIAL_HEADERS,
    MAX_HEADER_VALUE_BYTES,
    CredentialHeaderError,
    build_credential_headers,
    header_size_report,
)
from interceptor.jwt_claims import VerifiedIdentity

_SUBJECT = "11111111-2222-3333-4444-555555555555"
_SCOPE = "payments-core"


def _creds() -> dict[str, str]:
    """Return a fixed, obviously-fake vended credential triple."""
    return {
        "access_key_id": "ASIAFAKEKEYFORTESTS",
        "secret_access_key": "fake-secret-access-key",
        "session_token": "fake-session-token",
    }


# ---------------------------------------------------------------------------
# Interceptor side: what goes on the wire
# ---------------------------------------------------------------------------


class TestInterceptorEmitsHeaders:
    """The interceptor propagates the scope and credentials as headers only."""

    @pytest.fixture(autouse=True)
    def _stub_vend_and_scope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Avoid real STS / JWT: fixed identity, fixed vended credentials."""
        monkeypatch.setattr(
            interceptor_handler,
            "verified_identity_from_authorization",
            lambda _auth: VerifiedIdentity(served_scope=_SCOPE, subject=_SUBJECT),
        )
        monkeypatch.setattr(
            interceptor_handler,
            "_vend_for_tool",
            lambda _tool, _scope, **_kwargs: _creds(),
        )

    @staticmethod
    def _event(
        tool_name: str,
        arguments: dict[str, Any],
        auth_header: str = "Authorization",
    ) -> dict[str, Any]:
        """Build a tools/call REQUEST-interceptor payload."""
        return {
            "mcp": {
                "gatewayRequest": {
                    "headers": {auth_header: "Bearer token"},
                    "body": {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": arguments},
                    },
                }
            }
        }

    @staticmethod
    def _headers_of(result: dict[str, Any]) -> dict[str, str]:
        return result["mcp"]["transformedGatewayRequest"]["headers"]

    def test_exactly_four_headers_and_no_more(self) -> None:
        """The envelope carries exactly the four declared headers, with the vended values.

        'And no more' is the assertion that matters: an extra header would be an
        extra value crossing into the tool, and the allowlist has a budget.
        """
        result = interceptor_handler.handler(
            self._event("ReadDocument___read_document", {"doc_id": "PAY-001"}), None
        )
        headers = self._headers_of(result)

        assert set(headers) == set(CREDENTIAL_HEADERS)
        assert headers[SERVED_SCOPE_HEADER] == _SCOPE
        assert headers[ACCESS_KEY_ID_HEADER] == _creds()["access_key_id"]
        assert headers[SECRET_ACCESS_KEY_HEADER] == _creds()["secret_access_key"]
        assert headers[SESSION_TOKEN_HEADER] == _creds()["session_token"]

    @pytest.mark.parametrize(
        ("tool_name", "arguments"),
        [
            ("ReadDocument___read_document", {"doc_id": "PAY-001"}),
            ("SearchDocuments___search_documents", {"query": "refund"}),
            ("Reply___reply", {"doc_id": "PAY-001", "body": "hello"}),
        ],
    )
    def test_every_scoped_tool_gets_headers(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """All three scoped tools receive the credential headers.

        A target left out would have the Gateway drop its credentials and fail
        closed on every call, which looks like a tool bug rather than a wiring one.
        """
        result = interceptor_handler.handler(self._event(tool_name, arguments), None)
        assert set(self._headers_of(result)) == set(CREDENTIAL_HEADERS)

    @pytest.mark.parametrize(
        ("tool_name", "arguments"),
        [
            ("ReadDocument___read_document", {"doc_id": "PAY-001"}),
            ("SearchDocuments___search_documents", {"query": "refund"}),
            ("Reply___reply", {"doc_id": "PAY-001", "body": "hello"}),
            # A model that writes its own `context` / credential-looking arguments
            # changes nothing: the body is forwarded as-is and the tool never reads
            # it, so there is no key for a hostile value to land on.
            (
                "ReadDocument___read_document",
                {
                    "doc_id": "PAY-001",
                    "context": {"served_scope": "billing-internal"},
                    "x-tvm-served-scope": "billing-internal",
                },
            ),
        ],
    )
    def test_body_is_forwarded_byte_identical(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """The forwarded body is deep-equal to the one that arrived.

        This is the property the in-body design could not have. The model's output
        travels on the body; the credential travels on the headers; the two no
        longer share a channel, so nothing the model writes can name, read, or
        displace a credential.
        """
        event = self._event(tool_name, arguments)
        original = copy.deepcopy(event["mcp"]["gatewayRequest"]["body"])

        result = interceptor_handler.handler(event, None)

        assert result["mcp"]["transformedGatewayRequest"]["body"] == original
        # And the input event itself was not mutated (event immutability).
        assert event["mcp"]["gatewayRequest"]["body"] == original

    @pytest.mark.parametrize(
        "auth_header", ["authorization", "Authorization", "AUTHORIZATION"]
    )
    def test_authorization_header_casing(self, auth_header: str) -> None:
        """Any casing of Authorization still produces the credential headers.

        HTTP/2 lowercases field names on the wire, so a case-sensitive lookup here
        would fail closed on perfectly valid tokens.
        """
        result = interceptor_handler.handler(
            self._event(
                "ReadDocument___read_document", {"doc_id": "PAY-001"}, auth_header
            ),
            None,
        )
        assert set(self._headers_of(result)) == set(CREDENTIAL_HEADERS)

    def test_absent_authorization_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No verifiable identity short-circuits: no headers, no target call."""
        monkeypatch.setattr(
            interceptor_handler,
            "verified_identity_from_authorization",
            lambda _auth: None,
        )
        event = self._event("ReadDocument___read_document", {"doc_id": "PAY-001"})
        event["mcp"]["gatewayRequest"]["headers"] = {}

        result = interceptor_handler.handler(event, None)

        # transformedGatewayResponse means the Gateway answers immediately and the
        # target is never invoked — "no read occurs", not "an error was logged".
        assert "transformedGatewayResponse" in result["mcp"]
        assert "transformedGatewayRequest" not in result["mcp"]

    def test_oversize_value_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A credential over the 4096-byte header limit short-circuits.

        The must-fail leg for the one limit AWS does not document a bound for. A
        truncated credential would reach the tool as an unusable one and surface as
        an opaque AccessDenied from DynamoDB; refusing keeps the failure mode the
        same as every other impossible state in this component.
        """
        huge = {**_creds(), "session_token": "x" * (MAX_HEADER_VALUE_BYTES + 1)}
        monkeypatch.setattr(
            interceptor_handler,
            "_vend_for_tool",
            lambda _tool, _scope, **_kwargs: huge,
        )

        result = interceptor_handler.handler(
            self._event("ReadDocument___read_document", {"doc_id": "PAY-001"}), None
        )

        assert "transformedGatewayResponse" in result["mcp"]
        assert "transformedGatewayRequest" not in result["mcp"]


# ---------------------------------------------------------------------------
# The builder in isolation
# ---------------------------------------------------------------------------


class TestBuildCredentialHeaders:
    """The header builder's own contract."""

    def test_rejects_oversize_value_naming_size_not_value(self) -> None:
        secret = "s" * (MAX_HEADER_VALUE_BYTES + 1)
        with pytest.raises(CredentialHeaderError) as exc:
            build_credential_headers(_SCOPE, {**_creds(), "secret_access_key": secret})
        message = str(exc.value)
        assert str(MAX_HEADER_VALUE_BYTES) in message
        # The message must name the SIZE, never the value.
        assert secret not in message

    @pytest.mark.parametrize(
        "field", ["access_key_id", "secret_access_key", "session_token"]
    )
    def test_rejects_incomplete_vend(self, field: str) -> None:
        """A missing credential field refuses rather than sending three of four."""
        creds = {**_creds()}
        del creds[field]
        with pytest.raises(CredentialHeaderError):
            build_credential_headers(_SCOPE, creds)

    @pytest.mark.parametrize("scope", ["", "   ", None, 7])
    def test_rejects_unusable_scope(self, scope: Any) -> None:
        with pytest.raises(CredentialHeaderError):
            build_credential_headers(scope, _creds())

    def test_size_report_has_no_values(self) -> None:
        """The measurement log line cannot leak a credential.

        Every reported entry is an int, so a caller that logs this dict logs only
        lengths — which is what makes it safe to emit on every request.
        """
        headers = build_credential_headers(_SCOPE, _creds())
        report = header_size_report(headers)

        assert all(isinstance(size, int) for size in report.values())
        assert report["total"] == sum(
            len(v.encode("utf-8")) for v in headers.values()
        )
        for value in headers.values():
            assert value not in report


# ---------------------------------------------------------------------------
# Tool side: reading them back
# ---------------------------------------------------------------------------


class TestToolReadsHeaders:
    """The tool reads the scope from the client context, never from the event."""

    def test_tool_reads_scope_from_headers(self) -> None:
        assert served_scope_from_context(credential_context(_SCOPE)) == _SCOPE

    def test_scope_is_stripped(self) -> None:
        ctx = lambda_context(
            {
                SERVED_SCOPE_HEADER: f"  {_SCOPE}  ",
                ACCESS_KEY_ID_HEADER: "a",
                SECRET_ACCESS_KEY_HEADER: "b",
                SESSION_TOKEN_HEADER: "c",
            }
        )
        assert served_scope_from_context(ctx) == _SCOPE

    def test_header_names_are_matched_case_insensitively(self) -> None:
        """Uppercased header names still resolve.

        The interceptor sends lowercase and the measured round trip preserved it,
        but HTTP/2 is free to normalise casing and a case-sensitive read here would
        fail closed on a valid request.
        """
        ctx = lambda_context(
            {
                SERVED_SCOPE_HEADER.upper(): _SCOPE,
                ACCESS_KEY_ID_HEADER.upper(): "a",
                SECRET_ACCESS_KEY_HEADER.upper(): "b",
                SESSION_TOKEN_HEADER.upper(): "c",
            }
        )
        assert served_scope_from_context(ctx) == _SCOPE

    def test_tool_returns_generic_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tool handler surfaces a generic error and leaks no scope or credential."""
        monkeypatch.setenv("DOCUMENTS_TABLE_NAME", "toxic-flow-documents")
        import read_document.handler as read_handler  # noqa: PLC0415

        result = read_handler.handler({"doc_id": "PAY-001"}, lambda_context(None))

        assert "error" in result
        assert result["error"] == "document identifier is invalid"
        # No credential material and no scope in the response.
        assert "scope" not in result
        assert PROPAGATED_HEADERS_KEY not in str(result)

    @pytest.mark.parametrize(
        "bad",
        [None, "nope", 7, [], {}, {SERVED_SCOPE_HEADER: _SCOPE}],
    )
    def test_tool_fails_closed_on_malformed_headers(self, bad: Any) -> None:
        with pytest.raises(ScopedCredentialsError):
            served_scope_from_context(lambda_context(bad))
