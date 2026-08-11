"""The additive ``context`` wire contract.

These cover the individual pieces the contract is made of:

  * ``interceptor.scoped_credentials.build_tenant_context`` assembles the nested
    ``context`` object from an already-vended credentials dict;
  * ``common.scoped_credentials.context_credentials_from_event`` reads the served
    scope + credentials back out of ``event["context"]`` and fails closed on a
    missing/malformed context (never falling back to the execution role);
  * the ``read_document`` handler reads scope + credentials from ``event["context"]``;
  * the REQUEST interceptor injects a SINGLE ``context`` object on every scoped
    path (``read_document``, ``search_documents``, ``reply``) and writes no flat
    credential fields and no top-level ``served_scope`` argument.

The universal round-trip of the whole contract is covered by the property test in
``tests/test_context_injection.py``; the interceptor cases here are example/unit
coverage of that contract.
"""

from __future__ import annotations

from typing import Any

import pytest

import interceptor.handler as interceptor_handler
from interceptor.scoped_credentials import build_tenant_context
from common.scoped_credentials import (
    ScopedCredentialsError,
    context_credentials_from_event,
)
from read_document.handler import handler as read_document_handler

# A valid 20-character STS access-key-id shape (ASIA prefix + 16), matching the
# canonical example from the AWS documentation so fixtures never contradict the
# wire contract.
_ACCESS_KEY_ID = "ASIAIOSFODNN7EXAMPLE"
_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEXAMPLETOKEN"


def _creds() -> dict[str, str]:
    """Return a three-field snake_case credentials dict (as vend returns)."""
    return {
        "access_key_id": _ACCESS_KEY_ID,
        "secret_access_key": _SECRET_ACCESS_KEY,
        "session_token": _SESSION_TOKEN,
    }


def _context(scope: str = "payments-core") -> dict[str, Any]:
    """Return a well-formed injected ``context`` object for ``scope``."""
    return {"served_scope": scope, "tenant_credentials": _creds()}


class TestBuildTenantContext:
    """The interceptor-side context assembler."""

    def test_shape_is_exactly_served_scope_and_tenant_credentials(self) -> None:
        ctx = build_tenant_context("payments-core", _creds())
        assert set(ctx) == {"served_scope", "tenant_credentials"}
        assert ctx["served_scope"] == "payments-core"
        assert set(ctx["tenant_credentials"]) == {
            "access_key_id",
            "secret_access_key",
            "session_token",
        }
        assert ctx["tenant_credentials"] == _creds()

    def test_copies_only_the_three_named_fields(self) -> None:
        # Extra keys on the input creds (e.g. a stray Expiration) never leak into
        # the assembled tenant_credentials.
        noisy = {**_creds(), "Expiration": "2025-01-01T00:00:00Z", "junk": 1}
        ctx = build_tenant_context("s", noisy)
        assert set(ctx["tenant_credentials"]) == {
            "access_key_id",
            "secret_access_key",
            "session_token",
        }


class TestContextCredentialsFromEvent:
    """The tool-side reader of ``event["context"]``."""

    def test_returns_table_and_scope(self, scoped_env, documents_table) -> None:
        table, scope = context_credentials_from_event({"context": _context()})
        assert scope == "payments-core"
        assert table.name == documents_table.name

    def test_missing_context_fails_closed(self) -> None:
        with pytest.raises(ScopedCredentialsError):
            context_credentials_from_event({})

    def test_non_object_context_fails_closed(self) -> None:
        with pytest.raises(ScopedCredentialsError):
            context_credentials_from_event({"context": "not-an-object"})

    def test_empty_served_scope_fails_closed(self) -> None:
        with pytest.raises(ScopedCredentialsError):
            context_credentials_from_event({"context": _context(scope="   ")})

    def test_incomplete_credentials_fails_closed(self) -> None:
        ctx = _context()
        del ctx["tenant_credentials"]["session_token"]
        with pytest.raises(ScopedCredentialsError):
            context_credentials_from_event({"context": ctx})


class TestReadDocumentReadsContext:
    """The read_document handler reads scope + credentials from event["context"]."""

    def test_reads_document_from_served_partition(
        self, scoped_env, put_documents
    ) -> None:
        from tests.conftest import make_document

        put_documents([make_document("payments-core", "PAY-001", body="ledger")])
        event = {"doc_id": "PAY-001", "context": _context("payments-core")}

        result = read_document_handler(event, None)

        assert result == {"body": "ledger", "scope": "payments-core"}

    def test_missing_context_returns_generic_error(self, scoped_env) -> None:
        # No context injected -> fail closed with a generic, detail-free error.
        result = read_document_handler({"doc_id": "PAY-001"}, None)
        assert result == {"error": "document identifier is invalid"}
        # No credential material or scope leaks into the error response.
        assert "context" not in result
        assert "scope" not in result


class TestInterceptorInjectsSingleContext:
    """The interceptor writes ONE `context` key for every scoped tool.

    A single nested ``context`` object is written at ``arguments["context"]`` on
    all three scoped paths; no flat credential fields and no top-level
    ``served_scope`` argument are written; model-supplied arguments are untouched.
    """

    @pytest.fixture(autouse=True)
    def _stub_vend_and_scope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Avoid real STS / JWT: derive a fixed scope and vend fixed credentials.
        monkeypatch.setattr(
            interceptor_handler,
            "served_scope_from_authorization",
            lambda _auth: "payments-core",
        )
        monkeypatch.setattr(
            interceptor_handler,
            "_vend_for_tool",
            lambda _tool, _scope: _creds(),
        )

    @staticmethod
    def _tools_call_event(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "mcp": {
                "gatewayRequest": {
                    "headers": {"Authorization": "Bearer token"},
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
    def _assert_no_flat_or_scope_arg(
        args: dict[str, Any], model_keys: set[str]
    ) -> None:
        # Closed key-set check: the forwarded arguments are exactly the
        # model-supplied keys plus the single injected `context`. Any flat
        # credential field would surface as an extra key, so this exact-set
        # assertion proves their absence without naming any retired flat field.
        assert set(args.keys()) == model_keys | {"context"}
        # No top-level served_scope argument is written on any path.
        assert "served_scope" not in args

    def test_read_document_gets_single_context(self) -> None:
        event = self._tools_call_event(
            "ReadDocument___read_document", {"document_id": "PAY-001"}
        )
        result = interceptor_handler.handler(event, None)
        args = result["mcp"]["transformedGatewayRequest"]["body"]["params"]["arguments"]

        # Exactly one new key — the nested context object with the exact shape.
        assert args["context"] == build_tenant_context("payments-core", _creds())
        self._assert_no_flat_or_scope_arg(args, {"document_id"})
        # The model-supplied argument is left intact.
        assert args["document_id"] == "PAY-001"

    def test_search_documents_gets_single_context(self) -> None:
        event = self._tools_call_event(
            "SearchDocuments___search_documents", {"query": "refund"}
        )
        result = interceptor_handler.handler(event, None)
        args = result["mcp"]["transformedGatewayRequest"]["body"]["params"]["arguments"]

        assert args["context"] == build_tenant_context("payments-core", _creds())
        self._assert_no_flat_or_scope_arg(args, {"query"})
        # The model-supplied argument is left intact.
        assert args["query"] == "refund"

    def test_reply_gets_single_context(self) -> None:
        event = self._tools_call_event(
            "Reply___reply", {"document_id": "PAY-001", "body": "hello"}
        )
        result = interceptor_handler.handler(event, None)
        args = result["mcp"]["transformedGatewayRequest"]["body"]["params"]["arguments"]

        assert args["context"] == build_tenant_context("payments-core", _creds())
        self._assert_no_flat_or_scope_arg(args, {"document_id", "body"})
        # The model-supplied arguments are left intact.
        assert args["document_id"] == "PAY-001"
        assert args["body"] == "hello"

    def test_model_supplied_context_is_overwritten(self) -> None:
        # A value already present at arguments["context"] is
        # overwritten wholesale by the vended context (never merged/read).
        event = self._tools_call_event(
            "ReadDocument___read_document",
            {"document_id": "PAY-001", "context": {"evil": "value"}},
        )
        result = interceptor_handler.handler(event, None)
        args = result["mcp"]["transformedGatewayRequest"]["body"]["params"]["arguments"]

        assert args["context"] == build_tenant_context("payments-core", _creds())
        assert "evil" not in args["context"]


class TestInterceptorAuthorizationHeaderCaseInsensitive:
    """Regression: the Authorization header is read CASE-INSENSITIVELY.

    Header field names are case-insensitive (RFC 9110 §5.1), and HTTP/2 lowercases
    them on the wire (RFC 9113 §8.2). The AgentCore Gateway negotiates HTTP/2, so
    the bearer token arrives under the lowercase key ``authorization``. A prior
    case-sensitive ``headers.get("Authorization")`` missed it, so
    ``served_scope_from_authorization(None)`` returned None and the interceptor
    failed CLOSED ("No verifiable served_scope") on an otherwise-valid token —
    which rejected every request that arrived over HTTP/2.

    Unlike ``TestInterceptorInjectsSingleContext`` (whose stub ignores its
    argument), this class stubs ``served_scope_from_authorization`` FAITHFULLY to
    the real fail-closed contract: it yields a scope only when it actually
    receives a Bearer value, and returns None for the None a MISSED header lookup
    produces. That makes the handler's header-lookup casing observable — a
    case-sensitive lookup fails these tests, the case-insensitive fix passes them.
    """

    @pytest.fixture(autouse=True)
    def _stub_vend_and_scope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Vend fixed credentials without real STS.
        monkeypatch.setattr(
            interceptor_handler,
            "_vend_for_tool",
            lambda _tool, _scope: _creds(),
        )

        # Faithful to jwt_claims.served_scope_from_authorization: a resolvable
        # "Bearer <jwt>" value yields a scope; None/empty (what a missed header
        # lookup produces) fails closed. This is what makes case-sensitivity of
        # the handler's header lookup observable in these tests.
        def _scope_if_bearer(auth: Any) -> str | None:
            if isinstance(auth, str) and auth.strip().lower().startswith("bearer "):
                return "payments-core"
            return None

        monkeypatch.setattr(
            interceptor_handler,
            "served_scope_from_authorization",
            _scope_if_bearer,
        )

    @staticmethod
    def _event_with_header_key(header_key: str) -> dict[str, Any]:
        """A scoped ``read_document`` tools/call carrying the bearer token under
        ``header_key`` (used to vary the Authorization header's letter casing)."""
        return {
            "mcp": {
                "gatewayRequest": {
                    "headers": {header_key: "Bearer token"},
                    "body": {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "ReadDocument___read_document",
                            "arguments": {"doc_id": "PAY-001"},
                        },
                    },
                }
            }
        }

    def _assert_context_injected(self, result: dict[str, Any]) -> None:
        # ALLOW path (transformedGatewayRequest), NOT the fail-closed
        # short-circuit (transformedGatewayResponse): the handler resolved the
        # token, derived the scope, and injected the single `context`.
        mcp = result["mcp"]
        assert "transformedGatewayResponse" not in mcp
        assert "transformedGatewayRequest" in mcp
        args = mcp["transformedGatewayRequest"]["body"]["params"]["arguments"]
        assert args["context"] == build_tenant_context("payments-core", _creds())
        # Model-supplied argument is left intact; exactly one new key is added.
        assert args["doc_id"] == "PAY-001"
        assert set(args.keys()) == {"doc_id", "context"}

    def test_lowercase_authorization_header_injects_context(self) -> None:
        # HTTP/2 lowercases header field names -> the header arrives as
        # `authorization`. The case-insensitive lookup must resolve it (this is
        # the exact case that used to reject every request).
        result = interceptor_handler.handler(
            self._event_with_header_key("authorization"), None
        )
        self._assert_context_injected(result)

    def test_mixed_case_authorization_header_injects_context(self) -> None:
        # A gateway may normalize casing differently; any casing must resolve.
        result = interceptor_handler.handler(
            self._event_with_header_key("AuThOrIzAtIoN"), None
        )
        self._assert_context_injected(result)

    def test_capital_authorization_header_still_injects_context(self) -> None:
        # HTTP/1.1 canonical casing keeps working after the fix (no regression).
        result = interceptor_handler.handler(
            self._event_with_header_key("Authorization"), None
        )
        self._assert_context_injected(result)

    def test_absent_authorization_header_still_fails_closed(self) -> None:
        # No Authorization header under ANY casing -> lookup returns None ->
        # fail closed. Proves the case-insensitive lookup did not start
        # over-matching unrelated headers into a silent inject.
        event = self._event_with_header_key("X-Not-Authorization")
        result = interceptor_handler.handler(event, None)
        mcp = result["mcp"]
        assert "transformedGatewayRequest" not in mcp
        assert "transformedGatewayResponse" in mcp
