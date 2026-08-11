"""Transport-failure tests for the three tool handlers.

Each handler documents one error contract for its DynamoDB call: any failure
returns a GENERIC ``{"error": ...}`` dict, names no scope and no credential
field, and never falls back to the Lambda execution role. Every handler enforced
it by catching ``ClientError``.

That is only half of botocore's hierarchy. ``ClientError`` and ``BotoCoreError``
are SIBLINGS -- ``issubclass(ClientError, BotoCoreError)`` is False -- and the
transport-level failures (``EndpointConnectionError``, ``ReadTimeoutError``,
``ConnectTimeoutError``) descend from ``BotoCoreError`` alone. So a DynamoDB call
that failed below the API layer escaped the handler entirely: instead of the
generic error dict, the raw exception reached the Lambda runtime and the caller
got a traceback.

Scope confinement never depended on this -- a read that never completes returns
no data, and a write that never completes performs none -- so the gap is a broken
error contract rather than an isolation failure. It is still worth a test: the
contract is what stops a transient DynamoDB blip from looking, to the model,
like something other than a generic failure.

The suite had no coverage here. ``tests/test_fail_closed.py`` raises
``BotoCoreError`` only on the INTERCEPTOR's ``vend_scoped_credentials`` path, and
covers the handlers only for the malformed-context (``ScopedCredentialsError``)
case -- nothing raised a non-``ClientError`` botocore error at ``table.query`` /
``get_item`` / ``update_item``. Every test below fails against the old
two-element except tuple.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

import read_document.handler as read_document_module
import reply.handler as reply_module
import search_documents.handler as search_module
from read_document.handler import handler as read_document_handler
from reply.handler import handler as reply_handler
from search_documents.handler import handler as search_documents_handler
from tests.conftest import SERVED_SCOPE

_ENDPOINT = "https://dynamodb.us-east-1.amazonaws.com"

# Matches the credential shapes used by the other tool tests so no fixture
# contradicts the wire contract. These values never reach AWS: the table is
# replaced before any call is made.
_ACCESS_KEY_ID = "ASIAIOSFODNN7EXAMPLE"
_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEXAMPLETOKEN"

# The three failures a real deployment actually meets: a resolvable-but-
# unreachable endpoint, a connection that never establishes, and a response that
# never arrives. All are BotoCoreError, none is a ClientError.
_TRANSPORT_ERRORS = [
    pytest.param(
        EndpointConnectionError(endpoint_url=_ENDPOINT), id="EndpointConnectionError"
    ),
    pytest.param(ReadTimeoutError(endpoint_url=_ENDPOINT), id="ReadTimeoutError"),
    pytest.param(ConnectTimeoutError(endpoint_url=_ENDPOINT), id="ConnectTimeoutError"),
]


def _context(scope: str = SERVED_SCOPE) -> dict[str, Any]:
    """Return a well-formed injected ``context`` object for ``scope``."""
    return {
        "served_scope": scope,
        "tenant_credentials": {
            "access_key_id": _ACCESS_KEY_ID,
            "secret_access_key": _SECRET_ACCESS_KEY,
            "session_token": _SESSION_TOKEN,
        },
    }


class _FailingTable:
    """A table stub whose every DynamoDB operation raises ``error``.

    Replaces the scoped table the handler would build from the injected
    credentials, so the chosen botocore error is raised at exactly the call site
    the handler guards -- without needing a broken network.
    """

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.calls = 0

    def query(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise self._error

    def get_item(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise self._error

    def update_item(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise self._error


def _install_failing_table(
    monkeypatch: pytest.MonkeyPatch, module: Any, error: BaseException
) -> _FailingTable:
    """Point ``module``'s table builder at a stub that raises ``error``."""
    table = _FailingTable(error)
    monkeypatch.setattr(module, "documents_table_from_event", lambda _event: table)
    return table


# ---------------------------------------------------------------------------
# The hierarchy fact the fix rests on
# ---------------------------------------------------------------------------


class TestBotocoreHierarchy:
    """Pin the class relationship that makes the second except branch necessary."""

    def test_client_error_does_not_cover_botocore_error(self) -> None:
        # If this ever became True, catching ClientError alone would suffice and
        # the handlers' wider tuples could be simplified. It is False, and the
        # tests below exist because of it.
        assert not issubclass(ClientError, BotoCoreError)
        assert not issubclass(BotoCoreError, ClientError)

    @pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
    def test_transport_errors_are_botocore_errors_only(
        self, error: BaseException
    ) -> None:
        # Each error used below really is on the branch of the hierarchy the old
        # except tuple missed -- otherwise these tests would pass vacuously.
        assert isinstance(error, BotoCoreError)
        assert not isinstance(error, ClientError)


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------


class TestSearchDocumentsTransportFailure:
    """A ``Query`` that fails below the API layer returns the generic error."""

    @pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
    def test_returns_the_generic_error(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        table = _install_failing_table(monkeypatch, search_module, error)

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert table.calls == 1
        assert result == {"error": search_module._GENERIC_ERROR}
        # No partial result set leaks alongside the error: a caller that reads
        # "results" on an error response must find nothing to read.
        assert "results" not in result

    def test_the_error_echoes_no_scope_or_credential_value(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The generic-error contract is about what the message does NOT contain.
        # A transport failure carries an endpoint URL and the handler has the
        # served scope and credentials in hand; none may reach the response.
        _install_failing_table(
            monkeypatch, search_module, EndpointConnectionError(endpoint_url=_ENDPOINT)
        )

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        rendered = repr(result)
        for secret in (
            SERVED_SCOPE,
            _ACCESS_KEY_ID,
            _SECRET_ACCESS_KEY,
            _SESSION_TOKEN,
            _ENDPOINT,
        ):
            assert secret not in rendered


# ---------------------------------------------------------------------------
# read_document
# ---------------------------------------------------------------------------


class TestReadDocumentTransportFailure:
    """A ``GetItem`` that fails below the API layer returns the generic error."""

    @pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
    def test_returns_the_generic_not_found(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        table = _install_failing_table(monkeypatch, read_document_module, error)

        result = read_document_handler(
            {"doc_id": "DOC-0001", "context": _context()}, None
        )

        assert table.calls == 1
        assert result == {"error": "document not found"}
        # Crucially no body: a failed read must be indistinguishable from an
        # absent document, and must certainly not return content.
        assert "body" not in result
        assert "scope" not in result


# ---------------------------------------------------------------------------
# reply
# ---------------------------------------------------------------------------


class TestReplyTransportFailure:
    """An ``UpdateItem`` that fails below the API layer returns the generic error."""

    @pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
    def test_returns_the_generic_error(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        table = _install_failing_table(monkeypatch, reply_module, error)

        result = reply_handler(
            {"doc_id": "DOC-0001", "body": "Refund processed.", "context": _context()},
            None,
        )

        assert table.calls == 1
        assert result == {"error": reply_module._GENERIC_ERROR}
        # No success signal on a write that never landed.
        assert "success" not in result

    def test_transport_failure_is_not_reported_as_a_refused_append(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # reply distinguishes one ClientError -- the rejected ConditionExpression,
        # meaning the target document does not exist or its conversation is full --
        # from every other failure, because that one tells the model to stop
        # retrying. A BotoCoreError carries no ``response`` payload, so a branch
        # that reached for exc.response would raise AttributeError; and
        # mislabelling a transient failure as a permanent refusal would stop a
        # legitimate retry forever.
        _install_failing_table(
            monkeypatch, reply_module, ReadTimeoutError(endpoint_url=_ENDPOINT)
        )

        result = reply_handler(
            {"doc_id": "DOC-0001", "body": "Refund processed.", "context": _context()},
            None,
        )

        assert result == {"error": reply_module._GENERIC_ERROR}
        assert result["error"] != reply_module._APPEND_REFUSED_ERROR
