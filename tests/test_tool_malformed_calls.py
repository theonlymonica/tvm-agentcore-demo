"""Per-tool malformed-call unit tests.

One test each for ``read_document``, ``search_documents`` and ``reply`` invoked
with a malformed call. Each asserts the tool fails closed with a GENERIC error
object that leaks NO credential material and NO argument echo: the
response carries none of the forbidden credential/argument key names and none of
the supplied argument or credential VALUES, checked recursively over the
response's keys and its string values.

A "malformed call" here injects a ``context`` whose ``served_scope`` is blank so
the fail-closed path fires (``served_scope_from_event`` raises
``ScopedCredentialsError``), while still carrying a complete ``tenant_credentials``
block with distinctive credential values. That lets each single test exercise
BOTH exposure surfaces at once:

  * argument echo — the model-supplied ``doc_id`` / ``query`` / ``body`` values; and
  * credential material — the ``tenant_credentials`` key names and their values,

so an accidental echo of either would fail the recursive assertion. The import
style mirrors the other tool tests (``tools/`` is on ``sys.path`` via the root
conftest; each tool imports ``common.*``). The
``scoped_env`` fixture sets ``DOCUMENTS_TABLE_NAME`` as the deployed Lambdas would
see it, though on this fail-closed path the tool returns before building a client.
"""

from __future__ import annotations

from typing import Any

from read_document.handler import handler as read_document_handler
from search_documents.handler import handler as search_documents_handler
from reply.handler import handler as reply_handler

# Distinctive credential values (a valid 20-char ASIA access-key-id shape, per the
# canonical example in the AWS documentation) supplied inside the malformed context
# so the test can prove none of them leak into the response.
_ACCESS_KEY_ID = "ASIAIOSFODNN7EXAMPLE"
_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEXAMPLESESSIONTOKEN"

# Forbidden KEY names that must never appear as a key anywhere in a tool
# response: the injected-context/credential keys and the model-supplied argument
# keys. Checked against the response's keys only — NOT against string values,
# since a generic message such as "query is invalid" legitimately contains the
# word "query" as prose while carrying no "query" key or query value.
_FORBIDDEN_KEYS = frozenset(
    {
        "doc_id",
        "query",
        "body",
        "context",
        "tenant_credentials",
        "access_key_id",
        "secret_access_key",
        "session_token",
    }
)


def _malformed_context() -> dict[str, Any]:
    """Return a malformed injected ``context``: blank scope, full credentials.

    The blank ``served_scope`` trips the fail-closed validation, while the
    complete ``tenant_credentials`` block gives the test concrete credential values
    to assert are absent from the response.

    Returns:
        A ``context`` dict that fails validation on ``served_scope`` yet carries a
        full three-field ``tenant_credentials`` object.
    """
    return {
        "served_scope": "   ",
        "tenant_credentials": {
            "access_key_id": _ACCESS_KEY_ID,
            "secret_access_key": _SECRET_ACCESS_KEY,
            "session_token": _SESSION_TOKEN,
        },
    }


def _collect_keys_and_values(obj: Any) -> tuple[set[str], list[str]]:
    """Recursively collect every dict key and every string value in ``obj``.

    Args:
        obj: An arbitrary JSON-like structure (dict / list / tuple / scalar).

    Returns:
        A ``(keys, string_values)`` pair: the set of all dict key names reachable
        at any depth, and the list of all string values reachable at any depth
        (dict values, list elements, and nested combinations thereof).
    """
    keys: set[str] = set()
    values: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if isinstance(key, str):
                    keys.add(key)
                _walk(val)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            values.append(node)

    _walk(obj)
    return keys, values


def _assert_generic_error_no_leak(
    response: Any,
    *,
    expected_message: str,
    forbidden_values: list[str],
) -> None:
    """Assert ``response`` is a generic error leaking no forbidden key or value.

    Args:
        response: The tool handler return value.
        expected_message: The exact generic, detail-free error message the tool
            returns on the fail-closed path.
        forbidden_values: Supplied argument and credential values that must not
            appear in any response string value at any depth.
    """
    # It is a generic error object: exactly one `error` key carrying the generic,
    # detail-free message — no scope, no credential field, no argument echo.
    assert isinstance(response, dict)
    assert set(response.keys()) == {"error"}
    assert response["error"] == expected_message

    keys, values = _collect_keys_and_values(response)

    # No forbidden credential/argument KEY name appears anywhere in the response.
    leaked_keys = keys & _FORBIDDEN_KEYS
    assert not leaked_keys, f"response leaked forbidden key(s): {sorted(leaked_keys)}"

    # No supplied argument or credential VALUE appears in any response string, at
    # any depth. Substring containment (not equality) so a partial/embedded echo
    # is caught too.
    for forbidden in forbidden_values:
        for value in values:
            assert forbidden not in value, (
                f"response echoed forbidden value {forbidden!r} in {value!r}"
            )


def test_read_document_malformed_call_returns_generic_error(scoped_env) -> None:
    """read_document fails closed on a malformed context with no echo."""
    doc_id = "PAY-001"
    event = {"doc_id": doc_id, "context": _malformed_context()}

    response = read_document_handler(event, None)

    _assert_generic_error_no_leak(
        response,
        expected_message="document identifier is invalid",
        forbidden_values=[
            doc_id,
            _ACCESS_KEY_ID,
            _SECRET_ACCESS_KEY,
            _SESSION_TOKEN,
        ],
    )


def test_search_documents_malformed_call_returns_generic_error(scoped_env) -> None:
    """search_documents fails closed on a malformed context with no echo."""
    query = "refund"
    event = {"query": query, "context": _malformed_context()}

    response = search_documents_handler(event, None)

    _assert_generic_error_no_leak(
        response,
        expected_message="query is invalid",
        forbidden_values=[
            query,
            _ACCESS_KEY_ID,
            _SECRET_ACCESS_KEY,
            _SESSION_TOKEN,
        ],
    )


def test_reply_malformed_call_returns_generic_error(scoped_env) -> None:
    """reply fails closed on a malformed context with no echo."""
    doc_id = "PAY-001"
    body = "please append this conversation entry"
    event = {"doc_id": doc_id, "body": body, "context": _malformed_context()}

    response = reply_handler(event, None)

    _assert_generic_error_no_leak(
        response,
        expected_message="input is invalid",
        forbidden_values=[
            doc_id,
            body,
            _ACCESS_KEY_ID,
            _SECRET_ACCESS_KEY,
            _SESSION_TOKEN,
        ],
    )
