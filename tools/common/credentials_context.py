"""Tool-side parsing and fail-closed validation of the propagated credential HEADERS.

Wire contract (header channel)
------------------------------
The REQUEST interceptor propagates the authoritative scope and the vended STS
credentials as four custom request HEADERS. The Gateway delivers them to a Lambda
target inside the Lambda CLIENT CONTEXT, not in the event::

    context.client_context.custom["bedrockAgentCorePropagatedHeaders"] = {
        "x-tvm-served-scope":       "<scope>",
        "x-tvm-access-key-id":      "...",
        "x-tvm-secret-access-key":  "...",
        "x-tvm-session-token":      "...",
    }

The Lambda ``event`` therefore carries ONLY the model-supplied tool arguments
(``doc_id`` / ``query`` / ``body``) — the tool's declared ``inputSchema`` is the
whole of its contract.

Why these readers do not take the event
---------------------------------------
The functions here do not accept an ``event`` parameter at all. A reader that can
find a credential in the body is a path, even when nothing is supposed to use it,
and a tool that accepts a body-supplied credential accepts one the MODEL could
have written. Leaving the parameter out makes "the credential cannot come from
the body" a property of the signature instead of a rule someone has to keep.

Why the lookup is case-insensitive
----------------------------------
HTTP field names are case-insensitive (RFC 9110 §5.1) and HTTP/2 mandates
lowercase on the wire (RFC 9113 §8.2). The Gateway negotiates HTTP/2, so a header
sent as ``X-Tvm-Session-Token`` can arrive lowercased. The interceptor sends
lowercase names and the measured round trip preserved them exactly, but a
case-sensitive read here would fail closed on a valid request if that ever
changed — the same trap already documented for ``Authorization`` and
``Mcp-Session-Id`` on the interceptor side.

Fail-closed contract: a missing client context, a missing propagated-headers map,
or any missing/empty header raises :class:`ScopedCredentialsError`. Callers
surface a generic, detail-free error and NEVER fall back to the tool's execution
role (which holds no DynamoDB grant) or the default credential chain.

Security:
    This module NEVER logs the scope, a header name/value, or a credential, and
    its error message names none of them.

AWS documentation references (verified via the AWS Documentation MCP server, per
the workspace ``aws-docs-lookup`` rule):
    - Header propagation with Gateway (allowlist, interceptor-over-client
      precedence, 4096-byte per-value and 10-header-per-target limits):
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-headers.html
    - Lambda target context object. NOTE: this page documents the six
      ``bedrockAgentCore*`` fields and does NOT mention
      ``bedrockAgentCorePropagatedHeaders``; that key is UNDOCUMENTED in the
      service reference and was established by live measurement against a
      deployed gateway. Treat it as observed behaviour that could change, and
      re-verify it before relying on it:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-lambda.html
    - STS ``Credentials`` shape (AccessKeyId, SecretAccessKey, SessionToken):
      https://docs.aws.amazon.com/STS/latest/APIReference/API_Credentials.html
    - boto3 credential keyword names the three fields map onto:
      https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/credentials.html

Constants:
    PROPAGATED_HEADERS_KEY: the ``client_context.custom`` key the Gateway uses.
    SERVED_SCOPE_HEADER / HEADER_TO_SESSION_KWARG: the header names, and the
        mapping of the three credential headers onto the boto3 ``Session``
        keyword arguments.

Functions:
    propagated_headers: Return the propagated-header map, validated.
    served_scope_from_context: Return the authoritative served scope.
    session_kwargs_from_context: Return the boto3 credential keyword arguments.
"""

from __future__ import annotations

from typing import Any

#: The ``client_context.custom`` key under which the Gateway delivers the
#: allowlisted request headers to a Lambda target. Established by measurement, not
#: by the service reference — see the module docstring.
PROPAGATED_HEADERS_KEY = "bedrockAgentCorePropagatedHeaders"

#: Header carrying the authoritative, JWT-derived scope. MUST stay in sync with
#: ``interceptor/credential_headers.py``.
SERVED_SCOPE_HEADER = "x-tvm-served-scope"

#: Mapping of the three credential headers onto the boto3 ``Session`` keyword
#: arguments they populate. This mapping's KEYS are also the completeness set for
#: the fail-closed validation: all three must be present as non-empty strings.
HEADER_TO_SESSION_KWARG: dict[str, str] = {
    "x-tvm-access-key-id": "aws_access_key_id",
    "x-tvm-secret-access-key": "aws_secret_access_key",
    "x-tvm-session-token": "aws_session_token",
}

#: Single generic message used for EVERY failure branch, so the failure mode
#: cannot be distinguished by an attacker probing which part was malformed.
_GENERIC = "propagated credential context is missing or malformed"


class ScopedCredentialsError(RuntimeError):
    """Raised when the propagated credential headers are missing or malformed.

    Signals that the tool has no scoped credentials (or no served scope) to use.
    The tool handler surfaces a generic error and NEVER falls back to its own
    execution role (which holds no DynamoDB permission) or the default credential
    chain.
    """


def _custom_map(lambda_context: Any) -> dict[str, Any]:
    """Return ``client_context.custom`` as a dict, or raise.

    Args:
        lambda_context: The Lambda context object passed to the handler.

    Returns:
        The ``custom`` dict from the Lambda client context.

    Raises:
        ScopedCredentialsError: If there is no client context, or its ``custom``
            attribute is absent or not a dict. Attribute access is guarded with
            ``getattr`` because the client context is absent entirely on a direct
            (non-Gateway) invocation, and a raw ``AttributeError`` would escape
            the handler as a 5xx instead of the generic tool error.
    """
    client_context = getattr(lambda_context, "client_context", None)
    if client_context is None:
        raise ScopedCredentialsError(_GENERIC)
    custom = getattr(client_context, "custom", None)
    if not isinstance(custom, dict):
        raise ScopedCredentialsError(_GENERIC)
    return custom


def propagated_headers(lambda_context: Any) -> dict[str, str]:
    """Return the propagated request headers, lowercased and validated.

    Enforces the full fail-closed contract: the client context must exist, carry a
    ``bedrockAgentCorePropagatedHeaders`` object, and that object must hold
    non-empty string values for the served-scope header AND all three credential
    headers. Header names are lowercased so the caller's lookups are
    case-insensitive with respect to what the wire delivered.

    Args:
        lambda_context: The Lambda context object passed to the handler.

    Returns:
        The propagated headers with lowercased names. Guaranteed to contain the
        served-scope header and the three credential headers as non-empty strings.

    Raises:
        ScopedCredentialsError: On any missing or malformed part. The message
            names no header, scope, or value, and is identical for every branch.
    """
    raw = _custom_map(lambda_context).get(PROPAGATED_HEADERS_KEY)
    if not isinstance(raw, dict):
        raise ScopedCredentialsError(_GENERIC)

    headers = {
        str(name).lower(): value
        for name, value in raw.items()
        if isinstance(value, str)
    }

    for required in (SERVED_SCOPE_HEADER, *HEADER_TO_SESSION_KWARG):
        value = headers.get(required)
        if not (isinstance(value, str) and value.strip()):
            raise ScopedCredentialsError(_GENERIC)

    return headers


def served_scope_from_context(lambda_context: Any) -> str:
    """Return the authoritative served scope from the propagated headers.

    The value comes from the JWT the interceptor verified, carried on a channel
    the model never writes to, so it is authoritative in a way a body field could
    not be.

    Args:
        lambda_context: The Lambda context object passed to the handler.

    Returns:
        The served scope, stripped of surrounding whitespace.

    Raises:
        ScopedCredentialsError: If the propagated headers are missing or
            malformed. The caller surfaces a generic error and NEVER falls back to
            its execution role or the default chain.
    """
    return propagated_headers(lambda_context)[SERVED_SCOPE_HEADER].strip()


def session_kwargs_from_context(lambda_context: Any) -> dict[str, str]:
    """Return the boto3 credential keyword arguments from the propagated headers.

    Maps the three credential headers onto ``aws_access_key_id`` /
    ``aws_secret_access_key`` / ``aws_session_token`` by name.

    Args:
        lambda_context: The Lambda context object passed to the handler.

    Returns:
        A dict of boto3 ``Session``/``resource`` credential keyword arguments.
        Always complete — all three keys are present with non-empty values, which
        is what keeps the caller from ever constructing a client that would fall
        through to the default credential chain.

    Raises:
        ScopedCredentialsError: If the propagated headers are missing or
            malformed.
    """
    headers = propagated_headers(lambda_context)
    return {
        session_kwarg: headers[header]
        for header, session_kwarg in HEADER_TO_SESSION_KWARG.items()
    }
