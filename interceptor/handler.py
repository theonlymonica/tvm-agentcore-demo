"""
Scope-injecting REQUEST interceptor for the AgentCore Gateway.

Derives the authoritative ``served_scope`` from the request-time identity (a
signature-validated JWT claim) and injects it into the tool
arguments for the scoped tool set. It makes NO allow/block decision, reads NO
SSM, and tracks NO session state — enforcement is unconditional and structural.

Behavior:
- Non-``tools/call`` protocol messages (initialize, tools/list,
  notifications/initialized, ping, ...): pass through UNCHANGED.
- ``tools/call`` for a tool OUTSIDE the scoped set: pass through UNCHANGED.
- ``tools/call`` for a scoped tool (``read_document``, ``search_documents``,
  ``reply``): derive ``served_scope`` from the validated Authorization JWT, vend
  scoped credentials, and write a SINGLE ``context`` object at
  ``params.arguments["context"]`` (carrying the authoritative ``served_scope``
  and the ``tenant_credentials``) — any value already present at that key is
  OVERWRITTEN without being read — then return the modified request in
  ``transformedGatewayRequest.body``. No flat credential fields and no top-level
  ``served_scope`` argument are written on any path.
- FAIL CLOSED: when no verifiable ``served_scope`` can be derived, return a
  ``transformedGatewayResponse`` whose JSON-RPC result has ``isError`` true, a
  GENERIC message, and NO scope detail — so no document read occurs.

Security:
    The Authorization header and the JWT are NEVER logged. The
    ``Mcp-Session-Id`` request-header handling is preserved via
    ``interceptor.tool_classifier.extract_session_id``.

Environment variables:
    None required. This interceptor holds no data-plane permissions
    and reads no data-plane state. (An optional ``KNOWN_SCOPE_GROUPS`` override
    is consumed by ``interceptor.jwt_claims``.)

References (verified against the AWS documentation):
    - REQUEST interceptor input/output contract (``interceptorInputVersion`` /
      ``interceptorOutputVersion`` "1.0"; ``mcp.gatewayRequest.headers`` incl.
      ``Authorization`` / ``Mcp-Session-Id`` present only when
      ``passRequestHeaders=true``; ``mcp.gatewayRequest.body``;
      ``mcp.transformedGatewayRequest.body``; ``mcp.transformedGatewayResponse``
      short-circuits the gateway immediately):
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-types.html
    - Gateway tool naming (triple-underscore delimiter used by classify_tool):
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from interceptor.jwt_claims import served_scope_from_authorization
from interceptor.scoped_credentials import (
    READ_ACTIONS,
    WRITE_ACTIONS,
    build_tenant_context,
    vend_scoped_credentials,
)
from interceptor.tool_classifier import (
    classify_tool,
    extract_session_id,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOOLS_CALL_METHOD = "tools/call"

#: Tools for which the ``context`` object is injected authoritatively. ``reply``
#: is included: under the composite-key table it needs the served scope to build
#: its key and write via scoped credentials. All three are the
#: "scoped tool set".
_SCOPED_TOOLS = frozenset({"read_document", "search_documents", "reply"})

#: Read-path tools -> assume DocumentsAccessRole (GetItem/Query).
_READ_TOOLS = frozenset({"read_document", "search_documents"})

#: Write-path tools -> assume DocumentsWriteRole (UpdateItem).
_WRITE_TOOLS = frozenset({"reply"})

#: Env vars naming the scoped roles + table ARN the interceptor assumes/scopes
#: to. Wired in cdk/scoped_credentials_stack.py.
_ENV_ACCESS_ROLE_ARN = "DOCUMENTS_ACCESS_ROLE_ARN"
_ENV_WRITE_ROLE_ARN = "DOCUMENTS_WRITE_ROLE_ARN"
_ENV_TABLE_ARN = "DOCUMENTS_TABLE_ARN"

#: Generic fail-closed message — no scope detail is ever leaked.
_GENERIC_ERROR_MESSAGE = "request could not be processed"


# ---------------------------------------------------------------------------
# Request-header access (case-insensitive)
# ---------------------------------------------------------------------------


def _get_header(headers: dict[str, Any] | None, name: str) -> Optional[str]:
    """Return a request-header value via a CASE-INSENSITIVE key lookup.

    HTTP header field names are case-insensitive (RFC 9110 §5.1), and HTTP/2
    mandates lowercase field names on the wire (RFC 9113 §8.2). The AgentCore
    Gateway negotiates HTTP/2, so ``Authorization`` can arrive as the lowercase
    ``authorization``; a case-sensitive ``headers.get("Authorization")`` would
    then miss it and the interceptor would fail closed on an otherwise-valid
    token. This mirrors the case-insensitive ``Mcp-Session-Id`` lookup in
    ``interceptor.tool_classifier.extract_session_id`` (same rationale: header
    names are case-insensitive and gateways may normalize casing differently).

    SECURITY: like ``extract_session_id``, this never logs the headers dict —
    it carries the JWT / bearer token.

    Args:
        headers: The ``gatewayRequest.headers`` dict. May be None/empty.
        name: The header name to match, compared case-insensitively.

    Returns:
        The first matching header value, or None if the header is absent.
    """
    if not headers or not isinstance(headers, dict):
        return None
    target_lower = name.lower()
    for key, value in headers.items():
        if key.lower() == target_lower:
            return value
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """REQUEST interceptor Lambda entry point.

    Parses the MCP REQUEST interceptor payload, passes non-scoped traffic
    through unchanged, and for scoped ``tools/call`` messages injects the
    authoritative JWT-derived ``served_scope`` (or fails closed).

    Args:
        event: The REQUEST interceptor payload from the gateway (see the
            module references for the contract).
        context: The Lambda context (unused).

    Returns:
        An ``interceptorOutputVersion: "1.0"`` envelope carrying either
        ``mcp.transformedGatewayRequest.body`` (allow / pass-through) or
        ``mcp.transformedGatewayResponse`` (fail-closed short-circuit).
    """
    mcp = event.get("mcp", {}) or {}
    gateway_request = mcp.get("gatewayRequest", {}) or {}
    headers = gateway_request.get("headers", {}) or {}
    # --- Event immutability ---------------------------------------------------
    # Reach the request body through the defensive `.get()` chain (never a direct
    # subscript: a KeyError would surface as a Lambda exception -> Gateway 5xx),
    # then `copy.deepcopy` it. EVERY subsequent write into `params` / `arguments`
    # / `context` lands on this copy only, so the object graph reachable from
    # event["mcp"]["gatewayRequest"]["body"] stays deep-equal to its pre-call
    # state. Pass-through paths forward this copy, which is likewise deep-equal
    # to the input body.
    body = copy.deepcopy(gateway_request.get("body", {}) or {})

    method = body.get("method", "")
    logger.info("REQUEST interceptor: method=%s", method)

    # --- Non-tools/call protocol message -> pass through unchanged ---
    # initialize, tools/list, notifications/initialized, ping, etc.
    if method != _TOOLS_CALL_METHOD:
        return _allow(body)

    params = body.get("params", {}) or {}
    tool = classify_tool(params.get("name", ""))

    # Preserve Mcp-Session-Id handling (case-insensitive lookup) even though
    # the pre-stage interceptor makes no session-state decision. NEVER log the
    # headers dict — it carries the JWT / bearer token.
    session_id = extract_session_id(headers)
    logger.info(
        "tools/call tool=%s session_id=%s",
        tool,
        session_id if session_id else "<none>",
    )

    # --- tools/call for a tool outside the scoped set -> pass through ---
    if tool not in _SCOPED_TOOLS:
        return _allow(body)

    # --- Derive served_scope from the validated Authorization JWT claim ---
    # The gateway CUSTOM_JWT authorizer already validated the token; here we
    # only read the claim. Never log the Authorization header or the token.
    # Resolve the header CASE-INSENSITIVELY: header field names are
    # case-insensitive (RFC 9110 §5.1) and the Gateway negotiates HTTP/2, which
    # lowercases field names on the wire (RFC 9113 §8.2), so the value can arrive
    # under `authorization` — a case-sensitive `headers.get("Authorization")`
    # missed it and failed closed on valid tokens (mirrors extract_session_id).
    served_scope = served_scope_from_authorization(_get_header(headers, "authorization"))

    # --- Fail closed: no verifiable scope -> short-circuit, no read occurs ---
    if not served_scope:
        logger.info("No verifiable served_scope; failing closed for %s", tool)
        return _short_circuit_error(body.get("id"), _GENERIC_ERROR_MESSAGE)

    # --- Read the model-supplied arguments (left unmodified below) ---
    # The model-supplied fields (doc_id / document_id / query / body) are never
    # touched; the interceptor adds exactly one new key, `context` (below).
    arguments = params.get("arguments", {}) or {}

    # --- Vend scoped credentials in the interceptor -----------------------------
    # The tool execution roles hold NO sts:AssumeRole and NO DynamoDB permission.
    # The interceptor assumes the read/write role with a LeadingKeys session
    # policy (DurationSeconds=900) and hands the tool short-lived,
    # partition-confined credentials as UNDECLARED params.arguments fields.
    # A compromised tool therefore cannot mint or widen any
    # credential — it can only use what it was handed. Fail CLOSED on any vend
    # failure: the tool has no fallback path to the table. NEVER log credentials.
    try:
        _t0 = time.perf_counter()
        creds = _vend_for_tool(tool, served_scope)
        _assume_ms = (time.perf_counter() - _t0) * 1000.0
    except (ClientError, BotoCoreError, KeyError, RuntimeError) as exc:
        logger.info(
            "credential vending failed for tool=%s (%s); failing closed",
            tool,
            type(exc).__name__,
        )
        return _short_circuit_error(body.get("id"), _GENERIC_ERROR_MESSAGE)

    # --- Single `context` wire contract for ALL scoped tools -------------------
    # Write EXACTLY one new key — `context` — at arguments["context"] for every
    # scoped tool (read_document, search_documents, reply). The object carries
    # served_scope + tenant_credentials (build_tenant_context). No flat
    # credential fields and no top-level `served_scope` argument are written on
    # any path, and the model-supplied arguments (doc_id /
    # query / body) are left untouched. Any value already present at
    # arguments["context"] (e.g. a model-supplied one) is OVERWRITTEN without
    # being read and without being logged — this is a plain assign, so
    # the prior value is never inspected. NEVER logs the context object or any
    # credential value.
    arguments["context"] = build_tenant_context(served_scope, creds)

    # Latency instrumentation (observability only; no secret logged). assume_ms
    # is the wall-clock of the single sts:AssumeRole vend. There is no cache and
    # no cache_hit field — every call vends its own session.
    logger.info(
        "vend_timing tool=%s served_scope=%s assume_ms=%.2f",
        tool,
        served_scope,
        _assume_ms,
    )

    params["arguments"] = arguments
    body["params"] = params
    # Evidence log: scalar fields only — the tool name and the served-scope
    # value. No model_supplied/_SCOPE_ARG, no object, no credential.
    logger.info(
        "context injection: tool=%s served_scope=%s",
        tool,
        served_scope,
    )

    return _allow(body)


# ---------------------------------------------------------------------------
# Interceptor output envelope builders
# ---------------------------------------------------------------------------


def _allow(body: dict[str, Any]) -> dict[str, Any]:
    """Build a pass-through / allow REQUEST-interceptor output envelope.

    Args:
        body: The (possibly scope-injected) JSON-RPC request body to forward.

    Returns:
        An ``interceptorOutputVersion: "1.0"`` envelope carrying
        ``mcp.transformedGatewayRequest.body``.
    """
    return {
        "interceptorOutputVersion": "1.0",
        "mcp": {"transformedGatewayRequest": {"body": body}},
    }


def _short_circuit_error(req_id: Optional[Any], text: str) -> dict[str, Any]:
    """Build a fail-closed short-circuit REQUEST-interceptor output envelope.

    When ``transformedGatewayResponse`` is present the gateway responds with it
    immediately without calling the target, so no document read occurs. The
    JSON-RPC result carries ``isError`` true and a GENERIC message with no
    scope detail.

    Args:
        req_id: The JSON-RPC request id to echo back (may be None).
        text: The generic, detail-free error message.

    Returns:
        An ``interceptorOutputVersion: "1.0"`` envelope carrying
        ``mcp.transformedGatewayResponse``.
    """
    return {
        "interceptorOutputVersion": "1.0",
        "mcp": {
            "transformedGatewayResponse": {
                "statusCode": 200,
                "body": {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": True,
                    },
                },
            }
        },
    }


# === Credential-vending helper ===============================================


def _vend_for_tool(tool: str, served_scope: str) -> dict[str, str]:
    """Vend scoped credentials for a scoped tool by assuming its role.

    Selects ``DocumentsWriteRole`` (``UpdateItem``) for ``reply`` and
    ``DocumentsAccessRole`` (``GetItem``/``Query``) for the read tools, then
    assumes it with a ``LeadingKeys`` session policy scoped to ``served_scope``
    and a ``scope`` SESSION TAG carrying the same value (``DurationSeconds=900``).
    Both are required: the roles' identity policies confine
    ``dynamodb:LeadingKeys`` to ``${aws:PrincipalTag/scope}`` and their trust
    policies reject an untagged assume. Every call vends its own session — there
    is no cache. Reads the role ARNs and table ARN from the
    environment.

    Args:
        tool: The classified scoped tool name.
        served_scope: The authoritative, JWT-derived scope.

    Returns:
        The single credentials dict from :func:`vend_scoped_credentials`
        (``access_key_id`` / ``secret_access_key`` / ``session_token``; no tuple,
        no ``cache_hit``).

    Raises:
        KeyError: If a required environment variable is unset (fail closed).
        RuntimeError: ``ScopeTagError`` when ``served_scope`` cannot be safely
            expressed as the ``scope`` session tag the scoped roles require
            — raised before ``AssumeRole``, so nothing is minted.
        botocore.exceptions.ClientError / BotoCoreError: On ``AssumeRole``
            failure (fail closed), including a trust-policy rejection of an
            untagged assume.
    """
    table_arn = os.environ[_ENV_TABLE_ARN]
    if tool in _WRITE_TOOLS:
        role_arn = os.environ[_ENV_WRITE_ROLE_ARN]
        actions = WRITE_ACTIONS
    else:
        role_arn = os.environ[_ENV_ACCESS_ROLE_ARN]
        actions = READ_ACTIONS
    return vend_scoped_credentials(role_arn, served_scope, table_arn, actions)
