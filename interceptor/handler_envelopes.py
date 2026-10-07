"""REQUEST-interceptor output envelope builders.

Extracted from ``interceptor/handler.py``, which had reached this repository's
400-line per-module limit. These two functions form
a natural boundary: they know the Gateway's wire contract and nothing about scopes,
identity, or credentials, so they are the part of the handler that can be read and
reviewed without any of its security reasoning.

Which envelope is returned is the allow/deny decision itself:

- ``allow`` carries ``mcp.transformedGatewayRequest.body``, so the Gateway forwards
  the (possibly enriched) request to the target.
- ``short_circuit_error`` carries ``mcp.transformedGatewayResponse``, which the
  Gateway answers with IMMEDIATELY without calling the target — that is what makes
  failing closed mean "no document read occurs" rather than "an error was logged".

Reference (AWS Documentation MCP server, per the ``aws-docs-lookup`` rule):
    - REQUEST interceptor input/output contract, incl. the short-circuit behaviour of
      ``mcp.transformedGatewayResponse``:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-types.html

Functions:
    allow: Build the pass-through / allow envelope.
    short_circuit_error: Build the fail-closed short-circuit envelope.
"""

from __future__ import annotations

from typing import Any, Optional

#: The only output contract version this interceptor speaks.
INTERCEPTOR_OUTPUT_VERSION = "1.0"


def allow(
    body: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Build a pass-through / allow REQUEST-interceptor output envelope.

    ``headers``, when supplied, is emitted as
    ``mcp.transformedGatewayRequest.headers``. The Gateway MERGES those headers
    with the target's ``metadataConfiguration.allowedRequestHeaders`` allowlist and
    forwards them to the target, with interceptor-provided values taking precedence
    over client-provided ones. A header NOT in the target's allowlist is dropped
    (``Authorization`` is the single documented exception), so this argument alone
    does not make a header arrive — the target must allowlist it too.

    Omitting ``headers`` leaves the ``headers`` key out of the envelope entirely,
    so existing callers keep emitting the exact body-only shape they did before.

    Reference (AWS Documentation MCP server, per the ``aws-docs-lookup`` rule):
        - Header propagation from an interceptor lambda (the
          ``transformedGatewayRequest.headers`` shape, allowlist merge, and
          interceptor-over-client precedence):
          https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-headers.html

    Args:
        body: The (possibly scope-injected) JSON-RPC request body to forward.
        headers: Optional request headers to propagate to the target. Each name
            must satisfy the Gateway's ``^[a-zA-Z0-9_-]+$`` rule and must be
            allowlisted on the target to be forwarded.

    Returns:
        An ``interceptorOutputVersion: "1.0"`` envelope carrying
        ``mcp.transformedGatewayRequest.body``, plus
        ``mcp.transformedGatewayRequest.headers`` when ``headers`` is non-empty.
    """
    transformed: dict[str, Any] = {"body": body}
    if headers:
        transformed["headers"] = dict(headers)
    return {
        "interceptorOutputVersion": INTERCEPTOR_OUTPUT_VERSION,
        "mcp": {"transformedGatewayRequest": transformed},
    }


def short_circuit_error(req_id: Optional[Any], text: str) -> dict[str, Any]:
    """Build a fail-closed short-circuit REQUEST-interceptor output envelope.

    When ``transformedGatewayResponse`` is present the gateway responds with it
    immediately without calling the target, so no document read occurs. The
    JSON-RPC result carries ``isError`` true and a GENERIC message with no
    scope detail (Requirements 3.7, 3.8).

    Args:
        req_id: The JSON-RPC request id to echo back (may be None).
        text: The generic, detail-free error message.

    Returns:
        An ``interceptorOutputVersion: "1.0"`` envelope carrying
        ``mcp.transformedGatewayResponse``.
    """
    return {
        "interceptorOutputVersion": INTERCEPTOR_OUTPUT_VERSION,
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
