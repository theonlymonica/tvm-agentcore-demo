"""
Tool classification and session-id extraction for the gateway interceptor.

This module provides two functions used by the scope-injecting REQUEST
interceptor (``interceptor/handler.py``):

1. classify_tool — determines which known tool an action name corresponds to,
   using the confirmed triple-underscore delimiter `___` and an EXACT match on
   the segment after the final delimiter. There is no suffix/`endswith`
   fallback: it widened the match surface and was a scope-confinement bypass.
   An absent delimiter fails closed.

2. extract_session_id — reads the `Mcp-Session-Id` header from the gateway
   request headers, performing a case-insensitive lookup.

Security note: request headers carry the JWT / bearer token and MUST NEVER
be logged.

Functions:
    classify_tool: Classify a gateway action name into a known tool or UNCLASSIFIABLE.
    extract_session_id: Extract the Mcp-Session-Id from gateway request headers.

References:
    - Gateway tool naming (triple-underscore delimiter):
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html
    - Types of interceptors (passRequestHeaders, Mcp-Session-Id in headers):
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-types.html
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Tool classification constants
# ---------------------------------------------------------------------------

# The confirmed gateway delimiter between target name and tool name.
GATEWAY_DELIMITER = "___"

# Known tool name constants — the three tools in the demo.
TOOL_READ_DOCUMENT = "read_document"
TOOL_SEARCH_DOCUMENTS = "search_documents"
TOOL_REPLY = "reply"

# Sentinel returned when the action name cannot be mapped to a known tool.
TOOL_UNCLASSIFIABLE = "UNCLASSIFIABLE"

# The set of known tool suffixes used for classification.
_KNOWN_TOOLS = frozenset({TOOL_READ_DOCUMENT, TOOL_SEARCH_DOCUMENTS, TOOL_REPLY})


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def classify_tool(action_name: str | None) -> str:
    """Classify the gateway action name into a known tool or UNCLASSIFIABLE.

    The gateway forms action names as `{targetName}___{toolName}` using the
    confirmed triple-underscore delimiter. This function splits on the FINAL
    delimiter and requires the trailing segment to be EXACTLY a known tool name.

    There is deliberately no `endswith`/suffix fallback: it widened the match
    surface so a name crafted to end in a known tool (e.g.
    `evil___fake_read_document`) classified as that scoped tool and would be
    handed vended credentials. Because the gateway
    always emits `${target}___${tool}`, an absent delimiter is illegitimate and
    fails closed.

    If the delimiter is absent, or the trailing segment is not exactly a known
    tool name, TOOL_UNCLASSIFIABLE is returned so the caller can fail closed.

    Args:
        action_name: The full gateway action name (e.g.
            "ReadDocument___read_document") or the tool name field from the
            MCP body. May be None or empty.

    Returns:
        One of TOOL_READ_DOCUMENT, TOOL_SEARCH_DOCUMENTS, TOOL_REPLY, or
        TOOL_UNCLASSIFIABLE.

    Examples:
        >>> classify_tool("ReadDocument___read_document")
        'read_document'
        >>> classify_tool("SearchDocuments___search_documents")
        'search_documents'
        >>> classify_tool("Reply___reply")
        'reply'
        >>> classify_tool("UnknownTarget___unknown_tool")
        'UNCLASSIFIABLE'
        >>> classify_tool(None)
        'UNCLASSIFIABLE'
    """
    if not action_name or not isinstance(action_name, str):
        return TOOL_UNCLASSIFIABLE

    # Gateway tool names are ALWAYS "${target_name}___${tool_name}" (AWS gateway
    # tool-naming rule), so the delimiter is always present in a legitimately
    # routed tools/call name. An absent delimiter is therefore illegitimate and
    # MUST fail closed rather than be suffix-guessed.
    if GATEWAY_DELIMITER not in action_name:
        return TOOL_UNCLASSIFIABLE

    # Classify ONLY by EXACT match on the segment after the FINAL delimiter.
    # There is deliberately NO endswith/suffix fallback: it widened the match
    # surface and was a scope-confinement bypass — a name crafted to END in a
    # known tool (e.g. "evil___fake_read_document" or "x___notread_document")
    # would classify as that scoped tool and be handed vended credentials it
    # should never receive. Exact match closes it.
    suffix = action_name.rsplit(GATEWAY_DELIMITER, maxsplit=1)[-1]
    if suffix in _KNOWN_TOOLS:
        return suffix

    return TOOL_UNCLASSIFIABLE


def extract_session_id(headers: dict[str, Any] | None) -> str | None:
    """Extract the Mcp-Session-Id from gateway request headers.

    Performs a case-insensitive lookup since HTTP header names are
    case-insensitive and gateways may normalize casing differently.

    SECURITY: This function intentionally never logs the headers dict
    because it carries the JWT / bearer token.

    Args:
        headers: The headers dict from gatewayRequest.headers. May be None.

    Returns:
        The Mcp-Session-Id string value if found and non-empty, or None.
    """
    if not headers or not isinstance(headers, dict):
        return None

    # Case-insensitive search for the Mcp-Session-Id header.
    target_lower = "mcp-session-id"
    for key, value in headers.items():
        if key.lower() == target_lower:
            if value and isinstance(value, str) and value.strip():
                return value.strip()
    return None
