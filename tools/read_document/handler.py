"""
read_document Lambda handler (scope-partitioned, scoped-credential read).

Given a ``doc_id`` and the interceptor-injected authoritative ``served_scope``,
this handler performs a strongly-consistent composite-key ``GetItem`` against the
Documents table using *scoped temporary credentials* vended by
:func:`documents_table_from_event` — never the Lambda's own execution role.

Scope-partitioned schema:
    - Partition key: ``scope`` (the scope that owns the document).
    - Sort key:      ``doc_id``.

``scope`` is a DynamoDB reserved word; it is referenced only through the
``Key={...}`` primary-key map (which handles the escaping internally), never as a
bare attribute string in a projection/filter/condition expression.

The composite key means a foreign ``doc_id`` resolved against the served
partition simply does not exist there, so a cross-scope read surfaces as a
generic *not found* — no foreign data ever reaches the model.

Behavior:
    - Missing/empty/whitespace ``doc_id`` OR ``served_scope``: returns a
      generic error ("document identifier is invalid") with no DynamoDB call.
    - ``AssumeRole`` failure while vending scoped credentials, or a DynamoDB call
      that fails at the API or the transport layer (botocore ``ClientError`` /
      ``BotoCoreError``): returns a GENERIC error and NEVER falls back to the
      Lambda execution role for the read.
    - Well-formed key not found in the served partition: returns a generic
      "document not found" error with no body or scope in the response.
    - Valid existing key: returns ``body`` and ``scope`` from the document.

The handler is NOT an enforcement point and records no session state. The
authoritative ``served_scope`` is injected by the REQUEST interceptor; the
scoped session policy (LeadingKeys) is the defense-in-depth fallback.

AWS documentation references:
    - DynamoDB ``GetItem`` requires the ENTIRE primary key for a composite-key
      table (partition key AND sort key) and supports the optional
      ``ConsistentRead`` parameter for a strongly consistent read:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithItems.html
    - DynamoDB composite primary key (partition key + sort key):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.CoreComponents.html
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

# Add the tools directory to the path so common modules can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Scope-partitioned composite-key read using the interceptor-vended credentials.
# The REQUEST interceptor performed sts:AssumeRole with a LeadingKeys session
# policy and injected the scoped temporary credentials into the event; this tool
# builds its DynamoDB session from them and performs NO AssumeRole. The tool
# execution role holds NO sts:AssumeRole and NO DynamoDB permission, so it cannot
# reach the table any other way.
# Wire contract: read the authoritative served scope from
# event["context"]["served_scope"] via served_scope_from_event, and build the
# partition-confined DynamoDB Table from event["context"]["tenant_credentials"]
# via documents_table_from_event. Both readers fail CLOSED (raise
# ScopedCredentialsError) on a missing/malformed context and NEVER fall back to
# the execution role or the default chain. The handler calls the two readers
# directly, consistent with search_documents and reply.
from common.scoped_credentials import (
    ScopedCredentialsError,
    documents_table_from_event,
    served_scope_from_event,
)

# Module logger. Evidence logging only (observability); never logs the
# Authorization header or the JWT (those never reach this Lambda anyway).
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler for the read_document tool.

    Reads a single document from the served-scope partition using scoped
    temporary credentials.

    Args:
        event: The Lambda event — the map of ``inputSchema`` properties to
            their values, plus the interceptor-injected ``context`` object.
            Expected keys:
                - ``doc_id`` (str): the declared sort key of the document
                  to read (model-supplied).
                - ``context`` (object): the interceptor-injected object carrying
                  ``served_scope`` and ``tenant_credentials``.
        context: The Lambda context (unused).

    Returns:
        A dict representing the tool response. On success, contains ``body`` and
        ``scope``. On any failure (invalid input, missing/malformed context,
        AssumeRole failure, item not found), contains a generic ``error``
        message with no scope detail.
    """
    doc_id = _clean_str(event.get("doc_id"))

    # Read the served scope AND build the scoped DynamoDB client from the injected
    # `context` object (event["context"]) — NOT from any flat top-level field.
    # A missing/malformed context fails CLOSED: generic error, and the tool NEVER
    # falls back to its execution role (which holds no DynamoDB permission) or the
    # default credential chain. The generic message names no scope and no
    # credential field, and echoes no part of event/arguments/context.
    try:
        served_scope = served_scope_from_event(event)
        table = documents_table_from_event(event)
    except ScopedCredentialsError:
        return {"error": "document identifier is invalid"}

    # Validate the model-supplied identifier. Generic message — never echo which
    # field or scope was involved.
    if not doc_id:
        return {"error": "document identifier is invalid"}

    # Use the interceptor-vended scoped credentials. A DynamoDB failure MUST
    # surface as a generic error; we NEVER fall back to the Lambda execution role
    # (which holds no DynamoDB permission). The composite-key attribute names are
    # the published contract: partition key ``scope`` + sort key ``doc_id``, and
    # the model-supplied identifier value is read from the ``doc_id`` event field.
    # ``scope`` is a DynamoDB reserved word, so it is referenced ONLY through the
    # ``Key={...}`` primary-key map, which handles the escaping internally — never
    # as a bare attribute string in an expression.
    try:
        _t0 = time.perf_counter()
        resp = table.get_item(
            Key={"scope": served_scope, "doc_id": doc_id},
            ConsistentRead=True,
        )
        # Latency instrumentation (observability only).
        logger.info(
            "ddb_timing tool=read_document op=get_item ms=%.2f",
            (time.perf_counter() - _t0) * 1000.0,
        )
    except (BotoCoreError, ClientError, ScopedCredentialsError):
        # Vended-cred / DynamoDB client failure. Generic error, no fallback.
        # ``BotoCoreError`` is a SIBLING of ``ClientError``, not a superclass, so
        # it must be named explicitly: the transport-level failures
        # (EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError) derive
        # from it alone and would otherwise escape as a raw traceback.
        return {"error": "document not found"}

    item = resp.get("Item")
    if item is None:
        # Foreign ids resolved against the served partition simply do not exist
        # here (item-absence, the normal-path control). Generic not-found.
        return {"error": "document not found"}

    # Document found in the served partition — return body and owning scope. The
    # owning scope is read from the item's real ``scope`` attribute. This is a plain
    # dict access on the returned item, not a DynamoDB expression, so the reserved
    # word needs no alias here.
    return {
        "body": item.get("body", ""),
        "scope": item["scope"],
    }


def _clean_str(value: Any) -> str:
    """Return a trimmed string for a tool argument, or "" if not a valid string.

    Args:
        value: The raw argument value from the Lambda event.

    Returns:
        The stripped string if ``value`` is a non-empty string; otherwise "".
    """
    if not isinstance(value, str):
        return ""
    return value.strip()
