"""
reply Lambda handler (scope-partitioned, scoped-credential write).

Given a ``doc_id``, a ``body``, and the interceptor-injected authoritative
``served_scope``, this handler appends one entry to the target document's
``conversation`` List attribute using a DynamoDB composite-key ``UpdateItem``
performed with *scoped WRITE temporary credentials* vended by
:func:`documents_table_from_event` — never the Lambda's own execution role.

Scope-partitioned schema:
    - Partition key: ``scope`` (the scope that owns the document).
    - Sort key:      ``doc_id``.

``scope`` is a DynamoDB reserved word; it is referenced only through the
``Key={...}`` primary-key map (which handles the escaping internally), never as a
bare attribute string in an expression.

Why reply changed:
    The old handler keyed ``UpdateItem`` on the bare identifier alone using its own
    execution role's table-wide ``dynamodb:UpdateItem`` grant. Under the composite-key
    schema a valid key cannot be built without ``served_scope``, and a table-wide
    write grant is a cross-scope WRITE channel if a forged/missing ``served_scope``
    ever reaches the tool. So reply now:
        - requires the interceptor-injected ``served_scope`` to build the
          composite ``Key={"scope": served_scope, "doc_id": doc_id}``;
        - assumes ``DocumentsWriteRole`` (credentials read from the request context)
          whose ``LeadingKeys`` write session policy confines ``UpdateItem`` to the
          served partition — dropping the tool's own table-wide ``UpdateItem``
          reliance.
    reply is still NOT an enforcement point (it makes no allow/block decision),
    but it is scoped-by-construction on the write path.

Behavior:
    - Missing/empty/whitespace ``doc_id`` OR ``body`` OR ``served_scope``:
      returns a generic error ("input is invalid") with NO DynamoDB write.
    - ``body`` longer than ``_MAX_BODY_BYTES`` when UTF-8 encoded: returns a
      specific length-limit error with NO DynamoDB write (so an oversized reply
      costs no WCUs).
    - A ``doc_id`` naming no existing document in the served partition, OR a
      ``conversation`` already holding ``_MAX_CONVERSATION_LEN`` entries: the
      ``ConditionExpression`` rejects the write and ``_APPEND_REFUSED_ERROR`` is
      returned. One message covers both causes and it is deliberately not
      disambiguated -- see the constant for why. The two bounds are sized so their
      PRODUCT (200 KB) stays well under DynamoDB's 400 KB item limit, past which
      every further write to the document fails -- a list bound alone would still
      let a document be bricked one legal reply at a time.
    - The existence half of that condition is what stops ``reply`` CREATING
      documents. ``UpdateItem`` is an upsert: absent a condition it "adds a new
      item to the table if it does not already exist", so a model-invented
      ``doc_id`` did not fail, it materialised a new item inside the served
      partition. That could not be closed in IAM -- ``dynamodb:LeadingKeys``
      constrains the partition key and IAM has no condition key for the sort key,
      so no session policy can express "this document and no other". It is closed
      in the write operation itself, evaluated by DynamoDB rather than by a branch
      here, so there is no comparison in this handler for an injection to defeat.
      Writes to OTHER EXISTING documents in the caller's own partition remain
      possible and are outside the claim.
    - ``AssumeRole`` failure while vending scoped WRITE credentials, or an
      ``UpdateItem`` that fails at the API or the transport layer (botocore
      ``ClientError`` / ``BotoCoreError``): returns a GENERIC error and NEVER
      falls back to the reply Lambda's own execution role (which holds no direct
      write permission anyway).
    - Valid input: appends the body text to the ``conversation`` list and returns
      success. The document's ``body`` attribute is NEVER mutated. Repeated reply
      calls append further entries, so "the last one" is the last list element.

AWS documentation references:
    - DynamoDB ``UpdateItem`` with the ``SET`` action, ``list_append`` (appends
      list2 to the end of list1) and ``if_not_exists(path, value)`` (returns
      ``value`` when the attribute is absent) in an update expression:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html
    - DynamoDB composite primary key (partition key + sort key); ``UpdateItem``
      requires the ENTIRE primary key in ``Key``:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.CoreComponents.html
    - ``UpdateItem`` "edits an existing item's attributes, or adds a new item to
      the table if it does not already exist" -- the upsert behaviour the
      ``attribute_exists(doc_id)`` condition below removes:
      https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_UpdateItem.html
    - A condition expression on a composite-key table is evaluated against the
      single item identified by BOTH key values, so testing a key attribute's
      presence is a test of that item's existence:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html
    - DynamoDB condition expressions: the ``size(path)`` function returns the
      number of elements when the operand is a List, and ``UpdateItem`` fails with
      ``ConditionalCheckFailedException`` when the ``ConditionExpression``
      evaluates false:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.OperatorsAndFunctions.html
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html
    - DynamoDB item-size limit: the maximum item size is 400 KB, so an item that
      reaches it can no longer be written to:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ServiceQuotas.html
    - boto3 DynamoDB resource interface (Table.update_item):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/programming-with-python.html
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

# Add the tools directory to the path so common modules can be imported at the
# Lambda runtime (mirrors tools/read_document/handler.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Scope-partitioned composite-key write using the interceptor-vended credentials.
# The REQUEST interceptor assumed DocumentsWriteRole with a LeadingKeys WRITE
# session policy and injected the scoped temporary credentials into the event;
# this tool builds its DynamoDB session from them and performs NO AssumeRole. The
# tool execution role holds NO sts:AssumeRole and NO DynamoDB permission, so it
# cannot write to the table any other way.
from common.scoped_credentials import (
    ScopedCredentialsError,
    documents_table_from_event,
    served_scope_from_event,
)

# Module logger. Evidence/latency logging only (observability); never logs the
# vended credentials.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Bounded write (unbounded-append fix)
# ---------------------------------------------------------------------------
# ``body`` was previously only checked non-empty and then appended with
# ``list_append``, with no bound on either the entry size or the resulting list
# length. Both the tool schema and the handler were silent on length, so repeated
# or large replies grew the item toward DynamoDB's 400 KB item-size limit -- past
# which EVERY subsequent write to that document hard-fails, taking the document's
# reply path down permanently -- while inflating the amount of stored untrusted
# content that read_document later feeds back to the model.
#
# Two bounds, at the two layers that can actually enforce them:
#
#   _MAX_BODY_BYTES       per-entry, enforced locally BEFORE the write, so an
#                         oversized reply costs no WCUs at all.
#   _MAX_CONVERSATION_LEN list length, enforced by DynamoDB as a
#                         ``ConditionExpression`` on the update. It has to be
#                         server-side: a read-then-write check here would race
#                         concurrent replies to the same document, whereas the
#                         condition is evaluated atomically with the append.
#
# The two numbers are chosen so their PRODUCT is the guarantee, not just each one
# individually -- otherwise a document can still be driven past 400 KB one legal
# reply at a time, which is the exact failure these bounds exist to prevent:
#
#   _MAX_BODY_BYTES * _MAX_CONVERSATION_LEN = 4000 * 50 = 200 KB
#
# leaving ~200 KB of headroom for the rest of the item (the document's own title
# and body, the keys, and DynamoDB's per-element list overhead). The per-entry
# bound is measured in UTF-8 BYTES rather than characters because that is what
# DynamoDB counts toward the item size: a character count would let 4000
# astral-plane code points occupy 16 KB, and the product would then exceed the
# item limit by a factor of four.
_MAX_BODY_BYTES = 4000
_MAX_CONVERSATION_LEN = 50

# Generic, non-scope-revealing error for the fail-closed paths (invalid input,
# AssumeRole/DynamoDB failure).
_GENERIC_ERROR = "input is invalid"

# Pre-write bound error. It names only a static, published limit -- no scope, no
# credential field, no echo of any supplied argument -- so the model can shorten its
# body instead of retrying the same rejected write. Keeping it distinct from
# _GENERIC_ERROR is what stops a too-long reply from looking like a transient
# failure worth retrying. It stays specific because, unlike the condition below, it
# has exactly one cause and naming it is therefore true.
_BODY_TOO_LONG_ERROR = (
    f"reply body exceeds the maximum length of {_MAX_BODY_BYTES} bytes (UTF-8)"
)

# The single message for a rejected ``ConditionExpression``. That condition now has
# TWO causes -- the target document does not exist, or its conversation is full --
# and this message is deliberately true of both rather than naming either.
#
# It replaced a message that named the conversation cap. Once the existence clause
# landed, that text asserted something specific and FALSE about a document that
# does not exist, and this string does not stop at the handler: it is returned to
# the model, enters its context, and can end up in what the operator reads. A
# design whose argument is that the refusal is a property of the data rather than a
# story told in code should not have the code tell a false story on the way out.
#
# The signal the old text carried survives the rewording, because what stops the
# model retrying is that the failure is PERMANENT, not that the text names the cap:
# "no reply can be appended" is terminal under either cause. The cap itself is
# published where the model reads it BEFORE choosing to call -- the reply tool
# description in ``cdk/gateway_resources.py``, guarded against drift by
# ``tests/test_published_limits.py`` -- which is the better home for a rule that is
# constant anyway.
#
# Distinguishing the two causes was considered and rejected. It is expressible:
# ``ReturnValuesOnConditionCheckFailure=ALL_OLD`` returns the item in the exception
# when one existed and nothing when it did not. The reason for rejecting it is that
# it puts a comparison back in this handler and adds a code path whose only product
# is a nicer error. It is NOT that it would create an existence oracle: that oracle
# already exists, because read_document returns one generic "document not found"
# for an absent id and the same string for a foreign one, so an agent can already
# establish whether any given sort key is present in its own partition.
_APPEND_REFUSED_ERROR = "no reply can be appended to that document"


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler for the reply tool.

    Appends one entry to the target document's ``conversation`` List in the served
    partition of the Documents table using scoped WRITE credentials.

    Args:
        event: The Lambda event — the map of ``inputSchema`` properties to their
            values, plus the interceptor-injected ``context`` object. Expected
            keys:
                - ``doc_id`` (str): the sort key of the document to append to
                  (model-supplied).
                - ``body`` (str): the text to append to the conversation
                  (model-supplied).
                - ``context`` (object): the interceptor-injected object carrying
                  the authoritative ``served_scope`` (the partition key) and the
                  scoped ``tenant_credentials``.
        context: The Lambda context (unused).

    Returns:
        A dict representing the tool response. On success: ``{"success": True}``.
        On an oversized body: ``{"error": _BODY_TOO_LONG_ERROR}``, which names only
        the static published limit. On a rejected ``ConditionExpression``:
        ``{"error": _APPEND_REFUSED_ERROR}``, which has TWO causes -- ``doc_id``
        names no existing document in the served partition, or that document's
        conversation is full -- that are deliberately NOT distinguished. The
        message is true and terminal under either, and neither names a scope nor
        echoes an argument. On any other failure (invalid input, AssumeRole
        failure, transport error): a generic ``{"error": "input is invalid"}`` with
        no scope detail.
    """
    doc_id = _clean_str(event.get("doc_id"))
    body = _clean_str(event.get("body"))

    # Validate the model-supplied arguments: either of doc_id / body missing
    # or empty is an invalid request. Generic message, and NO DynamoDB write.
    if not doc_id or not body:
        return {"error": _GENERIC_ERROR}

    # Per-entry size bound, enforced BEFORE the write so an oversized reply
    # consumes no WCUs at all. Measured in UTF-8 bytes because that is what counts
    # toward DynamoDB's item size -- see the constants above for why a character
    # count would break the product guarantee. The message is distinct from the
    # generic error so the model shortens the body instead of retrying the
    # identical rejected call.
    if len(body.encode("utf-8")) > _MAX_BODY_BYTES:
        return {"error": _BODY_TOO_LONG_ERROR}

    # Read the authoritative served scope and build the scoped WRITE table client
    # from the interceptor-injected `context` object:
    # served_scope_from_event -> event["context"]["served_scope"];
    # documents_table_from_event -> event["context"]["tenant_credentials"] (the
    # DocumentsWriteRole credentials the interceptor vended with a LeadingKeys
    # write session policy confining UpdateItem to the served partition). A
    # missing/malformed context fails CLOSED: served_scope_from_event raises
    # ScopedCredentialsError, so we return a generic error and NEVER fall back to
    # the Lambda execution role (which holds no direct write permission) or the
    # default credential chain. The generic message names no scope and no
    # credential field, and echoes no part of event/arguments/context.
    #
    # UpdateExpression appends the body to the conversation list:
    #   SET conversation = list_append(if_not_exists(conversation, :empty), :entry)
    #   :empty  -> empty list, used when conversation does not yet exist
    #   :entry  -> single-element list containing the reply body text
    # This appends rather than overwrites; the document's `body` is never mutated.
    #
    # ConditionExpression carries TWO bounds, evaluated server-side and atomically
    # with the append:
    #
    #   attribute_exists(doc_id)
    #     AND (attribute_not_exists(conversation) OR size(conversation) < :max)
    #
    # ``attribute_exists(doc_id)`` is the write-target existence check. On a
    # composite-key table the condition is evaluated against the single item named
    # by BOTH key values, so a key attribute's presence IS that item's existence;
    # ``doc_id`` is the sort key, so it is structurally present on every item that
    # exists and the check cannot be evaded by a document lacking `title` or
    # `body`. It is also not a reserved word, so it needs no
    # ExpressionAttributeNames and `scope` still appears only inside `Key={...}`.
    # Without this clause ``UpdateItem`` is an upsert, and a doc_id the model
    # invented out of text it had just read did not fail -- it created a new item
    # in the served partition.
    #
    # ``attribute_not_exists(conversation) OR size(conversation) < :max_entries``
    # is the pre-existing list-length cap. ``size()`` on a List is its element
    # count, so this admits the first reply (the attribute is absent on seed) and
    # every reply up to the cap.
    #
    # The parentheses are for the reader, not for the machine. ``AND`` binds
    # tighter than ``OR`` in DynamoDB's precedence table, so the unparenthesised
    # form would parse as ``(exists AND not_exists) OR size < max`` -- which
    # happens to evaluate identically in every reachable state, because on an
    # absent item the size comparison is false too (a comparison on a missing
    # operand does not evaluate true). No test distinguishes the two forms; the
    # parentheses exist so nobody has to redo that reasoning.
    #
    # Both bounds MUST be conditions rather than read-then-write checks here: the
    # condition is evaluated atomically with the append, so concurrent replies to
    # the same document cannot race past either bound. Neither adds an IAM
    # requirement -- the LeadingKeys write session policy already permits
    # UpdateItem on this key.
    try:
        served_scope = served_scope_from_event(event)
        table = documents_table_from_event(event)
        _t0 = time.perf_counter()
        table.update_item(
            # Composite key REQUIRED: PK ``scope`` + SK ``doc_id``. The model-supplied
            # identifier value is read from the ``doc_id`` event field — the published
            # identifier contract. ``scope`` is a DynamoDB reserved word, referenced
            # only through this ``Key={...}`` map, which handles the escaping
            # internally.
            Key={"scope": served_scope, "doc_id": doc_id},
            UpdateExpression=(
                "SET conversation = list_append("
                "if_not_exists(conversation, :empty), :entry)"
            ),
            ConditionExpression=(
                "attribute_exists(doc_id) "
                "AND (attribute_not_exists(conversation) "
                "OR size(conversation) < :max_entries)"
            ),
            ExpressionAttributeValues={
                ":empty": [],
                ":entry": [body],
                ":max_entries": _MAX_CONVERSATION_LEN,
            },
        )
        # Latency instrumentation (observability only).
        logger.info(
            "ddb_timing tool=reply op=update_item ms=%.2f",
            (time.perf_counter() - _t0) * 1000.0,
        )
    except ClientError as exc:
        # A rejected ConditionExpression means either the target document does not
        # exist or its conversation cap was reached. Report that with one terminal
        # message so the model stops retrying a write that can never succeed,
        # without asserting which of the two it was; every OTHER ClientError stays
        # generic (no scope detail, no exec-role fallback).
        code = exc.response.get("Error", {}).get("Code")
        if code == "ConditionalCheckFailedException":
            return {"error": _APPEND_REFUSED_ERROR}
        return {"error": _GENERIC_ERROR}
    except BotoCoreError:
        # ``BotoCoreError`` is a SIBLING of ``ClientError``, not a superclass, so
        # it needs its own branch: the transport-level failures
        # (EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError) derive
        # from it alone and would otherwise escape as a raw traceback. It carries
        # no ``response`` payload, hence no condition-failure check here — an
        # append that never reached DynamoDB is simply a generic failure.
        return {"error": _GENERIC_ERROR}
    except ScopedCredentialsError:
        # Vended-cred failure. Generic error, no exec-role fallback.
        return {"error": _GENERIC_ERROR}

    return {"success": True}


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
