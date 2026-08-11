"""
search_documents Lambda handler (scope-partitioned, scoped-credential query).

This handler implements the Search_Documents_Tool. Given a keyword query and the
authoritative ``served_scope`` injected by the REQUEST interceptor, it issues a
DynamoDB **partition ``Query``** -- never a full-table ``Scan`` -- confined to the
served-scope partition via scoped temporary credentials, then filters the returned
items in memory for case-insensitive substring matches of the query against the
``title`` field ONLY (never the body).

Why ``Query`` and not ``Scan``:
    The scoped session policy (``tools/common/scoped_credentials.py``) grants only
    ``dynamodb:GetItem`` + ``dynamodb:Query`` gated by ``dynamodb:LeadingKeys``.
    ``dynamodb:Scan`` reads ALL partitions, carries no leading key, and is NOT among
    the allowed actions -- it would both fail the policy and defeat scope confinement.
    A partition ``Query`` whose ``KeyConditionExpression`` pins ``served_scope`` to the
    interceptor-injected value can only ever read the served partition, so cross-scope
    document discovery is impossible by construction: foreign documents are outside the
    query, not merely filtered out.

Behavior:
    - Missing/empty/whitespace ``query`` or ``served_scope``: returns a generic error
      response ("query is invalid") with no DynamoDB call.
    - ``AssumeRole`` failure while vending scoped credentials: returns a generic error
      ("query is invalid"); NEVER falls back to the Lambda execution role.
    - No matches: returns ``{"results": [], "truncated": False}``.
    - Matches: returns ``{"results": [...], "truncated": <bool>}`` where each entry
      contains only ``doc_id``, ``scope``, and ``title`` (never the body), and every
      entry belongs to the served-scope partition. The walk is BOUNDED -- at most
      ``_MAX_RESULTS`` matches are returned and at most ``_MAX_PAGE_READS`` pages
      are read -- and ``truncated`` reports whether a cap stopped it early, with a
      static ``note`` added when it did so the model knows the set is partial.

AWS documentation references:
    - DynamoDB Query key condition expressions -- the partition key must be specified
      as an equality condition (``Key("scope").eq(...)``). ``scope`` is a DynamoDB
      reserved word, and the ``boto3.dynamodb.conditions.Key("scope")`` builder emits
      the required ``#``-prefixed expression-attribute-name alias internally, so
      ``scope`` is never a bare attribute string in the expression:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.KeyConditionExpressions.html
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.KeyConditionExpressions.html
    - DynamoDB Query pagination -- use ``LastEvaluatedKey`` from a response as the
      ``ExclusiveStartKey`` of the next request until it is absent:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.Pagination.html
    - DynamoDB ``Limit`` is a PER-REQUEST maximum on the number of items evaluated,
      not a cap on the total across a paginated walk; a response can carry a
      ``LastEvaluatedKey`` purely because ``Limit`` was reached, so a total bound
      must come from the caller stopping the pagination loop:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.html
      https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_Query.html
    - boto3 DynamoDB resource interface (Table.query, boto3.dynamodb.conditions.Key):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/programming-with-python.html
    - ``dynamodb:Query`` is the correct fine-grained action gated by
      ``dynamodb:LeadingKeys`` (listed alongside GetItem under a ForAllValues
      LeadingKeys condition):
      https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_dynamodb_items.html
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from boto3.dynamodb.conditions import Key
from botocore.exceptions import BotoCoreError, ClientError

# Add the tools directory to the path so common modules can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Scope-partitioned partition Query using the interceptor-vended credentials.
# The REQUEST interceptor performed sts:AssumeRole with a LeadingKeys session
# policy and injected the scoped temporary credentials into the event; this tool
# builds its DynamoDB session from them and performs NO AssumeRole. A full-table
# Scan through the Lambda execution role is not an option — a Scan reads all
# partitions and is not permitted by the session policy.
from common.scoped_credentials import (
    ScopedCredentialsError,
    documents_table_from_event,
    served_scope_from_event,
)

# Module logger. Evidence logging only (observability); never logs the
# Authorization header or the JWT (those never reach this Lambda anyway).
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Generic, non-scope-revealing error message. Reused for every failure mode
# (invalid input AND AssumeRole failure) so no scope detail leaks to the model.
_GENERIC_ERROR = "query is invalid"

# ---------------------------------------------------------------------------
# Bounded search (unbounded-paging fix)
# ---------------------------------------------------------------------------
# Before these caps the handler walked EVERY page of the served partition and
# buffered every title match, so latency, consumed RCUs and the size of the
# result set injected into the model context all scaled with the partition size
# rather than with the number of results asked for. The Query is pinned to the
# interceptor-injected served_scope, so that is a self-inflicted cost and
# context-window lever rather than a cross-scope read -- but the model chooses
# the query string, so an injected instruction can issue deliberately broad
# queries repeatedly to amplify it.
#
# Two independent bounds, because neither alone is sufficient:
#
#   _MAX_RESULTS    caps what is RETURNED (the model-context / response-size
#                   bound). Matching happens in memory after the read, so a
#                   DynamoDB ``Limit`` cannot express this.
#   _MAX_PAGE_READS caps what is READ (the RCU / latency bound) for a query
#                   that matches few or no titles and would otherwise page the
#                   whole partition looking for more.
#
# ``Limit`` is per-request, NOT a total: it bounds the items evaluated per page
# so that _MAX_PAGE_READS * _PAGE_ITEM_LIMIT is a deterministic ceiling on items
# examined. On its own ``Limit`` would only shrink pages and add round trips --
# the total bound comes from breaking the pagination loop.
#
# Worst case per invocation: 5 Query requests, 500 items examined, 25 returned.
_MAX_RESULTS = 25
_MAX_PAGE_READS = 5
_PAGE_ITEM_LIMIT = 100


def _matches_title(item: dict[str, Any], query_lower: str) -> bool:
    """Check if the item's title contains the query as a substring.

    Matching is case-insensitive and performed ONLY against the ``title`` field --
    never against the body.

    Args:
        item: A document item from the served-scope partition.
        query_lower: The search query, already stripped and lowercased.

    Returns:
        True if the title contains the query substring (case-insensitive).
    """
    title = item.get("title", "")
    if not isinstance(title, str):
        return False
    return query_lower in title.lower()


def _query_served_partition(
    table: Any,
    served_scope: str,
    query_lower: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Query the served-scope partition and return title-matching documents.

    Issues a DynamoDB ``Query`` (on the interceptor-vended, partition-confined
    ``table``) whose ``KeyConditionExpression`` pins the partition key ``scope`` to
    the injected value via ``Key("scope").eq(...)``, so the query only ever reads
    that one partition. ``scope`` is a DynamoDB reserved word, referenced only
    through the ``Key("scope")`` builder (which emits the alias internally), never
    as a bare attribute string. Follows ``LastEvaluatedKey`` pagination across
    pages, filtering each page in memory for case-insensitive title substring
    matches.

    Pagination is BOUNDED (see the ``_MAX_RESULTS`` / ``_MAX_PAGE_READS`` /
    ``_PAGE_ITEM_LIMIT`` module constants): the walk stops once
    ``_MAX_RESULTS`` matches have been collected or ``_MAX_PAGE_READS`` pages have
    been read, whichever comes first, and reports whether it stopped early so the
    caller can tell the model the result set is partial.

    Args:
        table: The DynamoDB ``Table`` built from the interceptor-vended
            credentials (:func:`documents_table_from_event`).
        served_scope: The authoritative, interceptor-injected scope partition.
        query_lower: The search query, already stripped and lowercased.

    Returns:
        A ``(results, truncated)`` pair. ``results`` is a list of at most
        ``_MAX_RESULTS`` matching documents, each a dict with only ``doc_id``,
        ``scope``, and ``title``; all entries belong to ``served_scope``.
        ``truncated`` is True when a cap stopped the walk while more of the
        partition remained unexamined, and False when the partition was read to
        exhaustion (so the result set is known to be complete). It is decided per
        MATCH, not per item: a partition holding exactly ``_MAX_RESULTS`` matches
        followed by non-matching items is reported complete.

    Raises:
        botocore.exceptions.ClientError: If the ``Query`` is rejected by DynamoDB.
            The caller surfaces a generic error and does NOT fall back to the
            execution role.
        botocore.exceptions.BotoCoreError: If the ``Query`` fails below the API
            layer (endpoint resolution, connection, read timeout). A sibling of
            ``ClientError``, so the caller must catch it separately.
    """
    results: list[dict[str, Any]] = []
    query_kwargs: dict[str, Any] = {
        "KeyConditionExpression": Key("scope").eq(served_scope),
        # Per-request item cap (not a total) — see the module constants.
        "Limit": _PAGE_ITEM_LIMIT,
    }
    pages_read = 0

    while True:
        page = table.query(**query_kwargs)
        pages_read += 1

        for item in page.get("Items", []):
            if not _matches_title(item, query_lower):
                continue
            if len(results) >= _MAX_RESULTS:
                # A genuine MATCH exists beyond the cap, so the set is partial.
                # Testing this per match rather than per item matters: a partition
                # holding exactly _MAX_RESULTS matches followed by non-matching
                # items is complete, and a per-item check would wrongly flag it
                # partial and push the model into re-querying for results that do
                # not exist. Scanning the rest of the page to learn this is free --
                # the page was already read, and matching is in memory.
                return results, True
            # Read the item attributes ``doc_id`` and ``scope``. These are plain
            # dict reads on the returned item, not DynamoDB expressions, so the
            # reserved word ``scope`` needs no alias here. The response field key
            # is ``doc_id`` — the published identifier contract.
            results.append({
                "doc_id": item.get("doc_id", ""),
                "scope": item.get("scope", ""),
                "title": item.get("title", ""),
            })

        # Paginate: LastEvaluatedKey present and non-null means more pages remain.
        last_key = page.get("LastEvaluatedKey")
        if not last_key:
            # Partition read to exhaustion: the result set is complete, even if the
            # cap was reached on the final match of the final page.
            return results, False
        if len(results) >= _MAX_RESULTS or pages_read >= _MAX_PAGE_READS:
            # A cap stopped the walk with pages still unread. Reporting partial is
            # conservative here -- the unread pages might hold no match at all --
            # but establishing that would mean reading another page, which is the
            # cost this bound exists to avoid. Over-reporting partial is the safe
            # direction: the opposite would present a capped set as complete.
            return results, True
        query_kwargs["ExclusiveStartKey"] = last_key


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler for the search_documents tool.

    Args:
        event: The Lambda event (map of ``inputSchema`` properties plus the
            interceptor-injected ``context`` object). Expected keys:
            "query" (str, model-supplied) and "context" (object), from which the
            authoritative served scope and scoped credentials are read.
        context: The Lambda context (unused).

    Returns:
        A dict representing the tool response. On success,
        ``{"results": [...], "truncated": <bool>}`` with only served-scope matches
        (each with only ``doc_id``, ``scope``, ``title``), at most
        ``_MAX_RESULTS`` of them. ``truncated`` is True when a cap stopped the
        partition walk early, in which case a static ``note`` is added telling the
        model the result set is partial. On any error, ``{"error": <generic
        message>}``.
    """
    query = event.get("query")

    # Validate the model-supplied query argument: missing/empty/whitespace or a
    # non-string value is invalid. Generic message so no detail leaks.
    if query is None or not isinstance(query, str) or not query.strip():
        return {"error": _GENERIC_ERROR}

    query_lower = query.strip().lower()

    # Read the authoritative served scope and build the partition-confined table
    # client from the interceptor-injected `context` object:
    # served_scope_from_event -> event["context"]["served_scope"];
    # documents_table_from_event -> event["context"]["tenant_credentials"].
    # A missing/malformed context fails CLOSED: served_scope_from_event raises
    # ScopedCredentialsError, so we return a generic error and NEVER fall back to
    # the execution role (which holds no DynamoDB permission) or the default
    # credential chain. The generic message names no scope and no credential
    # field, and echoes no part of event/arguments/context. On a query failure we
    # likewise return the generic error and never fall back.
    try:
        served_scope = served_scope_from_event(event)
        table = documents_table_from_event(event)
        # Evidence logging (observability only): the authoritative served-scope
        # scalar. Never logs the event, arguments, context, or credentials.
        logger.info("search_documents served_scope=%r", served_scope)
        _t0 = time.perf_counter()
        results, truncated = _query_served_partition(table, served_scope, query_lower)
        # Latency instrumentation (observability only). Covers the full partition
        # Query including any LastEvaluatedKey pagination.
        logger.info(
            "ddb_timing tool=search_documents op=query ms=%.2f",
            (time.perf_counter() - _t0) * 1000.0,
        )
    # ``BotoCoreError`` is a SIBLING of ``ClientError``, not a superclass: the
    # transport-level failures (EndpointConnectionError, ReadTimeoutError,
    # ConnectTimeoutError) derive from it alone. Catching only ``ClientError``
    # let those escape the handler and surface a raw traceback to the caller
    # instead of this generic error, so both branches of the botocore hierarchy
    # are named here.
    except (BotoCoreError, ClientError, ScopedCredentialsError):
        return {"error": _GENERIC_ERROR}

    # ``truncated`` is always present so the model never has to infer completeness
    # from the result count. The note is a STATIC string: it explains the cap
    # without echoing the query, the scope, or any part of the injected context.
    # It names no specific number because either bound can be the one that fired
    # -- a page-cap stop can return FEWER than _MAX_RESULTS matches, and claiming
    # "capped at 25" there would hand the model a false explanation. The concrete
    # limit is published in the tool description instead.
    response: dict[str, Any] = {"results": results, "truncated": truncated}
    if truncated:
        response["note"] = (
            "The search stopped at a result or read limit and did not cover the "
            "whole document set; more matches may exist. Narrow the query instead "
            "of repeating it."
        )
    return response






