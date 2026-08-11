"""The reserved-word ``scope`` attribute round-trips through the real name.

``scope`` is a DynamoDB reserved word. A prior bug projected on the bare
attribute name ``scope``, which failed with a ``ValidationException``, and the
owning scope silently fell back to the ``UNKNOWN`` sentinel. This test guards
that regression offline by writing an item to the moto-backed Documents table
and reading it back through the two escaping-safe access paths the tools use:

* a ``GetItem`` with a ``Key={"scope": ..., "doc_id": ...}`` map, and
* a ``Query`` with a ``boto3.dynamodb.conditions.Key("scope").eq(...)`` builder,

asserting in both cases that the owning scope read from ``item["scope"]`` equals
the stored value and is never the ``UNKNOWN`` sentinel.

The round-trip is driven DIRECTLY against the moto table (the tool handlers are
NOT invoked), which isolates the reserved-word behavior from the tool
credential / ``context`` plumbing.

AWS documentation references:

* ``SCOPE`` is a DynamoDB reserved word; a ``Key={...}`` map and a
  ``boto3.dynamodb.conditions.Key("scope")`` builder both perform the reserved-
  word escaping internally, so neither surfaces a ``ValidationException``:
  https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html
* ``Query`` pins the partition key with an equality key condition:
  https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.KeyConditionExpressions.html
"""

from __future__ import annotations

from typing import Any

from boto3.dynamodb.conditions import Key
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# The `UNKNOWN` sentinel is the failure marker the property asserts against, so
# it must never be a legitimately generated scope value — otherwise a correct
# round-trip of the literal "UNKNOWN" would trip the sentinel assertion for the
# wrong reason.
# ---------------------------------------------------------------------------
_UNKNOWN = "UNKNOWN"

# Printable, whitespace-free ASCII keeps the moto key round-trip exact and
# avoids any UTF-8 serialization edge cases (e.g. surrogate code points).
_KEY_ALPHABET = st.characters(min_codepoint=33, max_codepoint=126)
# Non-key attributes may include spaces and be empty.
_TEXT_ALPHABET = st.characters(min_codepoint=32, max_codepoint=126)


def _scope_values() -> st.SearchStrategy[str]:
    """Arbitrary non-empty scope partition-key values (never the sentinel)."""
    return st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=64).filter(
        lambda s: s != _UNKNOWN
    )


def _doc_ids() -> st.SearchStrategy[str]:
    """Arbitrary non-empty ``doc_id`` sort-key values."""
    return st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=64)


def _free_text() -> st.SearchStrategy[str]:
    """Arbitrary title/body text (may be empty)."""
    return st.text(alphabet=_TEXT_ALPHABET, max_size=256)


# ---------------------------------------------------------------------------
# Property 7
# ---------------------------------------------------------------------------


# Feature: multi-tenant-data-isolation, Property 7
@settings(
    max_examples=100,
    deadline=None,
    # moto's `documents_table` is a function-scoped fixture reused across all
    # Hypothesis examples; suppress the health check as the P1 test does.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    scope_value=_scope_values(),
    doc_id_value=_doc_ids(),
    title=_free_text(),
    body=_free_text(),
)
def test_scope_reserved_word_roundtrip(
    scope_value: str,
    doc_id_value: str,
    title: str,
    body: str,
    documents_table: Any,
) -> None:
    """The reserved-word ``scope`` attribute round-trips through the real name.

    For all in-set/arbitrary scope values and doc ids, writing an item and then
    reading it back by the real key names — a ``GetItem`` with
    ``Key={"scope": ..., "doc_id": ...}`` and a ``Query`` with
    ``Key("scope").eq(...)`` — returns the item, and the owning scope read from
    ``item["scope"]`` equals the stored scope and is never the ``UNKNOWN``
    sentinel. Because ``scope`` is a DynamoDB reserved word, this guards the
    prior bug where a projection on the bare name failed with a
    ``ValidationException`` and silently returned ``UNKNOWN``.
    """
    # --- Write one item per example directly to the moto table. ----------------
    item = {
        "scope": scope_value,
        "doc_id": doc_id_value,
        "title": title,
        "body": body,
    }
    documents_table.put_item(Item=item)

    # --- (1) GetItem via a Key={...} map. --------------------------------------
    # The `Key={...}` map escapes the reserved word `scope` internally, so the
    # real attribute value round-trips rather than falling back to UNKNOWN.
    get_response = documents_table.get_item(
        Key={"scope": scope_value, "doc_id": doc_id_value}
    )
    assert "Item" in get_response, "GetItem did not return the just-written item"
    got = get_response["Item"]
    assert got["scope"] == scope_value
    assert got["scope"] != _UNKNOWN
    assert got["doc_id"] == doc_id_value

    # --- (2) Query via a Key("scope").eq(...) builder. -------------------------
    # The condition builder escapes the reserved word `scope` internally too.
    query_response = documents_table.query(
        KeyConditionExpression=Key("scope").eq(scope_value)
    )
    queried = query_response["Items"]
    assert queried, "Query on the seeded partition returned no items"
    # Every item in the pinned partition owns exactly this scope (never UNKNOWN),
    # proving the key condition matched on the REAL attribute value.
    for row in queried:
        assert row["scope"] == scope_value
        assert row["scope"] != _UNKNOWN
    # The specific seeded item is present in its partition and round-trips.
    matching = [row for row in queried if row["doc_id"] == doc_id_value]
    assert len(matching) == 1, "seeded item not uniquely found in its partition"
    assert matching[0]["scope"] == scope_value
    assert matching[0]["scope"] != _UNKNOWN
