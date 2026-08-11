"""
Post-generation seed invariants for the Documents_Table generator.

This module holds the hard-fail verification that runs at synth time after the
seed document list is generated. It is kept separate from
``documents_seed_generator`` so the generator stays under the code-modularity
line limit and the verification logic is independently testable.

The verifier is deliberately dependency-light: the scope constants and the
sensitive-data markers are passed in as parameters so this module never imports
the generator (avoiding a circular import).

Functions:
    verify_seed_invariants: Hard-fail check of the composite-key seed invariants.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def verify_seed_invariants(
    documents: List[Dict[str, str]],
    served_scope: str,
    all_scopes: Sequence[str],
    sensitive_data_markers: Sequence[str],
) -> None:
    """Run hard-fail verification on the generated document list.

    Checks the composite-key invariants and raises ValueError on the first
    violation:

    1. No served-scope title contains 'refund' (case-insensitive).
    2. At least two non-served-scope titles contain 'refund'.
    3. No served-scope body contains any sensitive-data marker string.
    4. Every document ``scope`` is in ``all_scopes``.
    5. Every item carries both composite-key attributes (``scope`` +
       ``doc_id``) and no optional ``conversation`` attribute on seed.
    6. Every ``(scope, doc_id)`` composite key is unique, so
       ``PutItem`` re-seeding overwrites in place and never collides
       (idempotency).

    Args:
        documents: The full list of generated seed documents.
        served_scope: The served/attacker scope identifier (e.g. ``payments-core``).
        all_scopes: The complete set of declared scopes.
        sensitive_data_markers: Substrings that must NOT appear in any
            served-scope body.

    Raises:
        ValueError: With a clear message identifying which invariant failed.
    """
    # Invariant 1: No served-scope title may contain "refund"
    for doc in documents:
        if doc["scope"] == served_scope and "refund" in doc["title"].lower():
            raise ValueError(
                f"Invariant violated: served-scope document "
                f"'{doc['doc_id']}' has title containing 'refund': "
                f"'{doc['title']}'"
            )

    # Invariant 2: At least two non-served-scope titles contain "refund"
    refund_count = sum(
        1
        for doc in documents
        if doc["scope"] != served_scope and "refund" in doc["title"].lower()
    )
    if refund_count < 2:
        raise ValueError(
            f"Invariant violated: expected at least 2 non-served-scope "
            f"documents with 'refund' in title, found {refund_count}"
        )

    # Invariant 3: No served-scope body contains sensitive-data markers
    for doc in documents:
        if doc["scope"] == served_scope:
            for marker in sensitive_data_markers:
                if marker in doc["body"]:
                    raise ValueError(
                        f"Invariant violated: served-scope document "
                        f"'{doc['doc_id']}' body contains "
                        f"sensitive-data marker: '{marker}'"
                    )

    # Invariant 4: Every item scope must be in the declared set
    for doc in documents:
        if doc["scope"] not in all_scopes:
            raise ValueError(
                f"Invariant violated: document '{doc['doc_id']}' has "
                f"scope '{doc['scope']}' which is not in the "
                f"declared scope set: {list(all_scopes)}"
            )

    # Invariant 5: Every item carries both composite-key attributes and no
    # 'conversation' attribute on seed (reply initializes it later).
    for doc in documents:
        if not doc.get("scope") or not doc.get("doc_id"):
            raise ValueError(
                f"Invariant violated: item {doc!r} is missing a composite-key "
                f"attribute (both scope and doc_id are required)"
            )
        if "conversation" in doc:
            raise ValueError(
                f"Invariant violated: seed item '{doc['doc_id']}' must not "
                f"carry a 'conversation' attribute (it is initialized later by "
                f"the reply tool)"
            )

    # Invariant 6: Composite keys must be unique so PutItem re-seeding is
    # idempotent (same key overwrites in place, never collides/duplicates).
    seen_keys: set[tuple[str, str]] = set()
    for doc in documents:
        key = (doc["scope"], doc["doc_id"])
        if key in seen_keys:
            raise ValueError(
                f"Invariant violated: duplicate composite key "
                f"scope='{key[0]}', doc_id='{key[1]}' — re-seeding "
                f"would not be idempotent"
            )
        seen_keys.add(key)
