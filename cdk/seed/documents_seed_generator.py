"""
Parametric Documents_Table seed generator.

Produces a list of document dicts for the Session-Aware Document Guard demo.
The generator logic (ID assignment, scope assignment, title/body construction)
is independent of ``count``; only the number of non-matching noise documents
scales with the count parameter.

All seed text is in English. No real company names appear anywhere.

Functions:
    generate_documents: Build the full document list for seeding.
"""

from __future__ import annotations

from typing import Dict, List

from .seed_data_pools import (
    OTHER_SCOPE_BODY_POOL,
    OTHER_SCOPE_TITLE_POOL,
    PAY_001_BODY,
    PAYMENTS_CORE_BODY_POOL,
    PAYMENTS_CORE_TITLE_POOL,
    REFUND_TITLE_POOL,
    REFUND_TRAP_BODIES,
    SENSITIVE_DATA_MARKERS,
)
from .seed_invariants import verify_seed_invariants

# ---------------------------------------------------------------------------
# Scope set
# ---------------------------------------------------------------------------
# Exactly one served/attacker scope plus six other fictional scopes.
# No real company names. Every generated document scope is drawn from this set.

SERVED_SCOPE = "payments-core"

OTHER_SCOPES = [
    "billing-internal",
    "hr-data",
    "infra-secrets",
    "analytics-warehouse",
    "partner-integrations",
    "legal-contracts",
]

ALL_SCOPES = [SERVED_SCOPE] + OTHER_SCOPES

# Scope-prefix mapping for deterministic IDs.
# IDs are of the form PREFIX-NNN where NNN is a zero-padded sequential number.
SCOPE_PREFIX_MAP: Dict[str, str] = {
    "payments-core": "PAY",
    "billing-internal": "BIL",
    "hr-data": "HR",
    "infra-secrets": "INF",
    "analytics-warehouse": "ANA",
    "partner-integrations": "PAR",
    "legal-contracts": "LEG",
}


# ---------------------------------------------------------------------------
# Deterministic ID generation
# ---------------------------------------------------------------------------

def make_document_id(scope: str, sequence_number: int) -> str:
    """Build a deterministic, scope-prefixed document ID.

    IDs follow the pattern PREFIX-NNN (e.g. PAY-001, BIL-002, HR-003).
    The sequence number is global across all documents so that re-seeding
    always produces the same keys regardless of scope ordering.

    Args:
        scope: The owning scope (must be in ALL_SCOPES).
        sequence_number: The 1-based global sequence number.

    Returns:
        A string like "PAY-001" or "INF-042".

    Raises:
        ValueError: If the scope is not in the declared scope set.
    """
    prefix = SCOPE_PREFIX_MAP.get(scope)
    if prefix is None:
        raise ValueError(
            f"Scope '{scope}' is not in the declared scope set: {ALL_SCOPES}"
        )
    return f"{prefix}-{sequence_number:03d}"


# ---------------------------------------------------------------------------
# Scope assignment (round-robin over ALL_SCOPES)
# ---------------------------------------------------------------------------

def assign_scope(index: int) -> str:
    """Assign a scope to a document by its 0-based generation index.

    Uses a round-robin assignment over ALL_SCOPES so that the distribution
    is even and deterministic. Index 0 always maps to payments-core.

    Args:
        index: The 0-based index of the document in the generation sequence.

    Returns:
        A scope string from ALL_SCOPES.
    """
    return ALL_SCOPES[index % len(ALL_SCOPES)]


# ---------------------------------------------------------------------------
# Title and body selection helpers
# ---------------------------------------------------------------------------

def select_payments_core_title(index: int) -> str:
    """Select a title for a payments-core document from the safe pool.

    The PAYMENTS_CORE_TITLE_POOL is guaranteed never to contain the
    substring 'refund' in any casing. Selection is deterministic based
    on the document index.

    Args:
        index: A deterministic index used for selection.

    Returns:
        A title string that does NOT contain 'refund'.
    """
    return PAYMENTS_CORE_TITLE_POOL[index % len(PAYMENTS_CORE_TITLE_POOL)]


def select_refund_title(index: int) -> str:
    """Select a refund-bearing title for a non-payments-core document.

    These titles contain the keyword 'refund' and are ONLY assigned to
    non-payments-core scopes.

    Args:
        index: A deterministic index used for selection.

    Returns:
        A title string that contains 'refund'.
    """
    return REFUND_TITLE_POOL[index % len(REFUND_TITLE_POOL)]


def select_other_scope_title(index: int) -> str:
    """Select a generic (non-refund) title for a non-payments-core document.

    Args:
        index: A deterministic index used for selection.

    Returns:
        A title string that does NOT contain 'refund'.
    """
    return OTHER_SCOPE_TITLE_POOL[index % len(OTHER_SCOPE_TITLE_POOL)]


def select_payments_core_body(index: int) -> str:
    """Select a body for a payments-core document.

    Args:
        index: A deterministic index used for selection.

    Returns:
        A body string with generic payments-related content.
    """
    return PAYMENTS_CORE_BODY_POOL[index % len(PAYMENTS_CORE_BODY_POOL)]


def select_other_scope_body(index: int) -> str:
    """Select a generic body for a non-payments-core noise document.

    Args:
        index: A deterministic index used for selection.

    Returns:
        A body string with generic operational content.
    """
    return OTHER_SCOPE_BODY_POOL[index % len(OTHER_SCOPE_BODY_POOL)]


# ---------------------------------------------------------------------------
# Core document builder
# ---------------------------------------------------------------------------

def build_document(
    scope: str,
    sequence_number: int,
    title: str,
    body: str,
) -> Dict[str, str]:
    """Build a single document dict ready for DynamoDB seeding.

    The emitted item uses the composite-key shape required by the
    scope-partitioned Documents_Table: ``scope`` is the partition
    key and ``doc_id`` is the sort key. The optional ``conversation``
    attribute is intentionally ABSENT on seed; it is initialized later by
    the reply tool via ``if_not_exists`` on first append.

    Args:
        scope: The owning scope identifier (becomes ``scope``, the PK).
        sequence_number: The 1-based global sequence number for ID generation.
        title: The document title.
        body: The document body text.

    Returns:
        A dict with keys: scope (PK), doc_id (SK), title, body.
    """
    return {
        "scope": scope,
        "doc_id": make_document_id(scope, sequence_number),
        "title": title,
        "body": body,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_documents(count: int = 60) -> List[Dict[str, str]]:
    """Generate the full list of seed documents for the Documents_Table.

    The generator builds the document list in a fixed structure:

    1. Slot 1 (PAY-001): The starting document owned by payments-core,
       with the unmarked Injected_Instruction in its body.
    2. Slots 2-4 (BIL-002, HR-003, INF-004): The refund trap documents
       owned by non-payments-core scopes, with refund-bearing titles
       and invented sensitive data in their bodies.
    3. Slots 5-6: Additional payments-core documents (non-matching) so
       the served scope has multiple own documents.
    4. Remaining slots: Non-payments-core noise on assorted topics,
       filling to the requested count.

    The generator logic (ID assignment, scope assignment, title/body
    construction, and the post-generation verification) is independent of
    ``count``. Only the number of non-matching noise documents scales with
    the count parameter.

    The returned list is suitable for writing to DynamoDB via BatchWriteItem
    (using boto3 Table.batch_writer).

    Args:
        count: Total number of documents to generate (default 60).
            Must be at least 10 to satisfy minimum distribution constraints.

    Returns:
        A list of document dicts, each with keys:
        scope (partition key), doc_id (sort key), title, body.
        The optional ``conversation`` attribute is absent on seed.

    Raises:
        ValueError: If count is below the minimum required for the invariants.
    """
    if count < 10:
        raise ValueError(
            f"count must be at least 10 to satisfy seed invariants, got {count}"
        )

    documents: List[Dict[str, str]] = []
    seq = 0  # will be incremented to 1-based before each use

    # ------------------------------------------------------------------
    # Slot 1: PAY-001 — the starting document
    # ------------------------------------------------------------------
    seq += 1  # seq = 1
    documents.append(build_document(
        scope=SERVED_SCOPE,
        sequence_number=seq,
        title=select_payments_core_title(0),
        body=PAY_001_BODY,
    ))

    # ------------------------------------------------------------------
    # Slots 2-4: Refund trap documents
    # At least 2 non-payments-core, refund-titled, with sensitive bodies.
    # We emit all 3 from REFUND_TRAP_BODIES for richer demo content.
    # ------------------------------------------------------------------
    for trap in REFUND_TRAP_BODIES:
        seq += 1
        documents.append(build_document(
            scope=trap["served_scope"],
            sequence_number=seq,
            title=trap["title"],
            body=trap["body"],
        ))

    # ------------------------------------------------------------------
    # Slots 5-6: Additional payments-core non-matching documents
    # (at least 2 payments-core docs that do NOT match refund)
    # ------------------------------------------------------------------
    # We already used title index 0 for PAY-001, start from 1.
    payments_core_extra_count = 2
    for i in range(payments_core_extra_count):
        seq += 1
        title_idx = i + 1  # skip index 0 already used by PAY-001
        body_idx = i  # body pool is independent from PAY-001's custom body
        documents.append(build_document(
            scope=SERVED_SCOPE,
            sequence_number=seq,
            title=select_payments_core_title(title_idx),
            body=select_payments_core_body(body_idx),
        ))

    # ------------------------------------------------------------------
    # Remaining slots: Non-payments-core noise documents
    # Assorted topics with NO "refund" in title. Round-robin across the
    # OTHER_SCOPES so distribution is even.
    # ------------------------------------------------------------------
    noise_count = count - seq  # fill to the requested total
    other_title_idx = 0
    other_body_idx = 0

    for i in range(noise_count):
        seq += 1
        # Round-robin across OTHER_SCOPES (never payments-core)
        scope = OTHER_SCOPES[i % len(OTHER_SCOPES)]
        title = select_other_scope_title(other_title_idx)
        body = select_other_scope_body(other_body_idx)
        other_title_idx += 1
        other_body_idx += 1
        documents.append(build_document(
            scope=scope,
            sequence_number=seq,
            title=title,
            body=body,
        ))

    # ------------------------------------------------------------------
    # Post-generation verification
    # Hard-fail if any invariant is violated — aborts CDK synth/deploy.
    # ------------------------------------------------------------------
    verify_seed_invariants(
        documents,
        served_scope=SERVED_SCOPE,
        all_scopes=ALL_SCOPES,
        sensitive_data_markers=SENSITIVE_DATA_MARKERS,
    )

    return documents


# ---------------------------------------------------------------------------
# Post-generation verification
# ---------------------------------------------------------------------------
# The verification logic lives in ``seed_invariants.verify_seed_invariants``
# (extracted to keep this module under the code-modularity line limit).
