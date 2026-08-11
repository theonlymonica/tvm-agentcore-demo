"""Shared pytest fixtures for the scoped-credentials test suite.

These fixtures stand up the offline AWS test environment the tests reuse:

* the session-policy builder unit tests (which need the scoped-env vars);
* the read-path tool tests (composite-key ``GetItem``, ``Query``-not-``Scan``);
* the reply tool tests (composite-key ``UpdateItem``, ``list_append``);
* the degraded IAM-fallback fixture.

The DynamoDB table created here matches the scope-partitioned schema:
partition key ``scope`` (String) and sort key ``doc_id`` (String),
billed ``PAY_PER_REQUEST``.

boto3 / moto are imported lazily inside the fixtures that need them so that
tests which need no AWS mocking (for example the pure session-policy JSON
assertions, or the interceptor tests) can still be collected and run without
the full AWS stack present.

Nothing here is a deployed artifact — these fixtures exist only for tests.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Per-container / per-thread boto3 object caches (latency optimization)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_boto3_caches() -> Iterator[None]:
    """Clear the shared STS client and the thread-local factory session.

    Two boto3 objects are now built once and reused rather than per request: the
    interceptor's STS client (``interceptor.scoped_credentials``) and the tool's
    credential-free factory session (``common.scoped_credentials``). Both are
    deliberately tenant-agnostic, so reuse is safe in production — but in the test
    suite it would leak one test's monkeypatched fake into every later test, since
    the cached object outlives the ``monkeypatch`` that produced it.

    Autouse, and clears both BEFORE and AFTER each test, so every test starts from
    a cold cache and leaves nothing behind. This is the only change the
    optimization required in the existing test fixtures; no assertion changed.

    Yields:
        None.
    """
    from common import scoped_credentials as tool_scoped_credentials
    from interceptor import scoped_credentials as interceptor_scoped_credentials

    interceptor_scoped_credentials.reset_sts_client()
    tool_scoped_credentials.reset_factory_session()
    yield
    interceptor_scoped_credentials.reset_sts_client()
    tool_scoped_credentials.reset_factory_session()

# ---------------------------------------------------------------------------
# Shared constants (mirror the scope-partitioned data model)
# ---------------------------------------------------------------------------

AWS_REGION = "us-east-1"
AWS_ACCOUNT_ID = "123456789012"

DOCUMENTS_TABLE_NAME = "DocumentsTable"
DOCUMENTS_TABLE_ARN = (
    f"arn:aws:dynamodb:{AWS_REGION}:{AWS_ACCOUNT_ID}:table/{DOCUMENTS_TABLE_NAME}"
)
DOCUMENTS_ACCESS_ROLE_ARN = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/DocumentsAccessRole"
DOCUMENTS_WRITE_ROLE_ARN = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/DocumentsWriteRole"

# The served scope for a session and the foreign scopes the injection targets.
SERVED_SCOPE = "payments-core"
FOREIGN_SCOPES = ("billing-internal", "infra-secrets", "hr-data")
KNOWN_SCOPES = (SERVED_SCOPE, *FOREIGN_SCOPES)


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy AWS credentials and a region so moto never touches real AWS.

    Args:
        monkeypatch: pytest monkeypatch fixture used to set env vars for the
            duration of a single test.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", AWS_REGION)
    monkeypatch.setenv("AWS_REGION", AWS_REGION)


@pytest.fixture
def scoped_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set the DOCUMENTS_* env vars the scoped-credentials helper reads.

    These mirror the environment the CDK wiring injects into the tool Lambdas.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        The mapping of environment variable names to the values that were set,
        for convenient assertion in tests.
    """
    env = {
        "DOCUMENTS_TABLE_NAME": DOCUMENTS_TABLE_NAME,
        "DOCUMENTS_TABLE_ARN": DOCUMENTS_TABLE_ARN,
        "DOCUMENTS_ACCESS_ROLE_ARN": DOCUMENTS_ACCESS_ROLE_ARN,
        "DOCUMENTS_WRITE_ROLE_ARN": DOCUMENTS_WRITE_ROLE_ARN,
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


# ---------------------------------------------------------------------------
# moto / boto3 fixtures (lazy imports so non-AWS tests need not have them)
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_aws(aws_credentials: None) -> Iterator[None]:
    """Activate moto's unified ``mock_aws`` for the duration of a test.

    Args:
        aws_credentials: ensures dummy credentials/region are set first.

    Yields:
        None. All boto3 calls made inside the ``with`` block are served by moto.
    """
    from moto import mock_aws  # lazy import — see module docstring

    with mock_aws():
        yield


@pytest.fixture
def dynamodb_resource(mocked_aws: None) -> Any:
    """Return a moto-backed boto3 DynamoDB *resource* in the test region.

    Args:
        mocked_aws: ensures the moto mock is active.

    Returns:
        A boto3 ``dynamodb`` service resource.
    """
    import boto3

    return boto3.resource("dynamodb", region_name=AWS_REGION)


@pytest.fixture
def documents_table(dynamodb_resource: Any) -> Any:
    """Create the scope-partitioned Documents table in moto and return it.

    Schema: partition key ``scope`` (S), sort key ``doc_id`` (S), billing
    ``PAY_PER_REQUEST``.

    Args:
        dynamodb_resource: moto-backed DynamoDB resource.

    Returns:
        A boto3 ``Table`` resource ready for GetItem/Query/UpdateItem.
    """
    table = dynamodb_resource.create_table(
        TableName=DOCUMENTS_TABLE_NAME,
        KeySchema=[
            {"AttributeName": "scope", "KeyType": "HASH"},
            {"AttributeName": "doc_id", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "scope", "AttributeType": "S"},
            {"AttributeName": "doc_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    return table


@pytest.fixture
def sts_client(mocked_aws: None) -> Any:
    """Return a moto-backed boto3 STS client in the test region.

    Args:
        mocked_aws: ensures the moto mock is active.

    Returns:
        A boto3 ``sts`` client.
    """
    import boto3

    return boto3.client("sts", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def make_document(
    served_scope: str,
    document_id: str,
    *,
    title: str | None = None,
    body: str | None = None,
) -> dict[str, Any]:
    """Build a composite-key document item in the scope-partitioned shape.

    The emitted item attributes use the article's wire-contract names: the
    owning scope is written under ``scope`` (partition key) and the identifier
    under ``doc_id`` (sort key). The parameter names are kept as
    ``served_scope`` / ``document_id`` so existing positional and keyword call
    sites continue to work.

    ``conversation`` is intentionally absent (it is absent on seed and is
    initialized by the reply tool via ``if_not_exists``).

    Args:
        served_scope: The owning scope value (emitted as the ``scope`` key).
        document_id: The document identifier (emitted as the ``doc_id`` key).
        title: Optional title; defaults to a value derived from the id.
        body: Optional body; defaults to a value derived from the id.

    Returns:
        A dict suitable for ``Table.put_item(Item=...)``.
    """
    return {
        "scope": served_scope,
        "doc_id": document_id,
        "title": title if title is not None else f"Title for {document_id}",
        "body": body if body is not None else f"Body for {document_id}",
    }


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """A small multi-scope document set for reuse by read/search/reply tests.

    One served-scope document plus one document in each foreign scope, so tests
    can assert same-scope liveness and cross-scope absence.

    Returns:
        A list of composite-key document items.
    """
    docs = [make_document(SERVED_SCOPE, "PAY-001", title="Release status")]
    docs.append(make_document("billing-internal", "BIL-002", title="Refund ledger"))
    docs.append(make_document("infra-secrets", "INF-004", title="Refund pipeline keys"))
    docs.append(make_document("hr-data", "HR-003", title="Refund approver list"))
    return docs


@pytest.fixture
def put_documents(documents_table: Any) -> Callable[[list[dict[str, Any]]], None]:
    """Return a helper that writes document items into the moto table.

    Args:
        documents_table: the moto-backed Documents table.

    Returns:
        A callable taking a list of item dicts and writing each via ``put_item``.
    """

    def _put(items: list[dict[str, Any]]) -> None:
        for item in items:
            documents_table.put_item(Item=item)

    return _put
