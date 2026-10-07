"""
Tool-side DynamoDB access from interceptor-vended credentials.

The REQUEST interceptor performs ``sts:AssumeRole`` with the inline
``dynamodb:LeadingKeys`` session policy and passes the resulting short-lived,
partition-confined credentials to the tool.

Wire contract: the interceptor propagates the authoritative scope and the vended
credentials as four custom request HEADERS, which the Gateway delivers to the
Lambda inside the CLIENT CONTEXT at
``context.client_context.custom["bedrockAgentCorePropagatedHeaders"]`` (see
``tools/common/credentials_context.py`` for the names and the validation). The
Lambda ``event`` carries only the model-supplied tool arguments, so the tool's
declared ``inputSchema`` is the whole of its contract.

The tool reads the served scope via :func:`served_scope_from_context` and builds a
boto3 DynamoDB resource via :func:`documents_table_from_context`, which maps the
three credential headers onto the boto3 ``Session`` keyword arguments
``aws_access_key_id`` / ``aws_secret_access_key`` / ``aws_session_token`` by name.
Both readers fail closed (raise :class:`ScopedCredentialsError`) on
missing/malformed propagated headers and NEVER fall back to the tool's execution
role or the default credential chain.

Neither reader accepts the ``event``. That is the point: a credential in the body
would be one the MODEL could have written, and removing the parameter makes its
absence a property of the signature rather than a rule to remember.

Those three credential fields are passed to ``Session.resource()`` rather than to
a fresh ``Session(...)``: a per-thread, CREDENTIAL-FREE session is reused as a
model-cache factory, which removed a measured 97.4 ms p50 of per-request service
and resource model parsing without ever sharing a tenant-bound object. See the
"Credential-free session factory" block below for the safety argument and the
thread-scope decision.

Module split: the *header parsing and validation* half of this module lives in
``tools/common/credentials_context.py``
(:func:`propagated_headers`, :func:`served_scope_from_context`,
:func:`session_kwargs_from_context`, :class:`ScopedCredentialsError`, and the
header-name constants). This file remains the *table-builder entry point* and
re-exports the error type and the scope reader so a caller can import either from
one place.

Trust boundary — why the tool does not assume the role itself:
    The obvious alternative is for each tool Lambda to call ``sts:AssumeRole``
    ITSELF and build the session policy from the ``served_scope`` in its event —
    but then its execution role has to hold ``sts:AssumeRole`` on the scoped
    roles, and code running inside the tool can re-assume with a wider (or no)
    session policy and reach the whole table. Moving the assume into the
    interceptor lets ``sts:AssumeRole`` (and every DynamoDB permission) be
    STRIPPED from the tool execution roles. The tool can then ONLY use the
    credentials it was handed — it cannot mint or widen any credential (verified
    by the compromised-tool test, which observes ``AccessDenied`` on
    ``sts:AssumeRole`` from the tool exec role).

Environment variables (tool side):
    DOCUMENTS_TABLE_NAME   Name of the Documents table (boto3 Table lookup).
    (The tool needs no DOCUMENTS_TABLE_ARN and no role ARNs — those belong to the
    interceptor, which does the assume.)

Security:
    This module NEVER logs the vended credentials it reads from the propagated
    request headers.

AWS documentation references:
    - STS session policy — the vended session is the INTERSECTION of the role
      identity policy and the inline session policy:
      https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_assumerole.html
    - STS ``Credentials`` shape (AccessKeyId, SecretAccessKey, SessionToken) —
      the source field names the interceptor propagates as the three credential
      headers (``x-tvm-access-key-id``, ``x-tvm-secret-access-key``,
      ``x-tvm-session-token``):
      https://docs.aws.amazon.com/STS/latest/APIReference/API_Credentials.html
    - boto3/AWS credential field names (``aws_access_key_id``,
      ``aws_secret_access_key``, ``aws_session_token``) that those three header
      values map onto for a session:
      https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/credentials.html
    - DynamoDB ``dynamodb:LeadingKeys`` (partition key; MUST use ``ForAllValues``;
      ``Null`` presence-check hardening):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/specifying-conditions.html
    - Item-level policy example (GetItem/Query/UpdateItem under one
      ``ForAllValues:StringEquals`` ``LeadingKeys`` condition):
      https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_dynamodb_items.html

Functions:
    served_scope_from_context: (re-exported from ``credentials_context``) return
        the authoritative served scope read from the propagated headers.
    documents_table_from_context: Build a DynamoDB Table from the vended
        credentials carried in the propagated headers.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import boto3

# Context parsing/validation half of the tool credentials module (kept in its own
# file so each module stays small). Re-exported below so existing call sites that do
# ``from common.scoped_credentials import ScopedCredentialsError`` /
# ``served_scope_from_context`` keep resolving from this module.
from common.credentials_context import (
    ScopedCredentialsError,
    served_scope_from_context,
    session_kwargs_from_context,
)

__all__ = [
    "ScopedCredentialsError",
    "served_scope_from_context",
    "documents_table_from_context",
    "reset_factory_session",
]


# ---------------------------------------------------------------------------
# Credential-free session factory (latency optimization)
# ---------------------------------------------------------------------------
# Measured: building a fresh ``boto3.Session`` and calling ``.resource()`` on it
# cost 97.4 ms at p50, versus 2.08 ms for a plain ``boto3.client()`` on the same
# containers. The 47x gap is the SERVICE and RESOURCE MODEL LOAD — static JSON
# shipped inside boto3 — not anything to do with the credentials being fresh. A
# new Session brings a new loader, so every request re-parsed those models.
#
# The fix keeps a session purely as a MODEL-CACHE FACTORY and passes the vended
# credentials at resource-creation time instead. The session is constructed with
# NO credentials, so it carries no tenant identity; the loader and cached service
# and resource models it holds are identical for every tenant.
#
# The rule: an object may be shared across requests IF AND ONLY IF it carries no
# tenant identity. This session qualifies. The DynamoDB resource built from the
# vended credentials does NOT, and is therefore still created per request and
# never cached or reused across requests.
#
# THREAD SCOPE — one session per THREAD, not one per container. boto3 documents
# that "Session objects are not thread safe and should not be shared across
# threads and processes" and recommends one Session per thread
# (https://docs.aws.amazon.com/boto3/latest/guide/session.html); the same applies
# to resources (https://docs.aws.amazon.com/boto3/latest/guide/resources.html).
# Rather than assume that concurrent ``resource()`` creation from one shared
# session is benign, the session is held in thread-local storage. Consequence:
# the model cache is per thread, so the FIRST request on each thread pays the
# model load. A Lambda container normally serves one request at a time, so in
# production this is effectively per container; under a multi-threaded caller the
# saving is proportionally smaller.
_thread_local = threading.local()


def _factory_session() -> Any:
    """Return this thread's credential-free boto3 session, creating it once.

    The session is built with NO credentials and is used ONLY as a factory, so
    that the service and resource models it caches are reused across requests
    instead of being re-parsed on each one. It is thread-local because boto3
    documents ``Session`` as not thread safe.

    Returns:
        The calling thread's credential-free boto3 ``Session``.
    """
    session = getattr(_thread_local, "factory_session", None)
    if session is None:
        session = boto3.Session()
        _thread_local.factory_session = session
    return session


def reset_factory_session() -> None:
    """Drop the calling thread's factory session so the next call rebuilds it.

    TEST SEAM ONLY. Production never calls this: the session is meant to live for
    the life of the thread. Tests that monkeypatch ``boto3.Session`` need the
    thread-local cleared between cases, otherwise the first test's fake would be
    reused by every later test.
    """
    if hasattr(_thread_local, "factory_session"):
        del _thread_local.factory_session


def _require_env(name: str) -> str:
    """Return a required environment variable value, or raise if missing/empty.

    Args:
        name: The environment variable name to read.

    Returns:
        The stripped value of the environment variable.

    Raises:
        ScopedCredentialsError: If the variable is unset or empty. Raised (rather
            than returned) so the tool handler surfaces a generic error and never
            falls back to an unscoped path.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise ScopedCredentialsError(
            f"{name} environment variable is not set. The Lambda must be "
            f"configured with the Documents table name."
        )
    return value


def documents_table_from_context(lambda_context: Any) -> Any:
    """Build a Documents table bound to the propagated credential headers.

    Reads the credentials from
    ``lambda_context.client_context.custom["bedrockAgentCorePropagatedHeaders"]``
    — the headers the REQUEST interceptor propagated — and maps the three
    credential headers onto the boto3 ``Session`` keyword arguments by name:
    ``x-tvm-access-key-id``→``aws_access_key_id``,
    ``x-tvm-secret-access-key``→``aws_secret_access_key``,
    ``x-tvm-session-token``→``aws_session_token``. The Documents ``Table`` is bound
    by name via ``DOCUMENTS_TABLE_NAME``.

    All DynamoDB access through the returned table therefore uses the scoped,
    partition-confined credentials the interceptor vended — the tool performs NO
    ``AssumeRole`` and uses NONE of its own execution-role permissions (it holds
    no DynamoDB grant). On missing/malformed propagated headers the tool fails
    closed and NEVER falls back to its execution role or the default credential
    chain.

    This reader deliberately does NOT accept the Lambda ``event``. A reader that
    could still find a credential in the body would be a path to a credential the
    MODEL could have written; removing the parameter makes the absence of that
    path a property of the signature rather than a rule to remember.

    Args:
        lambda_context: The Lambda context object, carrying the propagated
            headers in its client context.

    Returns:
        A boto3 DynamoDB ``Table`` resource confined to the vended scope.

    Raises:
        ScopedCredentialsError: If the propagated credential headers are missing
            or malformed, or if ``DOCUMENTS_TABLE_NAME`` is unset. The caller
            surfaces a generic error and does NOT fall back to any other
            credential source.
    """
    session_kwargs = session_kwargs_from_context(lambda_context)

    table_name = _require_env("DOCUMENTS_TABLE_NAME")

    # Build the DynamoDB resource from the thread's credential-free factory
    # session, passing the VENDED credentials at resource-creation time. The
    # session supplies only the cached (tenant-independent) service and resource
    # models; the credentials — and therefore the tenant scope of every call made
    # through the returned Table — come from this request's propagated headers
    # alone.
    #
    # Note on fail-closed: `session_kwargs_from_context` above has already raised
    # for missing or malformed headers, so this line is unreachable without a
    # complete set of vended credentials. That ordering is load-bearing. If
    # `session_kwargs` could ever arrive empty here, the factory session would
    # silently fall back to the DEFAULT CREDENTIAL CHAIN (the tool's execution
    # role) instead of failing. The read would still be denied, because that role
    # holds no DynamoDB grant, but it would surface as an opaque AccessDenied
    # rather than a clear contract violation.
    return (
        _factory_session()
        .resource("dynamodb", **session_kwargs)
        .Table(table_name)
    )

