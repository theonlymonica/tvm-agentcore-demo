"""
Interceptor-side scoped-credential vending.

The REQUEST interceptor — NOT the tool Lambda — performs ``sts:AssumeRole`` with
the inline ``dynamodb:LeadingKeys`` session policy and passes the resulting
short-lived credentials to the tool. The tool execution roles hold NO
``sts:AssumeRole`` and NO DynamoDB permission, so a compromised dependency inside
a tool cannot mint any credential or widen its own access — it can only use the
scoped credentials it was handed.

Why this module is duplicated from ``tools/common/scoped_credentials.py``:
    The interceptor Lambda is bundled from the ``interceptor/`` package ONLY
    (``cdk/toxic_flow_stack.py`` excludes ``tools``, ``cdk``, ``shared`` from the
    interceptor asset), so it cannot import ``common.scoped_credentials``. The
    ``build_session_policy`` shape and the READ/WRITE action sets are therefore
    re-declared here. They MUST stay byte-for-byte equivalent to the tool-side
    copy.

Credential channel (Lambda targets):
    A Lambda target receives only the ``inputSchema`` properties as its ``event``
    (no headers, no context credentials). ``params.arguments`` is the only
    interceptor->Lambda-target channel, and UNDECLARED fields survive intact at
    credential payload size. So the credentials ride as UNDECLARED
    ``params.arguments`` fields (kept OUT of every tool ``inputSchema`` so
    credential-shaped properties are never advertised to the model).

    DOCUMENTED DRAWBACK (inherent to the AWS reference "Design 2" for a Lambda
    target): credentials transit ``params.arguments``, which the gateway's vended
    APPLICATION_LOGS ``requestBody`` would capture verbatim IF that delivery were
    enabled — confirmed by live observation. Mitigations: a leaked credential is
    unusable after the 60-second session-policy window
    (``_SESSION_POLICY_TTL_SECONDS`` below), which is the real bound — the 900s
    ``DurationSeconds`` is only the STS session floor, not the usable window (IAM
    was observed denying a call at 98s with ~13 minutes of session left). And the
    stack MUST NOT enable APPLICATION_LOGS request-body delivery (see
    ``cdk/gateway_wiring.py``). The deployed stack delivers only TRACES (which
    omit arguments), and that is enforced mechanically rather than by convention:
    ``tests/test_synth_log_delivery.py`` fails the suite if any delivery source
    in the synthesized template declares a logType other than TRACES. Residual
    gap: a delivery created outside this template (in the console), or a
    same-name overwrite, is beyond what a synth-time check can see.

Security:
    This module NEVER logs the vended credentials or the Authorization header.

AWS documentation references (verified against the AWS documentation):
    - STS ``AssumeRole`` — inline session policy passed as ``Policy``; the session
      is the INTERSECTION of the role identity policy and the session policy;
      ``DurationSeconds`` / ``RoleSessionName`` are standard request parameters:
      https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_assumerole.html
    - STS ``Credentials`` response shape (AccessKeyId, SecretAccessKey,
      SessionToken, Expiration):
      https://docs.aws.amazon.com/STS/latest/APIReference/API_Credentials.html
    - DynamoDB ``dynamodb:LeadingKeys`` (partition key; MUST use ``ForAllValues``;
      ``Null`` presence check hardening):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/specifying-conditions.html
    - Lambda target event = ``inputSchema`` properties only (no headers):
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-lambda.html

Shared STS client (latency optimization):
    The STS client used to call ``AssumeRole`` is built ONCE per container and
    reused, instead of once per request. This is safe and changes no security
    property, because that client authenticates as the interceptor's OWN
    EXECUTION ROLE — it is the thing that CALLS ``AssumeRole`` in order to obtain
    tenant credentials, and it never holds any. Nothing about it varies per
    tenant, so sharing it places no tenant state in the process.

    The rule this follows: an object may be shared across requests IF AND ONLY IF
    it carries no tenant identity. The vended credentials, and the DynamoDB
    client built from them, DO carry tenant identity and are therefore still
    created per request and never cached.

    Sharing a CLIENT specifically is what boto3 documents as safe: "Unlike
    Resources and Sessions, clients are generally thread-safe", and the
    recommended pattern is to create the client once and hand it to workers
    (https://docs.aws.amazon.com/boto3/latest/guide/clients.html). The same page
    cautions that calling ``boto3.client()`` *inside* a concurrent context can
    cause response-ordering or SSL-module failures, which is why construction
    here is serialized under a lock and happens exactly once.

    Construction is LAZY rather than at import time so that importing this module
    never requires resolvable credentials (unit tests import it without AWS
    configuration), and so the cost lands on the first request the container
    serves rather than on module import.

Functions:
    build_session_policy: Build the inline session policy JSON string, pinned in
        space (LeadingKeys) AND time (DateLessThan/aws:CurrentTime). Pure.
    vend_scoped_credentials: AssumeRole (DurationSeconds=900) with a ``scope``
        session tag and the inline session policy; returns temp creds.
    build_tenant_context: Assemble the `context` wire-contract object.
    reset_sts_client: Drop the shared STS client (test seam only).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

# ---------------------------------------------------------------------------
# Action sets per capability. Every action ALWAYS carries a
# partition key, which is what lets the LeadingKeys condition confine it.
# NEVER add a keyless action (e.g. dynamodb:Scan) — see the footgun note in the
# tool-side copy. Kept byte-for-byte equivalent to
# tools/common/scoped_credentials.py READ_ACTIONS / WRITE_ACTIONS.
# ---------------------------------------------------------------------------
READ_ACTIONS: list[str] = ["dynamodb:GetItem", "dynamodb:Query"]
WRITE_ACTIONS: list[str] = ["dynamodb:UpdateItem"]

# ---------------------------------------------------------------------------
# Session tagging — the SECOND GATE.
# ---------------------------------------------------------------------------
# Without it, the scoped roles' identity policies would grant their actions on
# the WHOLE table and separation would live ONLY in the inline session policy
# below; any AssumeRole that omitted that policy would yield full cross-tenant
# read/write. The roles therefore carry their own ABAC condition
# (cdk/documents_roles.py:_scope_tag_conditions) requiring
# `dynamodb:LeadingKeys` to match `${aws:PrincipalTag/scope}`, and their trust
# policies require the tag to be present at all
# (cdk/lambda_iam.py:_REQUIRE_SCOPE_TAG_CONDITION). So this module MUST pass the
# tag on every assume: an untagged assume is rejected by STS, and a mis-tagged
# one is denied at the data plane. Both outcomes fail CLOSED.
#
# Tag semantics verified against AWS docs:
#   - Tags ride as a list of {Key, Value} on AssumeRole; up to 50 tags, key <=128
#     chars, value <=256 chars; keys are case-insensitive for uniqueness:
#     https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
#   - The tag lands in the request context of subsequent calls as
#     `aws:PrincipalTag/<key>`:
#     https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html
#   - TransitiveTagKeys is deliberately NOT set: the vended session never chains
#     into another role, so the tag must not survive a further assume.
#: Session-tag key carrying the authoritative served scope. MUST stay equal to
#: ``SCOPE_TAG_KEY`` in ``cdk/documents_roles.py`` (the two live in different
#: bundles — the interceptor asset excludes ``cdk/`` — so they cannot share one
#: constant; ``tests/test_scope_tag_abac.py`` pins the equality).
SCOPE_TAG_KEY = "scope"

#: IAM wildcard characters. The identity-policy condition must use
#: ``ForAllValues:StringLike`` (AWS requires ``StringLike`` when a multivalued
#: context key such as ``dynamodb:LeadingKeys`` is compared against a policy
#: VARIABLE), and ``StringLike`` treats ``*`` and ``?`` as wildcards. A tag value
#: containing either would therefore WIDEN the condition it is supposed to
#: confine — ``scope=*`` would match every partition. Scope values are already
#: allowlisted upstream (``interceptor/jwt_claims.py`` intersects
#: ``cognito:groups`` with the known-scope set), but that set is operator-supplied
#: via ``KNOWN_SCOPE_GROUPS``, so the wildcard rejection is enforced HERE, at the
#: point where the value becomes a policy comparand, rather than trusted upstream.
_TAG_VALUE_FORBIDDEN_CHARS = ("*", "?")

#: STS session-tag value length ceiling (AWS: values can't exceed 256 characters).
_TAG_VALUE_MAX_LEN = 256


class ScopeTagError(RuntimeError):
    """A served scope cannot be safely expressed as a session tag.

    Subclasses ``RuntimeError`` deliberately: the REQUEST interceptor handler
    already catches ``RuntimeError`` on the vend path and fails closed with a
    generic, detail-free short-circuit error, so a rejected tag value takes the
    existing fail-closed route and hands the tool no credential.
    """


def _scope_tag_or_raise(served_scope: str) -> list[dict[str, str]]:
    """Build the ``Tags`` argument for ``AssumeRole``, or fail closed.

    Rejects any value that could not be safely compared by the identity policy's
    ``ForAllValues:StringLike`` condition — empty, over the 256-character STS
    ceiling, or containing an IAM wildcard (``*`` / ``?``) that would widen the
    comparison to other partitions.

    Args:
        served_scope: The authoritative, JWT-derived scope.

    Returns:
        A one-element ``Tags`` list: ``[{"Key": "scope", "Value": served_scope}]``.

    Raises:
        ScopeTagError: If the value is empty, too long, or wildcard-bearing. The
            caller never reaches ``AssumeRole``, so no credential is minted.
    """
    if not served_scope:
        raise ScopeTagError("served_scope is empty; refusing to vend an untagged session")
    if len(served_scope) > _TAG_VALUE_MAX_LEN:
        raise ScopeTagError("served_scope exceeds the STS session-tag value limit")
    if any(char in served_scope for char in _TAG_VALUE_FORBIDDEN_CHARS):
        # NOTE: the offending value is NOT logged or echoed — it is identity-derived.
        raise ScopeTagError("served_scope contains an IAM wildcard character")
    return [{"Key": SCOPE_TAG_KEY, "Value": served_scope}]

# STS temporary-credential lifetime for a vended scoped session. Set explicitly
# to the 900-second STS minimum: minimizes the lifetime of the
# credentials that transit params.arguments.
_SESSION_DURATION_SECONDS = 900

# Session-policy temporal window. The session policy carries a
# `DateLessThan` on `aws:CurrentTime` set `_SESSION_POLICY_TTL_SECONDS` in the
# future, so the `Allow` stops matching one minute after the policy is built.
#
# Why 60 s, and why the bound lives in the POLICY rather than the SESSION length:
# the window only has to cover the tool's own execution (the tool Lambdas time
# out at 10 seconds), so a one-minute policy window is ample. It CANNOT be
# expressed as a shorter STS session instead, because 900 seconds (15 minutes)
# is the STS `DurationSeconds` FLOOR — the smallest session STS will issue
# (verified: STS AssumeRole `DurationSeconds` "Minimum value of 900",
# https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html). The
# session itself therefore persists for the 900 s floor, but once
# `aws:CurrentTime` passes the policy's timestamp the `Allow` no longer matches
# and access is denied for the remainder of the session, so a leaked credential
# is unusable after one minute.
_SESSION_POLICY_TTL_SECONDS = 60

# IAM RoleSessionName maximum length (STS constraint).
_ROLE_SESSION_NAME_MAX_LEN = 64

# ---------------------------------------------------------------------------
# Shared, tenant-agnostic STS client (see the module docstring for why this is
# safe to reuse while the vended credentials are not).
# ---------------------------------------------------------------------------
_sts_client_singleton: Any = None
_sts_client_lock = threading.Lock()


def _sts_client() -> Any:
    """Return the per-container STS client, constructing it on first use.

    Double-checked locking: the fast path is a plain read of the module global,
    and construction happens at most once per container even under concurrent
    first calls. Serializing construction avoids the documented hazard of
    invoking ``boto3.client()`` from within a concurrent context
    (https://docs.aws.amazon.com/boto3/latest/guide/clients.html).

    The returned client authenticates as the interceptor's own execution role and
    holds NO tenant credentials, so reusing it across requests and threads
    introduces no cross-tenant state.

    Returns:
        The shared boto3 STS client.
    """
    global _sts_client_singleton
    client = _sts_client_singleton
    if client is None:
        with _sts_client_lock:
            if _sts_client_singleton is None:
                _sts_client_singleton = boto3.client("sts")
            client = _sts_client_singleton
    return client


def reset_sts_client() -> None:
    """Drop the shared STS client so the next call rebuilds it.

    TEST SEAM ONLY. Production never calls this: the client is meant to live for
    the life of the container. Tests that monkeypatch ``boto3.client`` need the
    singleton cleared between cases, otherwise the first test's fake would be
    reused by every later test.
    """
    global _sts_client_singleton
    with _sts_client_lock:
        _sts_client_singleton = None


def build_session_policy(
    served_scope: str,
    table_arn: str,
    actions: list[str],
    expires_at: str,
) -> str:
    """Build the inline STS session policy confining access in space AND time.

    A **pure** function of its inputs: it computes no wall-clock time itself. The
    caller (:func:`vend_scoped_credentials`) computes ``expires_at`` and injects
    it, which keeps this builder deterministic and mirrors the article, which
    computes the expiry outside the builder.

    Produces the session-policy JSON: a single ``Allow`` statement granting
    ``actions`` on ``table_arn``, conditioned by three guards inside one
    ``Condition`` block:

    - **Space (unchanged):** ``ForAllValues:StringEquals`` on
      ``dynamodb:LeadingKeys`` equal to ``[served_scope]`` (confines the request's
      partition key to the served scope), plus a ``Null`` presence check requiring
      the key to be present (the ``ForAllValues`` footgun guard — a
      ``ForAllValues`` match is vacuously true when the key is absent).
    - **Time:** ``DateLessThan`` on ``aws:CurrentTime`` equal to
      ``expires_at``, so the ``Allow`` stops matching once the current time passes
      the caller-supplied expiry.

    The granted permissions are the intersection of the assumed role's
    identity-based policy and this session policy, so it can only narrow, never
    widen.

    AWS grounding (verified against the AWS documentation):
        - ``DateLessThan`` is a Date condition operator ("Matching before a
          specific date and time"), used with the ``aws:CurrentTime`` global
          condition key; the value is an ISO 8601 date/time string (the doc
          example uses ``"2020-06-30T23:59:59Z"``, i.e. ``%Y-%m-%dT%H:%M:%SZ``):
          https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_condition_operators.html
          https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_aws-dates.html
        - The resulting session's permissions are the INTERSECTION of the role's
          identity-based policy and the session policy:
          https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html

    Args:
        served_scope: The authoritative, JWT-derived scope to confine access to.
        table_arn: ARN of the Documents table (statement ``Resource``).
        actions: :data:`READ_ACTIONS` or :data:`WRITE_ACTIONS` — never a keyless
            action such as ``dynamodb:Scan``.
        expires_at: The caller-supplied, already-formatted ISO 8601 whole-second
            UTC expiry string ending in ``Z`` (format ``%Y-%m-%dT%H:%M:%SZ``),
            used verbatim as the ``DateLessThan`` / ``aws:CurrentTime`` value.

    Returns:
        The session policy as a JSON string (the STS ``Policy`` parameter is a
        string).
    """
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ScopedDocumentsAccess",
                "Effect": "Allow",
                "Action": actions,
                "Resource": table_arn,
                "Condition": {
                    "ForAllValues:StringEquals": {
                        "dynamodb:LeadingKeys": [served_scope],
                    },
                    "Null": {
                        "dynamodb:LeadingKeys": "false",
                    },
                    "DateLessThan": {
                        "aws:CurrentTime": expires_at,
                    },
                },
            },
        ],
    }
    return json.dumps(policy)


def vend_scoped_credentials(
    role_arn: str,
    served_scope: str,
    table_arn: str,
    actions: list[str],
) -> dict[str, str]:
    """Vend fresh temporary creds scoped to ``served_scope`` (no cache).

    Calls ``sts:AssumeRole`` **exactly once** with the inline ``LeadingKeys``
    session policy built by :func:`build_session_policy`, a ``scope`` SESSION TAG
    carrying ``served_scope`` (the roles' own ABAC condition and their trust
    policies both require it), and ``DurationSeconds=900`` (the STS floor), and
    returns ONLY the three snake_case credential strings the
    tool needs. There is NO cache and NO ``cache_hit`` flag: every call vends its
    own session, so the one-minute ``aws:CurrentTime``
    policy window can never hand out already-expired cached credentials. The
    credentials are confined to the ``served_scope`` partition for ``actions`` by
    TWO independent controls — the session policy AND the role's tag-conditioned
    identity policy.

    Args:
        role_arn: ARN of the scoped role to assume (read or write role).
        served_scope: The authoritative scope to confine the session to.
        table_arn: ARN of the Documents table (session-policy Resource).
        actions: :data:`READ_ACTIONS` or :data:`WRITE_ACTIONS`.

    Returns:
        A single credentials dict with exactly ``access_key_id`` /
        ``secret_access_key`` / ``session_token`` (no ``cache_hit``, no tuple).

    Raises:
        ScopeTagError: If ``served_scope`` cannot be safely expressed as a session
            tag (empty, over the STS 256-character value limit, or containing an
            IAM wildcard). Raised BEFORE ``AssumeRole`` is called, so no
            credential is minted. It subclasses ``RuntimeError``, which the
            handler already catches on the vend path, so it takes the same
            fail-closed route as an STS error.
        botocore.exceptions.ClientError / BotoCoreError: If ``AssumeRole`` fails.
            The error is PROPAGATED unchanged — it is never swallowed into a
            partial or ``None`` credential. The caller (the REQUEST interceptor
            handler) catches it and fails closed with a generic, detail-free
            short-circuit error that discloses no scope, role, table, or
            credential detail, handing the tool no credential and no fallback
            route to the table. A trust-policy rejection of an
            UNTAGGED assume surfaces here as ``AccessDenied`` and takes the same
            route.
    """
    # Compute the caller-side expiry and pass it into the PURE
    # build_session_policy. The value is now + _SESSION_POLICY_TTL_SECONDS,
    # formatted as a whole-second UTC ISO 8601 string ending in `Z`
    # (`%Y-%m-%dT%H:%M:%SZ`), which is the form the `DateLessThan` /
    # `aws:CurrentTime` condition expects.
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=_SESSION_POLICY_TTL_SECONDS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build the session tag BEFORE the call so a rejected value fails closed
    # without minting anything. The same string is used for the tag and
    # for the session policy's LeadingKeys value — they must agree, because the
    # role's own ABAC condition requires LeadingKeys == ${aws:PrincipalTag/scope}
    # and a disagreement denies every request.
    tags = _scope_tag_or_raise(served_scope)

    # Vend EXACTLY ONCE: no cache read, no cache write.
    # The CLIENT is shared per container (it holds no tenant identity, see the
    # module docstring); the CREDENTIALS it returns are still vended fresh on
    # every call and are never cached.
    response = _sts_client().assume_role(
        RoleArn=role_arn,
        RoleSessionName=f"scope-{served_scope}"[:_ROLE_SESSION_NAME_MAX_LEN],
        Policy=build_session_policy(served_scope, table_arn, actions, expires_at),
        Tags=tags,
        DurationSeconds=_SESSION_DURATION_SECONDS,
    )
    # Map the STS Credentials response into EXACTLY the three snake_case fields
    # that ride in `tenant_credentials`: AccessKeyId ->
    # access_key_id, SecretAccessKey -> secret_access_key, SessionToken ->
    # session_token. `Expiration` and every other response key are excluded from
    # the vended credentials the tool receives. STS `Credentials` response shape
    # (AccessKeyId, SecretAccessKey, SessionToken, Expiration) per the AWS
    # documentation:
    # https://docs.aws.amazon.com/STS/latest/APIReference/API_Credentials.html
    raw = response["Credentials"]
    return {
        "access_key_id": raw["AccessKeyId"],
        "secret_access_key": raw["SecretAccessKey"],
        "session_token": raw["SessionToken"],
    }


def build_tenant_context(
    served_scope: str,
    creds: dict[str, str],
) -> dict[str, Any]:
    """Assemble the ``context`` wire-contract object for ``arguments["context"]``.

    Canonical helper for the single-``context`` wire contract. The REQUEST
    interceptor writes the returned object as the sole ``context`` key it adds to
    the (deep-copied) tool ``arguments``.

    The object carries EXACTLY two keys::

        {
            "served_scope": "<scope>",
            "tenant_credentials": {
                "access_key_id": ...,
                "secret_access_key": ...,
                "session_token": ...,
            },
        }

    ``tenant_credentials`` is rebuilt here from the three named snake_case fields
    only, so any extra key present on ``creds`` is dropped and never reaches the
    wire. The STS-to-snake_case field mapping (``AccessKeyId``->``access_key_id``,
    ``SecretAccessKey``->``secret_access_key``, ``SessionToken``->``session_token``)
    and the exclusion of ``Expiration`` happen upstream in
    :func:`vend_scoped_credentials`, where the raw STS ``Credentials`` response is
    shaped.

    Args:
        served_scope: The authoritative, JWT-derived scope string.
        creds: The three-field credentials dict returned by
            :func:`vend_scoped_credentials` (``access_key_id`` /
            ``secret_access_key`` / ``session_token``).

    Returns:
        The ``context`` object carrying exactly ``served_scope`` and
        ``tenant_credentials`` (which in turn carries exactly the three
        snake_case credential fields).
    """
    return {
        "served_scope": served_scope,
        "tenant_credentials": {
            "access_key_id": creds["access_key_id"],
            "secret_access_key": creds["secret_access_key"],
            "session_token": creds["session_token"],
        },
    }
