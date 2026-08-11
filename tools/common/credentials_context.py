"""
Tool-side parsing and fail-closed validation of the injected ``context`` object.

Wire contract: the REQUEST interceptor injects a single UNDECLARED ``context``
object at ``arguments["context"]`` (mapped into the Lambda ``event`` as
``event["context"]``) rather than three flat top-level fields. The object carries
the authoritative served scope and the vended credentials::

    event["context"] = {
        "served_scope": "<scope>",
        "tenant_credentials": {
            "access_key_id": "...",
            "secret_access_key": "...",
            "session_token": "...",
        },
    }

This module owns the *context parsing and validation* half of the tool
credentials module. It was split out of ``tools/common/scoped_credentials.py``
when the combined module grew past ~350 lines; the table-builder entry points
(:func:`documents_table_from_event`, :func:`context_credentials_from_event`)
remain in ``scoped_credentials.py`` and import :func:`validated_context` and
:func:`served_scope_from_event` from here.

Fail-closed contract: a missing or malformed ``context`` raises
:class:`ScopedCredentialsError`. Callers surface a generic, detail-free error and
NEVER fall back to the tool's execution role or the default credential chain.

Security:
    This module NEVER logs the event-supplied scope or credentials, and its error
    message names no scope, credential field, or value.

AWS documentation references:
    - STS ``Credentials`` shape (AccessKeyId, SecretAccessKey, SessionToken) —
      the source field names the interceptor maps into ``tenant_credentials``:
      https://docs.aws.amazon.com/STS/latest/APIReference/API_Credentials.html
    - boto3/AWS credential field names (``aws_access_key_id``,
      ``aws_secret_access_key``, ``aws_session_token``) the three
      ``tenant_credentials`` snake_case fields map onto for a session:
      https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/credentials.html

Constants:
    CONTEXT_KEY / SERVED_SCOPE_KEY / TENANT_CREDENTIALS_KEY: the ``context``
        object key names.
    CONTEXT_CRED_TO_SESSION_KWARG: mapping of the three snake_case
        ``tenant_credentials`` fields onto the boto3 ``Session`` keyword
        arguments; its keys are also the completeness set.

Functions:
    validated_context: Return ``event["context"]`` after fail-closed validation.
    served_scope_from_event: Return the authoritative served scope string.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# `context` wire contract. The REQUEST interceptor writes a single UNDECLARED
# `context` object at arguments["context"] (mapped into the Lambda event as
# event["context"]); see the module docstring for the shape.
# ---------------------------------------------------------------------------
CONTEXT_KEY = "context"
SERVED_SCOPE_KEY = "served_scope"
TENANT_CREDENTIALS_KEY = "tenant_credentials"

# Mapping of the three snake_case tenant-credential fields (as written inside
# context["tenant_credentials"]) onto the boto3 Session keyword arguments they
# populate, by name. This mapping's KEYS are also the completeness set for the
# validation below: all three must be present as non-empty strings.
CONTEXT_CRED_TO_SESSION_KWARG: dict[str, str] = {
    "access_key_id": "aws_access_key_id",
    "secret_access_key": "aws_secret_access_key",
    "session_token": "aws_session_token",
}


class ScopedCredentialsError(RuntimeError):
    """Raised when the interceptor-injected ``context`` is missing or malformed.

    Signals that the tool has no scoped credentials (or no served scope) to use.
    The tool handler surfaces a generic error and NEVER falls back to its own
    execution role (which holds no DynamoDB permission anyway) or the default
    credential chain.
    """


def validated_context(event: dict[str, Any]) -> dict[str, Any]:
    """Return the injected ``context`` object after fail-closed validation.

    Enforces the wire contract on ``event["context"]``. It raises
    :class:`ScopedCredentialsError` (never returning) when ANY of the following
    holds, so the caller fails closed and NEVER falls back to the tool's execution
    role or the default credential chain:

      * ``event["context"]`` is missing, or is not an object (``dict``);
      * ``context["served_scope"]`` is not a non-empty string;
      * ``context["tenant_credentials"]`` is not a *complete* object — meaning a
        ``dict`` carrying non-empty string values for ALL three of
        ``access_key_id``, ``secret_access_key`` and ``session_token``.

    The error message names no scope, credential field, or value (the fail-closed
    path is deliberately detail-free); a single generic message covers every branch
    so the failure mode cannot be distinguished by an attacker.

    Args:
        event: The Lambda event, expected to carry ``event["context"]``.

    Returns:
        The validated ``context`` dict (its ``served_scope`` and
        ``tenant_credentials`` are guaranteed well-formed).

    Raises:
        ScopedCredentialsError: If the injected context is missing or malformed.
    """
    context = event.get(CONTEXT_KEY)
    if not isinstance(context, dict):
        raise ScopedCredentialsError("injected context is missing or malformed")

    raw_scope = context.get(SERVED_SCOPE_KEY)
    if not (isinstance(raw_scope, str) and raw_scope.strip()):
        raise ScopedCredentialsError("injected context is missing or malformed")

    creds = context.get(TENANT_CREDENTIALS_KEY)
    if not isinstance(creds, dict):
        raise ScopedCredentialsError("injected context is missing or malformed")

    for field in CONTEXT_CRED_TO_SESSION_KWARG:
        value = creds.get(field)
        if not (isinstance(value, str) and value):
            raise ScopedCredentialsError("injected context is missing or malformed")

    return context


def served_scope_from_event(event: dict[str, Any]) -> str:
    """Return the authoritative served scope from the injected ``context``.

    Reads ``event["context"]["served_scope"]`` — the JWT-derived scope the REQUEST
    interceptor injected — after the full fail-closed validation. The returned
    value is stripped of surrounding whitespace.

    Args:
        event: The Lambda event (the tool's declared ``inputSchema`` properties
            plus the interceptor-injected ``context`` object).

    Returns:
        The authoritative served scope string.

    Raises:
        ScopedCredentialsError: If the injected ``context`` is missing or
            malformed (missing/empty ``served_scope`` or an incomplete
            ``tenant_credentials`` object). The caller surfaces a generic error
            and NEVER falls back to its execution role or the default chain.
    """
    context = validated_context(event)
    return context[SERVED_SCOPE_KEY].strip()
