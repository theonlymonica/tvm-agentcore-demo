"""The one audit record the interceptor writes per scoped tool call.

WHAT THIS ANSWERS, and why no AWS service can answer it for us.

Work item 1 made the DynamoDB access attributable: a CloudTrail data event now names
the human (``sourceIdentity``) and the request (``RoleSessionName``). But three of the
auditor's questions are invisible to CloudTrail, because they are decided BEFORE any
AWS API is called:

- which tool the model actually asked for,
- with which arguments the model asked for it,
- which scope the interceptor decided to grant.

By the time STS is called, the tool name has become a role ARN and the arguments have
become nothing at all — DynamoDB sees a key, not a request. The interceptor is the only
component that ever holds all three facts, and it holds them for a few microseconds
before throwing them away. So this record is authored, not collected.

WHY IT IS WRITTEN BEFORE THE VEND, which is the important design property.

The record is emitted at the point where the granted scope is known and the credentials
DO NOT YET EXIST. That ordering is the guarantee: this record cannot leak a credential
because there is no credential in scope to leak. That is a structural property of the
call graph, not a redaction step someone has to remember to keep correct — and unlike a
scrubber, it cannot rot when a future field is added.

Two consequences worth knowing:

1. A tool call whose vend FAILS is still audited. An attempted access that was refused
   is exactly the kind of event an auditor cares about, and a record written after a
   successful vend would silently omit it.
2. This record therefore says what was REQUESTED and GRANTED, never what succeeded.
   The outcome lives in the CloudTrail data event (or in its absence).

WHAT IS DELIBERATELY EXCLUDED.

The model-supplied ``context`` key is dropped before the arguments are recorded, for two
independent reasons. First, the handler's contract is that any value already at
``arguments["context"]`` is overwritten WITHOUT BEING READ — recording it would break
that promise, and the value is attacker-controlled (the body is written by the model).
Second, ``context`` is the exact key that carries ``tenant_credentials`` in the ENRICHED
body, so excluding it means this record cannot carry a credential-shaped object even if
someone later moves the emit call after the vend. The structural guarantee above is the
primary defence; this is the belt to its braces.

Functions:
    build_audit_record: Assemble the record from the pre-vend facts.
    emit_audit_record: Write it as one parseable line.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from interceptor.scoped_credentials import (
    RoleSessionNameError,
    build_role_session_name,
)

#: Line prefix the record is emitted under. Chosen so a CloudWatch Logs Insights
#: query can select these lines and nothing else:
#:   fields @timestamp, @message | filter @message like /TOOL_CALL_AUDIT/
#: Kept as a bare token rather than JSON-only output because the interceptor's log
#: group also carries plain human lines, and `parse` on a mixed group needs an anchor.
AUDIT_MARKER = "TOOL_CALL_AUDIT"

#: Bumped whenever a field's MEANING changes, so a stored record can be read by the
#: rules that were true when it was written. Adding a field does not require a bump.
AUDIT_SCHEMA_VERSION = 1

#: Per-value cap on recorded argument strings. `reply` carries a model-authored body
#: that has no useful upper bound, and one runaway argument must not be able to push
#: the rest of the record out of a log line. Truncation is RECORDED rather than silent
#: (see `_arguments_for_record`): an audit record that quietly shortens evidence is
#: worse than one that admits it.
MAX_ARGUMENT_CHARS = 2000

#: The model-supplied key that is never recorded — see the module docstring.
_EXCLUDED_ARGUMENT_KEY = "context"


def _value_for_record(value: Any) -> Any:
    """Return a JSON-safe form of one argument value, marking any truncation.

    Args:
        value: The model-supplied argument value, of any type.

    Returns:
        The value unchanged when it is short and JSON-serializable; a dict of the
        form ``{"truncated": True, "chars": N, "value": "..."}`` when a string was
        longer than :data:`MAX_ARGUMENT_CHARS`; or its ``repr`` when it cannot be
        serialized, so an exotic type degrades the record instead of breaking the
        emit.
    """
    if isinstance(value, str) and len(value) > MAX_ARGUMENT_CHARS:
        return {
            "truncated": True,
            "chars": len(value),
            "value": value[:MAX_ARGUMENT_CHARS],
        }

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _arguments_for_record(arguments: dict[str, Any]) -> dict[str, Any]:
    """Copy the model-supplied arguments, dropping the excluded key.

    Args:
        arguments: The arguments exactly as the model supplied them, BEFORE the
            interceptor writes its ``context`` object.

    Returns:
        A new dict safe to serialize. The input is never mutated.
    """
    return {
        key: _value_for_record(value)
        for key, value in arguments.items()
        if key != _EXCLUDED_ARGUMENT_KEY
    }


def build_audit_record(
    *,
    tool_requested: str,
    tool_classified: str,
    arguments: dict[str, Any],
    granted_scope: str,
    subject: str,
    gateway_session_id: Optional[str],
    request_id: Any = None,
    trace_id: Optional[str] = None,
    traceparent: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the audit record from facts known before any credential exists.

    Every parameter is keyword-only and none of them is a credential: the signature
    is the enforcement point for "this record cannot contain a secret", so a future
    caller cannot pass one in positionally by accident.

    Args:
        tool_requested: The tool name EXACTLY as it arrived in the request, before
            classification. Recorded separately from the classified name because a
            mismatch between the two is itself audit-relevant.
        tool_classified: The name the interceptor resolved it to and acted on.
        arguments: The model-supplied arguments, unmodified.
        granted_scope: The authoritative JWT-derived scope the interceptor granted.
        subject: The Cognito ``sub`` — joins this record to
            ``sessionContext.sourceIdentity`` in CloudTrail.
        gateway_session_id: The Gateway ``Mcp-Session-Id``. May be None; the record
            is still written, with a null join key, so a refused call is not
            invisible.
        request_id: The JSON-RPC request id, echoed for correlation with the caller.
        trace_id: The X-Ray format trace header, if the Gateway forwarded one.
        traceparent: The W3C format trace header, if the Gateway forwarded one.

    Returns:
        The record as a plain dict, ready to serialize.
    """
    try:
        role_session_name = (
            build_role_session_name(gateway_session_id) if gateway_session_id else None
        )
    except RoleSessionNameError:
        # The derivation is TOTAL for any identifier with content: one too long, or
        # carrying a character STS forbids, comes back as the `gwh-<digest>` form
        # rather than failing, so the join still holds. What reaches here is an
        # identifier that is non-empty but has no content once stripped — whitespace
        # only. The vend fails closed on the same value; this record says so instead
        # of inventing a join key no CloudTrail event will ever carry.
        role_session_name = None

    return {
        "audit_schema": AUDIT_SCHEMA_VERSION,
        "subject": subject,
        "gateway_session_id": gateway_session_id,
        # The derived STS RoleSessionName, so joining to CloudTrail needs no
        # knowledge of the derivation rule. Same source of truth as the vend.
        "role_session_name": role_session_name,
        "tool_requested": tool_requested,
        "tool_classified": tool_classified,
        "granted_scope": granted_scope,
        # The trace context, when the Gateway forwards it. This is the join to the
        # REASONING half of the chain, and it is the only candidate that is
        # trustworthy: the trace id is generated by AgentCore's own instrumentation,
        # which "propagates trace context across agent boundaries without custom
        # code", and the same trace carries both the LLM calls and the tool
        # invocations. Recording it HERE rather than in the agent is the whole point
        # — the agent is the component this architecture assumes may be hostile, so
        # a correlation it wrote itself would be the audited party keeping its own
        # books. The interceptor is gateway-side and already the trust anchor.
        # Null when the Gateway forwards no trace header; see notes/ for whether it
        # does, which was settled by experiment rather than by the documented
        # example payload (that example is not exhaustive).
        "trace_id": trace_id,
        "traceparent": traceparent,
        "arguments": _arguments_for_record(arguments),
        "request_id": request_id,
    }


def emit_audit_record(
    log: logging.Logger,
    record: dict[str, Any],
) -> None:
    """Write the record as one parseable line.

    Serialized with sorted keys so two records for the same call are byte-comparable,
    and with ``default=repr`` so an unexpected type degrades the line instead of
    raising inside the request path — an audit emit must never be the reason a
    request fails.

    Args:
        log: The logger to write through.
        record: The record from :func:`build_audit_record`.
    """
    log.info(
        "%s %s",
        AUDIT_MARKER,
        json.dumps(record, sort_keys=True, default=repr),
    )
