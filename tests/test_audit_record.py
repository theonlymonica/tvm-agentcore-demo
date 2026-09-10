"""The audit record must answer three questions and leak nothing.

``interceptor/audit_record.py`` is the only place the requested tool, the ORIGINAL
model-supplied arguments and the granted scope exist together. It is also the only
record in this system that is ALLOWED to contain original arguments, which makes it
the one most worth pinning: a field added carelessly here is a credential or a
document body in a log group.

Two classes of assertion below, and they are not the same strength:

- Content: the three questions are answerable, and the join keys to CloudTrail are
  present. Ordinary regression cover.
- Absence: no credential value or credential-shaped key can appear. Asserted by
  feeding the builder arguments that CONTAIN planted secrets and proving they do not
  survive into the record — a test that fails if someone widens the copy.

The structural guarantee (the record is built before credentials exist) is enforced in
``interceptor/handler.py`` by call order; ``TestSignatureCannotAcceptCredentials``
below pins the half of it that is checkable here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from interceptor.audit_record import (
    AUDIT_MARKER,
    AUDIT_SCHEMA_VERSION,
    MAX_ARGUMENT_CHARS,
    build_audit_record,
    emit_audit_record,
)
from interceptor.scoped_credentials import build_role_session_name

SUBJECT = "11111111-2222-3333-4444-555555555555"
SESSION_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SCOPE = "payments-core"

#: Values planted into the arguments to prove they cannot reach the record.
PLANTED_SECRETS = (
    "ASIAEXAMPLEACCESSKEY",
    "wJalrXUtnFEMIEXAMPLESECRETKEY",
    "IQoJb3JpZ2luX2VjEXAMPLESESSIONTOKEN",
)


def _record(**overrides: Any) -> dict[str, Any]:
    """Build a record with sensible defaults, overridable per test."""
    kwargs: dict[str, Any] = {
        "tool_requested": "documents___read_document",
        "tool_classified": "read_document",
        "arguments": {"doc_id": "PAY-001"},
        "granted_scope": SCOPE,
        "subject": SUBJECT,
        "gateway_session_id": SESSION_ID,
        "request_id": 7,
    }
    kwargs.update(overrides)
    return build_audit_record(**kwargs)


class TestTheThreeQuestionsAreAnswerable:
    def test_records_the_requested_and_classified_tool(self) -> None:
        """Both names, because a mismatch between them is audit-relevant."""
        record = _record()
        assert record["tool_requested"] == "documents___read_document"
        assert record["tool_classified"] == "read_document"

    def test_records_the_original_arguments(self) -> None:
        """The arguments as the model supplied them — this record's whole point."""
        record = _record(arguments={"doc_id": "PAY-001", "query": "invoice"})
        assert record["arguments"] == {"doc_id": "PAY-001", "query": "invoice"}

    def test_records_the_granted_scope(self) -> None:
        """What the interceptor DECIDED, which no AWS service observes."""
        assert _record()["granted_scope"] == SCOPE

    def test_carries_a_schema_version(self) -> None:
        """A stored record must be readable by the rules in force when written."""
        assert _record()["audit_schema"] == AUDIT_SCHEMA_VERSION


class TestJoinKeysToCloudTrail:
    def test_subject_joins_to_source_identity(self) -> None:
        """The `sub` is what CloudTrail carries as sessionContext.sourceIdentity."""
        assert _record()["subject"] == SUBJECT

    def test_derived_role_session_name_matches_the_vend(self) -> None:
        """The derived name is recorded so joining needs no knowledge of the rule.

        Derived through the SAME function the vend uses, so the two cannot drift.
        """
        assert _record()["role_session_name"] == f"gw-{SESSION_ID}"

    def test_missing_session_id_still_produces_a_record(self) -> None:
        """A call that cannot be joined must not be an INVISIBLE call.

        The vend fails closed on a missing session id. If the record were skipped
        too, a refused attempt would leave no trace anywhere — the opposite of what
        this record is for.
        """
        record = _record(gateway_session_id=None)
        assert record["gateway_session_id"] is None
        assert record["role_session_name"] is None
        assert record["subject"] == SUBJECT

    def test_an_overlong_session_id_still_joins_via_the_digest_form(self) -> None:
        """An awkward identifier must still produce the name the vend will use.

        The derivation is total: an identifier too long (or carrying a character STS
        forbids) comes back as the ``gwh-<digest>`` form rather than failing. The
        record must carry that same form, because that is what CloudTrail will show
        — recording null here would break a join that actually works.
        """
        record = _record(gateway_session_id="x" * 500)
        assert record["role_session_name"] == build_role_session_name("x" * 500)
        assert record["role_session_name"].startswith("gwh-")

    def test_a_whitespace_only_session_id_records_a_null_join_key(self) -> None:
        """The one case with no derivable name must be null, never invented.

        A fabricated join key would point at a CloudTrail event that does not exist,
        which is worse than admitting the join is unavailable. The vend fails closed
        on the same value.
        """
        record = _record(gateway_session_id="   ")
        assert record["role_session_name"] is None


class TestNoCredentialCanAppear:
    def test_the_model_supplied_context_key_is_dropped(self) -> None:
        """`context` is the key that carries credentials in the ENRICHED body.

        Excluding it means this record cannot hold a credential-shaped object even
        if the emit were later moved after the vend.
        """
        record = _record(
            arguments={
                "doc_id": "PAY-001",
                "context": {"tenant_credentials": {"secret_access_key": "nope"}},
            }
        )
        assert "context" not in record["arguments"]
        assert record["arguments"] == {"doc_id": "PAY-001"}

    @pytest.mark.parametrize("secret", PLANTED_SECRETS)
    def test_planted_credentials_inside_context_do_not_survive(
        self, secret: str
    ) -> None:
        """Serialize the whole record and prove the secret is absent from the text."""
        record = _record(
            arguments={
                "doc_id": "PAY-001",
                "context": {
                    "served_scope": SCOPE,
                    "tenant_credentials": {
                        "access_key_id": secret,
                        "secret_access_key": secret,
                        "session_token": secret,
                    },
                },
            }
        )
        assert secret not in json.dumps(record)

    def test_no_credential_shaped_key_at_the_top_level(self) -> None:
        """The record's own keys are a closed set that contains no secret."""
        assert set(_record()) == {
            "audit_schema",
            "subject",
            "gateway_session_id",
            "role_session_name",
            "tool_requested",
            "tool_classified",
            "granted_scope",
            "trace_id",
            "traceparent",
            "arguments",
            "request_id",
        }


class TestTraceContextIsRecordedButNeverTrusted:
    def test_trace_headers_are_recorded_when_present(self) -> None:
        """The trace id is the join to the model's reasoning.

        Recorded by the INTERCEPTOR, from the request headers, because the trace is
        generated by AgentCore's own instrumentation. The agent is not asked to
        report it: this architecture assumes the agent may be hostile, so a
        correlation the agent supplied would be the audited party keeping its own
        books.
        """
        record = _record(
            trace_id="Root=1-5759e988-bd862e3fe1be46a994272793",
            traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        )
        assert record["trace_id"] == "Root=1-5759e988-bd862e3fe1be46a994272793"
        assert record["traceparent"].startswith("00-4bf92f3577b34da6a3ce929d0e")

    def test_absent_trace_headers_are_null_not_omitted(self) -> None:
        """A missing join must be visible as null rather than as a missing key.

        An absent key reads as "this record predates trace capture"; an explicit null
        reads as "the Gateway forwarded no trace context for this call", which is the
        fact an auditor needs.
        """
        record = _record()
        assert record["trace_id"] is None
        assert record["traceparent"] is None


class TestSignatureCannotAcceptCredentials:
    def test_every_parameter_is_keyword_only(self) -> None:
        """No positional slot exists for a credential to be passed into by accident.

        Half of the structural guarantee: the other half is call ORDER in the
        handler, which this test cannot see.
        """
        import inspect

        params = inspect.signature(build_audit_record).parameters
        positional = [
            name
            for name, param in params.items()
            if param.kind
            in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        assert positional == [], (
            "build_audit_record must take keyword-only arguments so no caller can "
            f"pass an unnamed value into it; found positional: {positional}"
        )

    def test_no_parameter_is_named_after_a_credential(self) -> None:
        """The signature is the declaration of what this record is allowed to hold."""
        forbidden = {"creds", "credentials", "tenant_credentials", "token", "secret"}
        params = set(inspect_parameters())
        assert not (params & forbidden), f"credential-ish parameter: {params & forbidden}"


def inspect_parameters() -> list[str]:
    """Return build_audit_record's parameter names."""
    import inspect

    return list(inspect.signature(build_audit_record).parameters)


class TestArgumentsAreBoundedButHonest:
    def test_a_long_argument_is_truncated_and_says_so(self) -> None:
        """A runaway `reply` body must not push the rest of the record out.

        Truncation is recorded rather than silent: an audit record that quietly
        shortens evidence is worse than one that admits it.
        """
        body = "x" * (MAX_ARGUMENT_CHARS + 500)
        record = _record(arguments={"body": body})
        recorded = record["arguments"]["body"]

        assert recorded["truncated"] is True
        assert recorded["chars"] == MAX_ARGUMENT_CHARS + 500
        assert len(recorded["value"]) == MAX_ARGUMENT_CHARS

    def test_a_short_argument_is_untouched(self) -> None:
        """No wrapping when there is nothing to truncate."""
        assert _record(arguments={"body": "short"})["arguments"]["body"] == "short"

    def test_an_unserializable_value_degrades_instead_of_raising(self) -> None:
        """An exotic type must not break the emit inside the request path."""
        record = _record(arguments={"weird": {1, 2, 3}})
        assert isinstance(record["arguments"]["weird"], str)
        json.dumps(record)

    def test_the_input_arguments_are_never_mutated(self) -> None:
        """The handler forwards this same dict; the record must not disturb it."""
        arguments = {"doc_id": "PAY-001", "context": {"model": "supplied"}}
        before = {"doc_id": "PAY-001", "context": {"model": "supplied"}}
        _record(arguments=arguments)
        assert arguments == before


class TestEmit:
    def test_emits_one_parseable_line_under_the_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One line, marker then JSON, so a Logs Insights filter can select it."""
        log = logging.getLogger("audit-test")
        with caplog.at_level(logging.INFO, logger="audit-test"):
            emit_audit_record(log, _record())

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert message.startswith(f"{AUDIT_MARKER} ")

        payload = json.loads(message[len(AUDIT_MARKER) + 1 :])
        assert payload["subject"] == SUBJECT
        assert payload["granted_scope"] == SCOPE

    def test_serialization_is_stable(self, caplog: pytest.LogCaptureFixture) -> None:
        """Sorted keys, so two records for one call are byte-comparable."""
        log = logging.getLogger("audit-test-stable")
        with caplog.at_level(logging.INFO, logger="audit-test-stable"):
            emit_audit_record(log, _record())
            emit_audit_record(log, _record())

        first, second = (r.getMessage() for r in caplog.records)
        assert first == second
