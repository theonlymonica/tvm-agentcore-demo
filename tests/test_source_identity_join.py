"""WORK ITEM 1 — CloudTrail attribution: SourceIdentity and the join key.

What these tests pin, and why each one earns its place:

The session policy and the ``scope`` session tag answer "what may this credential
touch". They say nothing about "who caused this", so every DynamoDB action taken
with a vended credential used to appear in CloudTrail under the role session name
``scope-payments-core`` — identical for every user, every tool and every request.
This module pins the two parameters that fix that, and pins them at the level where
each is actually decided:

1. ``SourceIdentity`` carries the Cognito ``sub``, so an action is attributable to a
   PERSON. It is immutable once set and present in the request context of every
   subsequent action, which is what makes it usable as an audit anchor.
2. ``RoleSessionName`` is derived from the GATEWAY-supplied identifier, so an action
   is attributable to a REQUEST. It is the field an auditor joins on: it appears in
   the ``AssumeRole`` event and in the ``sessionContext`` of every DynamoDB data
   event the credentials produce.
3. Both roles' TRUST policies allow ``sts:SetSourceIdentity``. This is not optional
   hardening — an ``AssumeRole`` passing ``SourceIdentity`` FAILS outright when the
   trust policy omits the action, so the vend depends on it.
4. NEITHER value is sourced from the request body. The body is written by the model,
   so a body-derived identity would let an injected model choose the identity it is
   audited under — which would make the audit trail actively misleading rather than
   merely absent.
5. A token with no usable ``sub`` fails CLOSED. A credential nobody can be held to
   is worse than a refused request.

AWS documentation references (verified against the AWS documentation):
    - ``AssumeRole`` request parameters. ``RoleSessionName``: "Length Constraints:
      Minimum length of 2. Maximum length of 64. Pattern: [\\w+=,.@-]*".
      ``SourceIdentity``: same pattern and window, plus "You cannot use a value that
      begins with the text aws:. This prefix is reserved for AWS internal use":
      https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
    - Source identity requires ``sts:SetSourceIdentity`` in the role trust policy or
      "the AssumeRole* operation will fail"; the value is visible in CloudTrail:
      https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_monitor.html
    - ``aws:SourceIdentity`` — "after the source identity is set, the value cannot be
      changed. It is present in the request context for all actions taken by the
      role", unlike ``sts:RoleSessionName``:
      https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from aws_cdk.assertions import Template

import synth_helpers as sh  # noqa: F401  (puts cdk/ on sys.path)
import interceptor.handler as interceptor_handler
import interceptor.scoped_credentials as scoped_credentials
from interceptor.jwt_claims import VerifiedIdentity
from interceptor.scoped_credentials import (
    READ_ACTIONS,
    RoleSessionNameError,
    SourceIdentityError,
    build_role_session_name,
    vend_scoped_credentials,
)

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

_ROLE_ARN = "arn:aws:iam::123456789012:role/DocumentsAccessRole"
_TABLE_ARN = "arn:aws:dynamodb:us-east-1:123456789012:table/DocumentsTable"
_SERVED_SCOPE = "payments-core"

#: A Cognito ``sub``: a UUID, so 36 characters of hex and hyphens. It satisfies the
#: STS pattern and the 2–64 window with no derivation, which is what
#: ``test_cognito_sub_shape_satisfies_the_published_constraints`` asserts explicitly
#: rather than assuming.
_SUBJECT = "4f8a1b2c-3d4e-5f60-7182-93a4b5c6d7e8"

#: The Gateway-supplied identifier (``Mcp-Session-Id``).
_GATEWAY_ID = "mcp-session-0001"

_FAKE_STS_CREDENTIALS = {
    "AccessKeyId": "ASIAIOSFODNN7EXAMPLE",
    "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "SessionToken": "FQoGZXIvYXdzEBYaD" + "A" * 320,
    "Expiration": "2026-01-01T00:00:00Z",
}

_ASSUME_ACTION = "sts:AssumeRole"
_TAG_SESSION_ACTION = "sts:TagSession"
_SET_SOURCE_IDENTITY_ACTION = "sts:SetSourceIdentity"


class _RecordingFakeSts:
    """Stand-in STS client recording every ``assume_role`` call's kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def assume_role(self, **kwargs: Any) -> dict[str, Any]:
        """Record the kwargs and return fixed temporary credentials."""
        self.calls.append(kwargs)
        return {"Credentials": dict(_FAKE_STS_CREDENTIALS)}


@pytest.fixture
def fake_sts(monkeypatch: pytest.MonkeyPatch) -> _RecordingFakeSts:
    """Swap the interceptor's STS client for a recording fake."""
    sts = _RecordingFakeSts()
    monkeypatch.setattr(
        scoped_credentials.boto3,
        "client",
        lambda service_name, *a, **k: sts,
    )
    return sts


def _vend() -> dict[str, str]:
    """Perform one vend with the fixed identity."""
    return vend_scoped_credentials(
        _ROLE_ARN,
        _SERVED_SCOPE,
        _TABLE_ARN,
        READ_ACTIONS,
        subject=_SUBJECT,
        gateway_identifier=_GATEWAY_ID,
    )


# ---------------------------------------------------------------------------
# 1. SourceIdentity is set, and it is the subject
# ---------------------------------------------------------------------------


class TestAssumeRolePassesSourceIdentity:
    """The vend must name the human, not just the partition."""

    def test_source_identity_is_the_subject_verbatim(
        self, fake_sts: _RecordingFakeSts
    ) -> None:
        """``SourceIdentity`` equals the ``sub`` claim, unaltered."""
        _vend()

        assert len(fake_sts.calls) == 1, "AssumeRole must be called exactly once"
        assert fake_sts.calls[0]["SourceIdentity"] == _SUBJECT, (
            "SourceIdentity must carry the token's sub verbatim — a rewritten or "
            "truncated value would name a person who does not exist under that id"
        )

    def test_source_identity_is_passed_alongside_the_tag_and_policy(
        self, fake_sts: _RecordingFakeSts
    ) -> None:
        """Attribution is ADDITIVE — it must not displace either containment gate."""
        _vend()

        call = fake_sts.calls[0]
        assert call["SourceIdentity"] == _SUBJECT
        assert call["Tags"] == [{"Key": "scope", "Value": _SERVED_SCOPE}], (
            "the scope session tag must survive: it is one of the two containment "
            "gates, and attribution is not a substitute for containment"
        )
        assert call["Policy"], "the inline session policy must still be passed"
        assert call["DurationSeconds"] == 900

    def test_cognito_sub_shape_satisfies_the_published_constraints(self) -> None:
        """A Cognito ``sub`` is expressible as a ``SourceIdentity``.

        Asserted rather than assumed, because the whole design rests on it. Note
        the two AWS pages disagree on the maximum (the API reference says 64, the
        IAM User Guide says 256); a 36-character UUID satisfies BOTH, so this test
        holds under either reading.
        """
        assert 2 <= len(_SUBJECT) <= 64
        assert scoped_credentials._STS_NAME_PATTERN.match(_SUBJECT)
        assert not _SUBJECT.lower().startswith("aws:")


# ---------------------------------------------------------------------------
# 2. RoleSessionName is derived from the Gateway identifier
# ---------------------------------------------------------------------------


class TestRoleSessionNameFromGatewayIdentifier:
    """The join key: derived from the Gateway, deterministic, always valid."""

    def test_role_session_name_carries_the_gateway_identifier(
        self, fake_sts: _RecordingFakeSts
    ) -> None:
        """The identifier reaches STS as the session name, not the scope."""
        _vend()

        name = fake_sts.calls[0]["RoleSessionName"]
        assert name == f"gw-{_GATEWAY_ID}"
        assert _GATEWAY_ID in name, (
            "the Gateway identifier must be recoverable from the session name — it "
            "is the field the CloudTrail events are joined on"
        )

    def test_role_session_name_is_no_longer_derived_from_the_scope(
        self, fake_sts: _RecordingFakeSts
    ) -> None:
        """Regression pin on the defect this work item exists to fix.

        The previous name was ``f"scope-{served_scope}"``, which is identical for
        every user and every request in a scope, so CloudTrail could not tell two
        people apart.
        """
        _vend()

        assert fake_sts.calls[0]["RoleSessionName"] != f"scope-{_SERVED_SCOPE}"

    @pytest.mark.parametrize(
        ("identifier", "case"),
        [
            ("mcp-session-0001", "plain"),
            ("A" * 61, "exactly-at-the-ceiling"),
            ("abc_def.ghi@jkl=mno,pqr+stu", "every-permitted-punctuation"),
            ("x", "single-char"),
        ],
    )
    def test_permitted_identifiers_survive_verbatim(
        self, identifier: str, case: str
    ) -> None:
        """An identifier STS accepts is carried through, so it stays readable."""
        name = build_role_session_name(identifier)
        assert name == f"gw-{identifier}"
        assert len(name) <= 64
        assert scoped_credentials._STS_NAME_PATTERN.match(name)

    @pytest.mark.parametrize(
        ("identifier", "case"),
        [
            ("A" * 62, "one-over-the-ceiling"),
            ("A" * 400, "far-over-the-ceiling"),
            ("urn:uuid:1234", "colon-is-forbidden"),
            ("a/b", "slash-is-forbidden"),
            ("a b", "space-is-forbidden"),
            ("sessión-01", "non-ascii-is-forbidden"),
        ],
    )
    def test_awkward_identifiers_are_digested_not_rejected(
        self, identifier: str, case: str
    ) -> None:
        """An identifier STS would refuse is shortened deterministically.

        The derivation is TOTAL: every non-empty identifier yields a valid name, so
        an unusual Gateway identifier degrades the name's readability, never the
        vend's availability.
        """
        name = build_role_session_name(identifier)
        expected = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:32]

        assert name == f"gwh-{expected}"
        assert len(name) <= 64
        assert scoped_credentials._STS_NAME_PATTERN.match(name)

    def test_the_two_forms_are_never_ambiguous(self) -> None:
        """An auditor can tell a verbatim identifier from a digest by prefix."""
        verbatim = build_role_session_name("mcp-session-0001")
        digested = build_role_session_name("urn:uuid:mcp-session-0001")

        assert verbatim.startswith("gw-") and not verbatim.startswith("gwh-")
        assert digested.startswith("gwh-")

    def test_derivation_is_deterministic(self) -> None:
        """The same identifier always yields the same name, or the join breaks."""
        for identifier in ("mcp-session-0001", "urn:uuid:abc", "A" * 400):
            assert build_role_session_name(identifier) == build_role_session_name(
                identifier
            )

    @pytest.mark.parametrize("identifier", ["", "   ", None])
    def test_absent_identifier_fails_closed_and_mints_nothing(
        self, fake_sts: _RecordingFakeSts, identifier: Any
    ) -> None:
        """No Gateway identifier means no vend.

        There is deliberately no fallback name: a session that cannot be joined to
        the request that caused it defeats the purpose of naming it at all.
        """
        with pytest.raises(RoleSessionNameError):
            vend_scoped_credentials(
                _ROLE_ARN,
                _SERVED_SCOPE,
                _TABLE_ARN,
                READ_ACTIONS,
                subject=_SUBJECT,
                gateway_identifier=identifier,
            )

        assert fake_sts.calls == [], "AssumeRole must not be reached"

    def test_role_session_name_error_is_a_runtime_error(self) -> None:
        """So the handler's existing fail-closed catch covers it."""
        assert issubclass(RoleSessionNameError, RuntimeError)


# ---------------------------------------------------------------------------
# 3. Subject constraints — fail closed, never repair
# ---------------------------------------------------------------------------


class TestUnusableSubjectFailsClosed:
    """A subject STS would refuse must refuse the vend, not be rewritten."""

    @pytest.mark.parametrize(
        ("subject", "case"),
        [
            ("", "empty"),
            ("a", "below-the-minimum"),
            ("A" * 65, "over-the-maximum"),
            ("aws:internal", "reserved-prefix"),
            ("AWS:internal", "reserved-prefix-other-casing"),
            ("has space", "space-is-forbidden"),
            ("has/slash", "slash-is-forbidden"),
            ("has:colon", "colon-is-forbidden"),
            ("sessión", "non-ascii-is-forbidden"),
        ],
    )
    def test_unusable_subject_raises_and_mints_nothing(
        self, fake_sts: _RecordingFakeSts, subject: str, case: str
    ) -> None:
        """Validated BEFORE the call, so no credential exists to leak."""
        with pytest.raises(SourceIdentityError):
            vend_scoped_credentials(
                _ROLE_ARN,
                _SERVED_SCOPE,
                _TABLE_ARN,
                READ_ACTIONS,
                subject=subject,
                gateway_identifier=_GATEWAY_ID,
            )

        assert fake_sts.calls == [], (
            "AssumeRole must not be reached: no credential may be minted for a "
            "subject that cannot be expressed as a SourceIdentity"
        )

    def test_source_identity_error_is_a_runtime_error(self) -> None:
        """So the handler's existing fail-closed catch covers it."""
        assert issubclass(SourceIdentityError, RuntimeError)

    def test_rejection_does_not_echo_the_subject(
        self, fake_sts: _RecordingFakeSts
    ) -> None:
        """The subject identifies a person and must not land in an error string."""
        subject = "aws:forbidden-subject-value"
        with pytest.raises(SourceIdentityError) as excinfo:
            vend_scoped_credentials(
                _ROLE_ARN,
                _SERVED_SCOPE,
                _TABLE_ARN,
                READ_ACTIONS,
                subject=subject,
                gateway_identifier=_GATEWAY_ID,
            )

        disclosed = f"{excinfo.value}\n{excinfo.value.args!r}"
        assert subject not in disclosed


# ---------------------------------------------------------------------------
# 4. Neither value comes from the request body
# ---------------------------------------------------------------------------


class TestIdentityNeverComesFromTheRequestBody:
    """The model writes the body, so the body must not choose the audit identity."""

    #: Values planted in the model-controlled body. If any of these reaches STS, an
    #: injected model can forge how its own actions are attributed.
    _HOSTILE = {
        "sub": "00000000-dead-beef-0000-000000000000",
        "source_identity": "attacker-chosen-identity",
        "SourceIdentity": "attacker-chosen-identity-2",
        "role_session_name": "gw-attacker-session",
        "RoleSessionName": "gw-attacker-session-2",
        "gateway_identifier": "mcp-session-forged",
    }

    @staticmethod
    def _event() -> dict[str, Any]:
        """A scoped ``read_document`` call whose body is stuffed with hostile keys."""
        arguments: dict[str, Any] = {"doc_id": "PAY-001"}
        arguments.update(TestIdentityNeverComesFromTheRequestBody._HOSTILE)
        return {
            "mcp": {
                "gatewayRequest": {
                    "headers": {
                        "Authorization": "Bearer token",
                        "Mcp-Session-Id": _GATEWAY_ID,
                    },
                    "body": {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "ReadDocument___read_document",
                            "arguments": arguments,
                        },
                    },
                }
            }
        }

    @pytest.fixture(autouse=True)
    def _stub_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolve the identity from the HEADER path, as production does."""
        monkeypatch.setattr(
            interceptor_handler,
            "verified_identity_from_authorization",
            lambda _auth: VerifiedIdentity(
                served_scope=_SERVED_SCOPE, subject=_SUBJECT
            ),
        )

    def test_sts_receives_the_claim_and_header_values_not_the_body_values(
        self, fake_sts: _RecordingFakeSts, scoped_env: dict[str, str]
    ) -> None:
        """The body's planted identity fields are ignored entirely."""
        interceptor_handler.handler(self._event(), None)

        assert len(fake_sts.calls) == 1
        call = fake_sts.calls[0]

        assert call["SourceIdentity"] == _SUBJECT
        assert call["RoleSessionName"] == f"gw-{_GATEWAY_ID}"

        rendered = repr(call)
        for name, hostile_value in self._HOSTILE.items():
            assert hostile_value not in rendered, (
                f"a request-body value ({name}) reached the AssumeRole call — an "
                "injected model could then choose its own audit identity"
            )

    def test_body_cannot_suppress_the_session_name(
        self, fake_sts: _RecordingFakeSts, scoped_env: dict[str, str]
    ) -> None:
        """Removing the header, while the body still offers one, fails closed.

        This is the sharper half of the previous test: it is not enough that the
        body is ignored when a header is present — the body must not become a
        FALLBACK when the header is absent.
        """
        event = self._event()
        event["mcp"]["gatewayRequest"]["headers"].pop("Mcp-Session-Id")

        result = interceptor_handler.handler(event, None)

        assert fake_sts.calls == [], (
            "with no Gateway header the vend must fail closed, not fall back to the "
            "gateway_identifier the body supplied"
        )
        assert result["mcp"]["transformedGatewayResponse"]["body"]["result"][
            "isError"
        ] is True
        assert "transformedGatewayRequest" not in result["mcp"]


# ---------------------------------------------------------------------------
# 5. A missing subject fails closed at the handler
# ---------------------------------------------------------------------------


class TestMissingSubjectFailsClosed:
    """No subject, no credential — and no read."""

    @staticmethod
    def _event() -> dict[str, Any]:
        return {
            "mcp": {
                "gatewayRequest": {
                    "headers": {
                        "Authorization": "Bearer token",
                        "Mcp-Session-Id": _GATEWAY_ID,
                    },
                    "body": {
                        "jsonrpc": "2.0",
                        "id": 7,
                        "method": "tools/call",
                        "params": {
                            "name": "ReadDocument___read_document",
                            "arguments": {"doc_id": "PAY-001"},
                        },
                    },
                }
            }
        }

    def test_absent_subject_short_circuits_and_mints_nothing(
        self,
        fake_sts: _RecordingFakeSts,
        scoped_env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A token that verifies but carries no ``sub`` is refused.

        ``verified_identity_from_authorization`` returns None for this case — the
        same value an unresolvable scope produces — so the handler takes its
        existing fail-closed route and no document read occurs.
        """
        monkeypatch.setattr(
            interceptor_handler,
            "verified_identity_from_authorization",
            lambda _auth: None,
        )

        result = interceptor_handler.handler(self._event(), None)

        assert fake_sts.calls == [], "no credential may be minted without a subject"
        assert "transformedGatewayRequest" not in result["mcp"]
        response = result["mcp"]["transformedGatewayResponse"]
        assert response["body"]["result"]["isError"] is True
        assert response["body"]["id"] == 7

    def test_generic_message_discloses_nothing(
        self,
        fake_sts: _RecordingFakeSts,
        scoped_env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The refusal must not reveal WHY, at the scope or the subject level."""
        monkeypatch.setattr(
            interceptor_handler,
            "verified_identity_from_authorization",
            lambda _auth: None,
        )

        result = interceptor_handler.handler(self._event(), None)
        text = result["mcp"]["transformedGatewayResponse"]["body"]["result"][
            "content"
        ][0]["text"]

        for leak in ("sub", "subject", "SourceIdentity", _SERVED_SCOPE, _SUBJECT):
            assert leak not in text

    def test_jwt_claims_requires_a_subject(self) -> None:
        """The claim-level half of the same rule, without minting a signed token."""
        from interceptor.jwt_claims import _subject_from_claims

        assert _subject_from_claims({"sub": _SUBJECT}) == _SUBJECT
        assert _subject_from_claims({}) is None
        assert _subject_from_claims({"sub": ""}) is None
        assert _subject_from_claims({"sub": "   "}) is None
        assert _subject_from_claims({"sub": None}) is None
        assert _subject_from_claims({"sub": 12345}) is None
        assert _subject_from_claims({"sub": ["a"]}) is None


# ---------------------------------------------------------------------------
# 6. The IAM side — both trust policies must allow sts:SetSourceIdentity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_template() -> Template:
    """Synthesize the real shipped stack.

    ``sh.build_full_stack()`` returns ``(stack, template)``; only the template is
    needed here.

    Returns:
        The synthesized ``aws_cdk.assertions.Template``.
    """
    _stack, template = sh.build_full_stack()
    return template


def _role_by_name(template: Template, role_name: str) -> dict[str, Any]:
    """Return the ``AWS::IAM::Role`` properties whose ``RoleName`` matches.

    Args:
        template: The synthesized template.
        role_name: The frozen role name.

    Returns:
        The role's ``Properties`` mapping.
    """
    for resource in template.find_resources("AWS::IAM::Role").values():
        if resource["Properties"].get("RoleName") == role_name:
            return resource["Properties"]
    raise AssertionError(f"no AWS::IAM::Role named {role_name!r} in the template")


class TestTrustPolicyGrantsSetSourceIdentity:
    """Without the trust-side grant the vend fails outright, not silently."""

    @pytest.mark.parametrize(
        "role_name",
        ["DocumentsAccessRole", "DocumentsWriteRole"],
        ids=["read-role", "write-role"],
    )
    def test_trust_policy_allows_set_source_identity(
        self, full_template: Template, role_name: str
    ) -> None:
        """AWS: a role trust policy without it makes ``AssumeRole*`` fail."""
        trust = _role_by_name(full_template, role_name)["AssumeRolePolicyDocument"]
        statements = trust["Statement"]

        assert len(statements) == 1, (
            "exactly one trust statement — a second, unconditioned statement would "
            "re-open the untagged-assume path"
        )
        actions = statements[0]["Action"]
        actions = [actions] if isinstance(actions, str) else actions

        assert _SET_SOURCE_IDENTITY_ACTION in actions, (
            "sts:SetSourceIdentity must be allowed in the TRUST policy or the "
            "AssumeRole carrying SourceIdentity fails outright"
        )

    @pytest.mark.parametrize(
        "role_name",
        ["DocumentsAccessRole", "DocumentsWriteRole"],
        ids=["read-role", "write-role"],
    )
    def test_all_three_actions_share_the_conditioned_statement(
        self, full_template: Template, role_name: str
    ) -> None:
        """The scope-tag condition must still govern the assume itself.

        Splitting ``sts:AssumeRole`` into its own unconditioned statement to make
        room for the new action would re-open the untagged-assume hole, so the
        action set and the condition are pinned together.
        """
        trust = _role_by_name(full_template, role_name)["AssumeRolePolicyDocument"]
        statement = trust["Statement"][0]
        actions = statement["Action"]
        actions = [actions] if isinstance(actions, str) else actions

        assert set(actions) == {
            _ASSUME_ACTION,
            _TAG_SESSION_ACTION,
            _SET_SOURCE_IDENTITY_ACTION,
        }
        assert statement["Condition"]["StringLike"] == {"aws:RequestTag/scope": "*"}

    def test_interceptor_identity_policy_allows_set_source_identity(
        self, full_template: Template
    ) -> None:
        """The identity side is mandatory too — both sides or AccessDenied.

        ``grant_assume_role`` covers neither ``sts:TagSession`` nor
        ``sts:SetSourceIdentity``, so the interceptor needs an explicit grant. The
        statement is located by its ``Sid`` rather than by the function name,
        because the interceptor is a container-image Lambda that declares no
        explicit ``FunctionName``.
        """
        statements = [
            statement
            for policy in full_template.find_resources("AWS::IAM::Policy").values()
            for statement in policy["Properties"]["PolicyDocument"]["Statement"]
            if statement.get("Sid") == "TagScopedDocumentsSessions"
        ]

        assert len(statements) == 1, (
            "exactly one TagScopedDocumentsSessions statement is expected — it is "
            "the interceptor's identity-side grant for both session parameters"
        )
        actions = statements[0]["Action"]
        actions = [actions] if isinstance(actions, str) else actions

        assert set(actions) == {_TAG_SESSION_ACTION, _SET_SOURCE_IDENTITY_ACTION}, (
            "the interceptor must hold sts:SetSourceIdentity on the identity side "
            "as well, or the assume fails AccessDenied the moment SourceIdentity "
            "is passed"
        )

    def test_identity_grant_is_scoped_to_the_two_documents_roles(
        self, full_template: Template
    ) -> None:
        """The grant must not widen to every role in the account.

        ``sts:SetSourceIdentity`` on ``"*"`` would let the interceptor stamp any
        identity onto any assumable role, which is a bigger capability than the
        one this work item needs.
        """
        statement = next(
            statement
            for policy in full_template.find_resources("AWS::IAM::Policy").values()
            for statement in policy["Properties"]["PolicyDocument"]["Statement"]
            if statement.get("Sid") == "TagScopedDocumentsSessions"
        )

        resources = statement["Resource"]
        resources = [resources] if not isinstance(resources, list) else resources

        assert len(resources) == 2, "exactly the two scoped-role ARNs"
        assert "*" not in repr(resources), (
            "the source-identity grant must stay pinned to the two Documents roles"
        )
