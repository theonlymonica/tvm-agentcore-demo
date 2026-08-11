"""Scope session tag — the SECOND enforcement gate.

WHAT THIS PINS
--------------
Before this gate existed, the two scoped roles granted their DynamoDB actions on
the WHOLE table with no partition constraint, and tenant separation lived
ENTIRELY in the per-request inline session policy the interceptor passes to
``AssumeRole``. One enforcement point: any assume that omitted that policy
yielded full cross-tenant read/write of every partition, including
``infra-secrets`` and ``hr-data``.

The fix adds a second, independent gate that lives OUTSIDE the caller's code
path — ABAC on a session tag:

1. The interceptor passes ``Tags=[{"Key": "scope", "Value": served_scope}]`` on
   every ``AssumeRole`` (``interceptor/scoped_credentials.py``).
2. Each role's identity policy requires ``dynamodb:LeadingKeys`` to match
   ``${aws:PrincipalTag/scope}`` (``cdk/documents_roles.py``), so the ROLE
   itself cannot touch a foreign partition even with no session policy at all.
3. Each role's TRUST policy requires that tag to be present
   (``cdk/lambda_iam.py``), so an untagged assume mints nothing.

Each numbered claim above has tests below, in that order, plus the two invariants
that hold the three files together: the tag key literal is identical across the
bundle boundary, and the tag value always equals the session policy's
``LeadingKeys`` value (a disagreement would deny everything).

WHY THESE ASSERTIONS AND NOT OTHERS
-----------------------------------
The negative test ``test_every_dynamodb_allow_is_tag_conditioned`` is the one
that survives refactoring: it walks BOTH roles' inline policy documents and fails
on ANY ``Allow`` statement carrying a ``dynamodb:`` action without the
``aws:PrincipalTag/scope`` condition. A future statement added without the
condition re-opens the original single-enforcement-point weakness, and this test
is what catches it.

NOT VERIFIED HERE (deliberately)
--------------------------------
These are synthesis-time and unit-level assertions. Nothing here proves AWS
actually denies a cross-partition read for a tagged session — that needs a live
account.

AWS documentation grounding:
    - Session tags land in the request context as ``aws:PrincipalTag/<key>`` and
      are usable in policy conditions; ``sts:TagSession`` must be allowed in the
      role's trust policy or ``AssumeRole`` fails:
      https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html
    - A multivalued context key (``dynamodb:LeadingKeys``) compared against a
      policy VARIABLE requires the ``StringLike`` operator:
      https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-single-vs-multi-valued-context-keys.html
    - ``ForAllValues`` matches vacuously when the request key is absent, which is
      why the ``Null`` presence guard is asserted alongside it:
      https://docs.aws.amazon.com/IAM/latest/UserGuide/access-analyzer-reference-policy-checks.html
    - ``Tags`` is a list of ``{Key, Value}``; values are capped at 256 characters:
      https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pytest

from interceptor import scoped_credentials
from interceptor.scoped_credentials import (
    READ_ACTIONS,
    SCOPE_TAG_KEY,
    ScopeTagError,
    vend_scoped_credentials,
)

# ---------------------------------------------------------------------------
# Import-path setup: put cdk/ on sys.path so the flat cdk imports resolve
# (same idiom as tests/test_synth_config.py).
# ---------------------------------------------------------------------------

_CDK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cdk"
)
if _CDK_DIR not in sys.path:
    sys.path.insert(0, _CDK_DIR)

import aws_cdk as cdk  # noqa: E402
import aws_cdk.aws_lambda as lambda_  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402

import documents_roles  # noqa: E402
import lambda_iam  # noqa: E402
from data_resources import create_data_resources  # noqa: E402

import synth_helpers as sh  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen expectations
# ---------------------------------------------------------------------------

_TABLE_ARN = "arn:aws:dynamodb:us-east-1:123456789012:table/DocumentsTable"
_ROLE_ARN = "arn:aws:iam::123456789012:role/DocumentsAccessRole"
_SERVED_SCOPE = "payments-core"

#: The policy variable the identity policies must compare LeadingKeys against.
_PRINCIPAL_TAG_VARIABLE = "${aws:PrincipalTag/scope}"
#: The trust-policy condition key that must be required present.
_REQUEST_TAG_KEY = "aws:RequestTag/scope"

_ASSUME_ACTION = "sts:AssumeRole"
_TAG_SESSION_ACTION = "sts:TagSession"

_FAKE_STS_CREDENTIALS: dict[str, Any] = {
    "AccessKeyId": "ASIAIOSFODNN7EXAMPLE",
    "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "SessionToken": "IQoJb3JpZ2luX2VjEXAMPLETOKEN",
    "Expiration": datetime(2099, 1, 1, tzinfo=timezone.utc),
}


class _RecordingFakeSts:
    """Stand-in STS client recording every ``assume_role`` call's kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def assume_role(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D401
        """Record the kwargs and return fixed temporary credentials."""
        self.calls.append(kwargs)
        return {"Credentials": _FAKE_STS_CREDENTIALS}


@pytest.fixture
def fake_sts(monkeypatch: pytest.MonkeyPatch) -> _RecordingFakeSts:
    """Patch the interceptor's ``boto3.client`` to a recording fake STS.

    Returns:
        The fake, so a test can inspect the recorded ``assume_role`` kwargs.
    """
    sts = _RecordingFakeSts()
    monkeypatch.setattr(
        scoped_credentials.boto3,
        "client",
        lambda service_name, *a, **k: sts,
    )
    return sts


# ---------------------------------------------------------------------------
# Claim 1 — the interceptor tags every vended session
# ---------------------------------------------------------------------------


class TestVendPassesScopeTag:
    """``vend_scoped_credentials`` always tags the session with the scope."""

    def test_assume_receives_exactly_one_scope_tag(
        self, fake_sts: _RecordingFakeSts
    ) -> None:
        """``Tags`` is the single ``{"Key": "scope", "Value": served_scope}`` pair."""
        vend_scoped_credentials(_ROLE_ARN, _SERVED_SCOPE, _TABLE_ARN, READ_ACTIONS)

        assert len(fake_sts.calls) == 1, "AssumeRole must be called exactly once"
        assert fake_sts.calls[0]["Tags"] == [
            {"Key": "scope", "Value": _SERVED_SCOPE}
        ], "the vended session must carry exactly the scope session tag"

    def test_transitive_tag_keys_not_set(self, fake_sts: _RecordingFakeSts) -> None:
        """The tag must NOT be transitive — the session never chains onward."""
        vend_scoped_credentials(_ROLE_ARN, _SERVED_SCOPE, _TABLE_ARN, READ_ACTIONS)

        assert "TransitiveTagKeys" not in fake_sts.calls[0], (
            "a transitive tag would survive a further AssumeRole; the vended "
            "session is terminal and must not propagate its scope"
        )

    def test_session_policy_still_passed_alongside_the_tag(
        self, fake_sts: _RecordingFakeSts
    ) -> None:
        """The tag ADDS a gate; it does not replace the session policy."""
        vend_scoped_credentials(_ROLE_ARN, _SERVED_SCOPE, _TABLE_ARN, READ_ACTIONS)

        assert fake_sts.calls[0]["Policy"], (
            "the inline session policy must still be passed — the two gates are "
            "ANDed (the session is the intersection of both), not alternatives"
        )

    @pytest.mark.parametrize(
        "scope",
        [
            pytest.param("payments-core", id="known-scope"),
            pytest.param("billing-internal", id="other-known-scope"),
            pytest.param("infra-secrets", id="foreign-partition"),
            pytest.param("a", id="single-char"),
            pytest.param("x" * 256, id="max-length-value"),
        ],
    )
    def test_tag_value_agrees_with_session_policy_leading_keys(
        self, fake_sts: _RecordingFakeSts, scope: str
    ) -> None:
        """Tag value and ``LeadingKeys`` value must be the SAME string.

        They are compared against each other at authorization time (the identity
        policy requires ``LeadingKeys == ${aws:PrincipalTag/scope}``), so any
        divergence denies every request. Pinning the agreement here keeps a future
        edit from silently bricking the data path.
        """
        vend_scoped_credentials(_ROLE_ARN, scope, _TABLE_ARN, READ_ACTIONS)

        call = fake_sts.calls[0]
        policy = json.loads(call["Policy"])
        leading_keys = policy["Statement"][0]["Condition"][
            "ForAllValues:StringEquals"
        ]["dynamodb:LeadingKeys"]

        assert call["Tags"][0]["Value"] == scope
        assert leading_keys == [call["Tags"][0]["Value"]], (
            "session-policy LeadingKeys and the session tag must carry the "
            "identical scope string"
        )


class TestUntaggableScopeFailsClosed:
    """A scope that cannot be safely tagged never reaches STS."""

    @pytest.mark.parametrize(
        "scope",
        [
            pytest.param("", id="empty"),
            pytest.param("*", id="bare-wildcard"),
            pytest.param("payments-*", id="suffix-wildcard"),
            pytest.param("*-core", id="prefix-wildcard"),
            pytest.param("payments?core", id="single-char-wildcard"),
            pytest.param("x" * 257, id="over-sts-value-limit"),
        ],
    )
    def test_wildcard_or_oversized_scope_raises_and_mints_nothing(
        self, fake_sts: _RecordingFakeSts, scope: str
    ) -> None:
        """Reject before assuming — a wildcard tag would WIDEN the ABAC condition.

        The identity policy compares ``LeadingKeys`` with ``StringLike`` (required
        for a multivalued key compared against a variable), so ``scope=*`` would
        match EVERY partition. The value is allowlisted upstream from
        ``cognito:groups``, but that allowlist is operator-supplied via
        ``KNOWN_SCOPE_GROUPS``, so the rejection is enforced at the point of use.
        """
        with pytest.raises(ScopeTagError):
            vend_scoped_credentials(_ROLE_ARN, scope, _TABLE_ARN, READ_ACTIONS)

        assert fake_sts.calls == [], (
            "AssumeRole must not be reached: no credential may be minted for a "
            "scope that cannot be safely expressed as a session tag"
        )

    def test_scope_tag_error_is_caught_by_the_handler_fail_closed_path(self) -> None:
        """``ScopeTagError`` must subclass a type the handler already catches.

        ``interceptor/handler.py`` catches ``(ClientError, BotoCoreError, KeyError,
        RuntimeError)`` on the vend path and returns the generic short-circuit
        error. Subclassing ``RuntimeError`` is what routes a rejected tag into
        that existing fail-closed path instead of escaping as a 500.
        """
        assert issubclass(ScopeTagError, RuntimeError)


# ---------------------------------------------------------------------------
# Cross-boundary invariant — the three copies of the tag key must agree
# ---------------------------------------------------------------------------


def test_tag_key_literal_is_identical_across_the_bundle_boundary() -> None:
    """Interceptor and CDK must name the same tag key.

    They cannot share a constant: the interceptor Lambda asset excludes ``cdk/``.
    A drift here fails closed (the ``Null`` guard denies everything) but would
    take the whole data path down, so it is pinned by test instead.
    """
    assert SCOPE_TAG_KEY == documents_roles.SCOPE_TAG_KEY == "scope"
    assert lambda_iam._REQUIRE_SCOPE_TAG_CONDITION["StringLike"] == {
        f"aws:RequestTag/{SCOPE_TAG_KEY}": "*"
    }


# ---------------------------------------------------------------------------
# Claims 2 and 3 — synthesized IAM shape
# ---------------------------------------------------------------------------


def _stub_lambda(scope: cdk.Stack, construct_id: str) -> lambda_.Function:
    """Create a zip-packaged PYTHON_3_14 stub Lambda (no Docker build).

    Args:
        scope: The stack to attach the function to.
        construct_id: Unique construct id.

    Returns:
        A minimal ``lambda_.Function`` standing in for the Docker-image interceptor.
    """
    return lambda_.Function(
        scope,
        construct_id,
        runtime=lambda_.Runtime.PYTHON_3_14,  # the pinned Python runtime
        handler="index.handler",
        code=lambda_.Code.from_inline("def handler(event, context):\n    return {}\n"),
    )


@pytest.fixture(scope="module")
def abac_template() -> Template:
    """Synthesize the real data layer + real IAM wiring on a minimal stack.

    Invokes the production ``create_data_resources`` (which builds both scoped
    roles) and the production ``wire_lambda_iam`` (which grants the assume/tag
    permissions and replaces both trust documents), substituting a zip stub for
    the Docker-image interceptor. The IAM shapes under test come entirely from
    production code.

    Returns:
        The synthesized ``aws_cdk.assertions.Template``.
    """
    app = cdk.App()
    stack = cdk.Stack(
        app,
        "TestScopeTagAbacStack",
        env=cdk.Environment(account="123456789012", region="us-east-1"),
    )
    data = create_data_resources(stack)
    interceptor_fn = _stub_lambda(stack, "StubInterceptor")
    lambda_iam.wire_lambda_iam(data, interceptor_fn)
    return Template.from_stack(stack)


def _role_by_name(template: Template, role_name: str) -> dict[str, Any]:
    """Return the ``AWS::IAM::Role`` properties whose ``RoleName`` matches.

    Args:
        template: The synthesized template.
        role_name: The frozen role name (``DocumentsAccessRole`` /
            ``DocumentsWriteRole``).

    Returns:
        The role's ``Properties`` mapping.
    """
    for resource in template.find_resources("AWS::IAM::Role").values():
        if resource["Properties"].get("RoleName") == role_name:
            return resource["Properties"]
    raise AssertionError(f"no AWS::IAM::Role named {role_name!r} in the template")


def _data_allow_statements(role_properties: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect every ``Allow`` statement carrying a DynamoDB action.

    Args:
        role_properties: A role's ``Properties`` mapping.

    Returns:
        The matching statements from every inline policy document.
    """
    statements: list[dict[str, Any]] = []
    for inline in role_properties.get("Policies", []):
        for statement in inline["PolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            actions = [actions] if isinstance(actions, str) else actions
            if statement.get("Effect") == "Allow" and any(
                action.startswith("dynamodb:") for action in actions
            ):
                statements.append(statement)
    return statements


@pytest.mark.parametrize(
    "role_name",
    [
        pytest.param("DocumentsAccessRole", id="read-role"),
        pytest.param("DocumentsWriteRole", id="write-role"),
    ],
)
class TestIdentityPolicyIsTagConditioned:
    """Each scoped role confines itself to its session's tagged partition."""

    def test_every_dynamodb_allow_is_tag_conditioned(
        self, abac_template: Template, role_name: str
    ) -> None:
        """THE regression guard: no unconditioned data-plane Allow may exist.

        This assertion is what guarantees that EVERY DynamoDB grant the role
        carries is confined to the session's tagged partition. A statement added
        later without the condition restores the original single-enforcement-point
        weakness, and this test fails on it.
        """
        statements = _data_allow_statements(_role_by_name(abac_template, role_name))
        assert statements, f"{role_name} must grant at least one DynamoDB action"

        for statement in statements:
            condition = statement.get("Condition", {})
            assert condition.get("ForAllValues:StringLike", {}).get(
                "dynamodb:LeadingKeys"
            ) == [_PRINCIPAL_TAG_VARIABLE], (
                f"{role_name} statement {statement.get('Sid')!r} must confine "
                "dynamodb:LeadingKeys to ${aws:PrincipalTag/scope}"
            )

    def test_null_guards_close_the_forallvalues_footgun(
        self, abac_template: Template, role_name: str
    ) -> None:
        """Both presence guards must be present.

        ``ForAllValues`` matches VACUOUSLY when the request key is absent, so
        without ``Null`` on ``dynamodb:LeadingKeys`` a keyless request would pass.
        The second guard denies an UNTAGGED session at the data plane even if the
        trust-policy tag requirement were relaxed.
        """
        for statement in _data_allow_statements(
            _role_by_name(abac_template, role_name)
        ):
            null_block = statement.get("Condition", {}).get("Null", {})
            assert null_block.get("dynamodb:LeadingKeys") == "false"
            assert null_block.get(f"aws:PrincipalTag/{SCOPE_TAG_KEY}") == "false"

    def test_condition_uses_stringlike_not_stringequals(
        self, abac_template: Template, role_name: str
    ) -> None:
        """``StringEquals`` with a variable on a multivalued key is not supported.

        AWS requires ``StringLike`` when a multivalued context key is compared
        against a variable; ``ForAllValues:StringEquals`` here would be a silent
        mis-evaluation rather than a synthesis error.
        """
        for statement in _data_allow_statements(
            _role_by_name(abac_template, role_name)
        ):
            condition = statement.get("Condition", {})
            assert "ForAllValues:StringEquals" not in condition, (
                "a policy VARIABLE on the multivalued dynamodb:LeadingKeys key "
                "requires the StringLike operator"
            )

    def test_trust_policy_requires_the_scope_tag(
        self, abac_template: Template, role_name: str
    ) -> None:
        """An untagged assume must be rejected before any credential is minted."""
        trust = _role_by_name(abac_template, role_name)["AssumeRolePolicyDocument"]
        statements = trust["Statement"]

        assert len(statements) == 1, (
            "exactly one trust statement — a second, unconditioned statement "
            "would re-open the untagged-assume path"
        )
        statement = statements[0]
        actions = statement["Action"]
        actions = [actions] if isinstance(actions, str) else actions

        assert set(actions) == {_ASSUME_ACTION, _TAG_SESSION_ACTION}, (
            "sts:TagSession must be allowed in the TRUST policy or the tagged "
            "AssumeRole fails; sts:AssumeRole must share the statement so the "
            "tag condition governs the assume itself"
        )
        assert statement["Condition"]["StringLike"] == {_REQUEST_TAG_KEY: "*"}, (
            "the trust statement must require a scope session tag to be present"
        )

    def test_placeholder_account_principal_does_not_ship(
        self, abac_template: Template, role_name: str
    ) -> None:
        """The temporary ``AccountPrincipal`` trust must be fully replaced."""
        trust = _role_by_name(abac_template, role_name)["AssumeRolePolicyDocument"]
        assert "root" not in json.dumps(trust["Statement"][0]["Principal"]), (
            "the AccountPrincipal placeholder (…:root) must not appear in the "
            "shipped trust document"
        )


class TestInterceptorRoleCanTagSessions:
    """The identity side of the tagging requirement."""

    def test_interceptor_role_holds_assume_and_tag_session(
        self, abac_template: Template
    ) -> None:
        """``grant_assume_role`` does NOT cover ``sts:TagSession``.

        Both actions must be granted to the interceptor exec role, or every vend
        fails ``AccessDenied`` the moment ``Tags`` is passed.
        """
        functions = sh.lambda_functions(abac_template)
        stub = next(
            resource
            for logical_id, resource in functions.items()
            if logical_id.startswith("StubInterceptor")
        )
        role_logical_id = sh.function_role_logical_id(stub)
        actions = sh.iam_actions_targeting_role(abac_template, role_logical_id)

        assert _ASSUME_ACTION in actions
        assert _TAG_SESSION_ACTION in actions
