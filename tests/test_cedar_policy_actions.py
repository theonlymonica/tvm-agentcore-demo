"""The Cedar action strings must name real gateway targets.

Why this module exists
----------------------
Every ``action ==`` string in ``cedar/*.cedar`` must equal a composite gateway
target name (``{targetName}___{toolName}``); a mismatch means the policy permits
an action no caller can invoke, so the real tool is denied by default — a failure
that is safe but silent. ``tests/test_synth_config.py`` freezes those names on the
*gateway* side only, and nothing tied the two sides together.

The Policy Engine does check this itself, but only at deploy time, and only under
``PolicyValidationMode.FAIL_ON_ANY_FINDINGS`` — which turned out to be unusable
here: every ``cedar/`` policy leaves ``principal`` unconstrained by design, the
Policy Engine reports that as an "Overly Permissive" finding, and the mode has no
setting that fails on a bad action while tolerating that advisory, so it fails
every deploy. The deploy-time check is therefore gone and these assertions are
the ONLY thing standing between a misspelled action and a silently denied tool.
They run at synth, where a mismatch costs seconds rather than a failed deploy,
and they run against the **synthesized** template, so the ``<gateway-arn>``
substitution is covered too.

Nothing here says the Cedar policies confine a tenant — they do not. The Cedar
policies are not the tenant boundary, and each policy's header explains why; the
boundary itself is covered by ``tests/test_session_policy.py`` and
``tests/test_scope_tag_abac.py``.

The fixture builds a minimal stack from the production factories with stub zip
Lambdas (the real ``ScopedCredentialsStack`` would trigger two container builds that
neither the policies nor the targets depend on), mirroring
``tests/test_synth_config.py`` — including its precedent of keeping the stub
factory and minimal-stack fixture local, since ``tests/synth_helpers.py`` is
scoped to the full-stack helpers. ``cdk/`` goes on ``sys.path`` because the
``cdk/`` modules import each other flat and ``conftest.py`` does not add it.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Import-path setup: put cdk/ on sys.path so the flat cdk imports resolve.
# ---------------------------------------------------------------------------

_CDK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cdk"
)
if _CDK_DIR not in sys.path:
    sys.path.insert(0, _CDK_DIR)

import aws_cdk as cdk  # noqa: E402
import aws_cdk.aws_lambda as lambda_  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402

from auth_resources import create_auth_resources  # noqa: E402
from data_resources import create_data_resources  # noqa: E402
from gateway_resources import create_gateway  # noqa: E402
from policy_resources import (  # noqa: E402
    add_cedar_policies,
    create_policy_engine,
)

# ---------------------------------------------------------------------------
# Frozen values.
# ---------------------------------------------------------------------------

_POLICY_TYPE = "AWS::BedrockAgentCore::Policy"
_TARGET_TYPE = "AWS::BedrockAgentCore::GatewayTarget"
_GATEWAY_TYPE = "AWS::BedrockAgentCore::Gateway"

#: Validation is SUPPRESSED, and must stay suppressed: every cedar/ policy leaves
#: `principal` unconstrained by design, which the Policy Engine reports as an
#: "Overly Permissive" finding. PolicyValidationMode has only two values, so
#: FAIL_ON_ANY_FINDINGS makes that deliberate design decision fatal and no deploy
#: can succeed (verified live — it aborts on ReplyPolicy and rolls back). The
#: action/target check it was reached for lives in
#: `test_actions_match_the_gateway_targets` below, at synth. See the long comment
#: on `validation_mode` in cdk/policy_resources.py.
_EXPECTED_VALIDATION_MODE = "IGNORE_ALL_FINDINGS"

#: The synth-time placeholder ``_create_cedar_policy`` substitutes away. It MUST
#: still be present in the source files and MUST NOT survive into the template.
_GATEWAY_ARN_PLACEHOLDER = "<gateway-arn>"

#: ``action == AgentCore::Action::"<composite tool name>"``. Cedar ``//`` comments
#: are stripped first, so a comment mentioning an action cannot satisfy it.
_ACTION_RE = re.compile(r'action\s*==\s*AgentCore::Action::"([^"]+)"')

_CEDAR_DIR = Path(__file__).resolve().parent.parent / "cedar"

#: One policy file per tool. Named rather than globbed so a deleted file is a
#: failure, not a silently smaller parametrization. Used ONLY for the source-file
#: assertions — the synthesized policy count is derived from the gateway targets.
_CEDAR_FILES = ("read_document.cedar", "search_documents.cedar", "reply.cedar")


# ---------------------------------------------------------------------------
# Minimal-stack synthesis (module-scoped: synthesize once, assert many).
# ---------------------------------------------------------------------------


def _stub_lambda(scope: cdk.Stack, construct_id: str) -> lambda_.Function:
    """Create a zip-packaged PYTHON_3_14 stub Lambda (no Docker build).

    Args:
        scope: The stack to attach the function to.
        construct_id: Unique construct id for the function.

    Returns:
        A minimal ``lambda_.Function`` construct.
    """
    return lambda_.Function(
        scope,
        construct_id,
        runtime=lambda_.Runtime.PYTHON_3_14,  # the runtime this project pins
        handler="index.handler",
        code=lambda_.Code.from_inline("def handler(event, context):\n    return {}\n"),
    )


@pytest.fixture(scope="module")
def policy_template() -> Template:
    """Synthesize gateway + Cedar policies once and return the template.

    Returns:
        The synthesized ``aws_cdk.assertions.Template``.
    """
    app = cdk.App()
    stack = cdk.Stack(
        app,
        "TestCedarPolicyStack",
        env=cdk.Environment(account="123456789012", region="us-east-1"),
    )

    create_data_resources(stack)
    auth = create_auth_resources(stack)
    policy_engine = create_policy_engine(scope=stack)

    gateway = create_gateway(
        scope=stack,
        interceptor_fn=_stub_lambda(stack, "StubInterceptor"),
        policy_engine=policy_engine,
        auth=auth,
        tool_fns={
            "read_document": _stub_lambda(stack, "StubReadDocument"),
            "search_documents": _stub_lambda(stack, "StubSearchDocuments"),
            "reply": _stub_lambda(stack, "StubReply"),
        },
    )

    # The subject under test: the real production wiring of the .cedar files.
    add_cedar_policies(scope=stack, policy_engine=policy_engine, gateway=gateway)

    return Template.from_stack(stack)


# ---------------------------------------------------------------------------
# Template helpers.
# ---------------------------------------------------------------------------


def _literal_text(node: Any) -> str:
    """Concatenate the literal string parts of a (possibly ``Fn::Join``) value.

    A policy definition embeds the gateway ARN as an ``Fn::GetAtt`` token, so
    CloudFormation renders the Cedar statement as an ``Fn::Join``. Only the
    literal segments carry the policy text; intrinsic nodes are skipped.

    Args:
        node: A template value — string, list, or intrinsic dict.

    Returns:
        The concatenated literal text (unresolved tokens contribute nothing).
    """
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_literal_text(item) for item in node)
    if isinstance(node, dict):
        join = node.get("Fn::Join")
        if join is not None:
            delimiter, parts = join
            return delimiter.join(_literal_text(part) for part in parts)
        return ""
    return ""


def _has_gateway_arn_getatt(node: Any, gateway_logical_id: str) -> bool:
    """Report whether the value pins THIS gateway's ``GatewayArn`` attribute.

    The logical id is checked, not just the attribute name: in a multi-gateway
    stack a policy that resolved some *other* gateway's ARN would otherwise
    satisfy this, and the policy would authorize the tool on the wrong gateway.

    Args:
        node: A template value — string, list, or intrinsic dict.
        gateway_logical_id: Logical id of the gateway the policy must pin.

    Returns:
        ``True`` when an ``Fn::GetAtt`` on that gateway's ``GatewayArn`` appears
        anywhere below.
    """
    if isinstance(node, list):
        return any(_has_gateway_arn_getatt(item, gateway_logical_id) for item in node)
    if isinstance(node, dict):
        get_att = node.get("Fn::GetAtt")
        if isinstance(get_att, list) and get_att == [gateway_logical_id, "GatewayArn"]:
            return True
        return any(
            _has_gateway_arn_getatt(value, gateway_logical_id)
            for value in node.values()
        )
    return False


def _gateway_logical_id(template: Template) -> str:
    """Return the logical id of the stack's sole gateway resource.

    Args:
        template: The synthesized template.

    Returns:
        The gateway's CloudFormation logical id.
    """
    gateways = template.find_resources(_GATEWAY_TYPE)
    assert len(gateways) == 1, f"expected exactly one gateway, got {sorted(gateways)!r}"
    return next(iter(gateways))


def _policy_resources(template: Template) -> dict[str, dict]:
    """Return each Cedar policy resource, keyed by policy name.

    Args:
        template: The synthesized template.

    Returns:
        Mapping of ``Properties.Name`` to the whole resource dict (so callers can
        read ``DependsOn`` as well as ``Properties``).
    """
    policies = template.find_resources(_POLICY_TYPE)
    by_name: dict[str, dict] = {}
    for resource in policies.values():
        by_name[resource["Properties"]["Name"]] = resource
    assert len(by_name) == len(policies), "policy names must be unique"
    return by_name


def _policy_statements(template: Template) -> dict[str, Any]:
    """Return each Cedar policy's statement value, keyed by policy name.

    Args:
        template: The synthesized template.

    Returns:
        Mapping of ``Properties.Name`` to the raw ``Definition.Cedar.Statement``.
    """
    return {
        name: resource["Properties"]["Definition"]["Cedar"]["Statement"]
        for name, resource in _policy_resources(template).items()
    }


def _composite_target_names(template: Template) -> set[str]:
    """Return the gateway's composite tool names (``{target}___{tool}``).

    Args:
        template: The synthesized template.

    Returns:
        The set of composite names the gateway publishes.
    """
    composites: set[str] = set()
    for resource in template.find_resources(_TARGET_TYPE).values():
        props = resource["Properties"]
        inline = props["TargetConfiguration"]["Mcp"]["Lambda"]["ToolSchema"][
            "InlinePayload"
        ]
        assert len(inline) == 1, f"expected one inline tool definition, got {inline}"
        composites.add(f"{props['Name']}___{inline[0]['Name']}")
    return composites


def _strip_comments(cedar_source: str) -> str:
    """Remove Cedar line comments (``//`` to end of line) from policy source.

    Args:
        cedar_source: The raw contents of a ``.cedar`` file.

    Returns:
        The source with every ``//`` comment removed.
    """
    return "\n".join(line.split("//", 1)[0] for line in cedar_source.splitlines())


# ===========================================================================
# Cedar action strings vs gateway targets
# ===========================================================================


class TestCedarSourceFiles:
    """Each ``.cedar`` file declares exactly one action and one placeholder."""

    @pytest.mark.parametrize("filename", _CEDAR_FILES)
    def test_file_declares_exactly_one_action(self, filename: str) -> None:
        source = (_CEDAR_DIR / filename).read_text(encoding="utf-8")
        actions = _ACTION_RE.findall(_strip_comments(source))
        assert len(actions) == 1, (
            f"cedar/{filename} must declare exactly one action, got {actions!r}"
        )

    @pytest.mark.parametrize("filename", _CEDAR_FILES)
    def test_file_uses_the_gateway_arn_placeholder(self, filename: str) -> None:
        # _create_cedar_policy substitutes this literal; a policy that hardcodes
        # an ARN (or misspells the placeholder) would pin the wrong gateway.
        source = _strip_comments(
            (_CEDAR_DIR / filename).read_text(encoding="utf-8")
        )
        assert _GATEWAY_ARN_PLACEHOLDER in source, (
            f"cedar/{filename} must reference the resource as "
            f"{_GATEWAY_ARN_PLACEHOLDER!r} for substitution"
        )


class TestSynthesizedPolicies:
    """The synthesized policies match the gateway targets, in both directions."""

    def test_one_policy_per_gateway_target(self, policy_template: Template) -> None:
        # Counted against the synthesized targets, NOT against a frozen list: a
        # legitimate fourth tool should then need no bookkeeping edit here, and
        # an unprotected tool still fails.
        statements = _policy_statements(policy_template)
        targets = _composite_target_names(policy_template)
        assert statements, "no Cedar policies were synthesized at all"
        assert len(statements) == len(targets), (
            f"expected one Cedar policy per gateway target ({len(targets)}), got "
            f"{sorted(statements)!r}"
        )

    def test_validation_mode_is_suppressed(self, policy_template: Template) -> None:
        # Pinned so the flip back to FAIL_ON_ANY_FINDINGS cannot happen by
        # accident: it makes the deliberately unconstrained `principal` a fatal
        # finding and no deploy carrying these policies can succeed. The
        # action/target check is covered by
        # test_actions_match_the_gateway_targets, at synth.
        for name, resource in _policy_resources(policy_template).items():
            mode = resource["Properties"]["ValidationMode"]
            assert mode == _EXPECTED_VALIDATION_MODE, (
                f"policy {name!r} must use {_EXPECTED_VALIDATION_MODE}, got "
                f"{mode!r}"
            )

    def test_policies_are_ordered_after_every_target(
        self, policy_template: Template
    ) -> None:
        # This ordering still holds on its own terms: the targets are NOT
        # children of the gateway construct, so depending on the gateway alone
        # lets a policy be created before the target its action names. The
        # ordering no longer serves a validation mode (see
        # _EXPECTED_VALIDATION_MODE), but creating a policy after the target it
        # references is the right order regardless.
        target_ids = set(policy_template.find_resources(_TARGET_TYPE))
        assert target_ids, "no gateway targets were synthesized at all"
        for name, resource in _policy_resources(policy_template).items():
            missing = target_ids - set(resource.get("DependsOn", []))
            assert not missing, (
                f"policy {name!r} is not ordered after target(s) "
                f"{sorted(missing)!r}; with validation enabled it can be created "
                "before the target its action names and fail the deploy"
            )

    def test_actions_match_the_gateway_targets(
        self, policy_template: Template
    ) -> None:
        # THE assertion of this module. Set equality covers both directions — an
        # action naming no target (dead policy, tool denied by default) and a
        # target named by no action (unprotected tool).
        policy_actions: set[str] = set()
        for name, statement in _policy_statements(policy_template).items():
            found = _ACTION_RE.findall(_strip_comments(_literal_text(statement)))
            assert len(found) == 1, (
                f"policy {name!r} must declare exactly one action, got {found!r}"
            )
            policy_actions.add(found[0])

        # Guard against the vacuous pass: two empty sets compare equal, so a
        # renamed CFN resource type would otherwise satisfy the comparison.
        assert policy_actions, "no Cedar actions found in the synthesized policies"

        assert policy_actions == _composite_target_names(policy_template), (
            "every Cedar action must name a gateway target and every target must "
            "be named by a policy"
        )

    def test_gateway_arn_placeholder_is_substituted(
        self, policy_template: Template
    ) -> None:
        gateway_id = _gateway_logical_id(policy_template)
        for name, statement in _policy_statements(policy_template).items():
            text = _literal_text(statement)
            assert _GATEWAY_ARN_PLACEHOLDER not in text, (
                f"policy {name!r} still carries the unsubstituted "
                f"{_GATEWAY_ARN_PLACEHOLDER!r} placeholder"
            )
            assert _has_gateway_arn_getatt(statement, gateway_id), (
                f"policy {name!r} must pin {gateway_id}'s GatewayArn via GetAtt"
            )
