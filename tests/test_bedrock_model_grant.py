"""Bedrock invoke-grant tests: the Runtime role may invoke ONE pinned model.

Covers the two halves of the model-pinning fix:

* unit — ``cdk/bedrock_model_access.py`` turns one configured model id into the
  correct least-privilege ARN set (inference profile + underlying foundation
  model conditioned on that profile), and refuses a wildcarded id; and
* synth — the REAL ``ScopedCredentialsStack`` template grants ``bedrock:InvokeModel*``
  on nothing but those pinned resources: no ``foundation-model/*`` and no
  ``inference-profile/*`` wildcard survives anywhere in the template.

The synth half is the regression guard that matters: the previous grant covered
every foundation model and every inference profile in the account, so a caller
who could reach the Runtime could pin an arbitrary (expensive) model.

Import-path note
----------------
The ``cdk/`` modules import each other flat (``from bedrock_model_access import
...``), so ``cdk/`` must be on ``sys.path``. This module prepends it exactly as
``tests/test_synth_config.py`` and ``tests/synth_helpers.py`` do.
"""

from __future__ import annotations

import os
import sys

import pytest

_CDK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cdk"
)
if _CDK_DIR not in sys.path:
    sys.path.insert(0, _CDK_DIR)

import bedrock_model_access as bma  # noqa: E402

from synth_helpers import build_full_stack, join_to_str  # noqa: E402

# The account/region the full-stack fixture synthesizes with.
_ACCOUNT = "123456789012"
_REGION = "us-east-1"


def _as_list(value) -> list:
    """Normalize a rendered ``Action``/``Resource`` value to a list.

    CDK collapses a single-element list to a bare string when rendering a policy
    statement, so assertions must tolerate both shapes.

    Args:
        value: A rendered statement field (string or list).

    Returns:
        The value as a list.
    """
    return value if isinstance(value, list) else [value]


# ---------------------------------------------------------------------------
# Unit: ARN derivation
# ---------------------------------------------------------------------------


def test_geo_profile_id_yields_profile_and_foundation_model_arns() -> None:
    """A ``us.``-prefixed id needs BOTH the profile and the bare model ARN."""
    profile_arn, model_arn = bma.model_resource_arns(
        "us.anthropic.claude-sonnet-4-6", _REGION, _ACCOUNT
    )

    assert profile_arn == (
        f"arn:aws:bedrock:{_REGION}:{_ACCOUNT}:"
        "inference-profile/us.anthropic.claude-sonnet-4-6"
    )
    # The foundation-model ARN drops the geo prefix and carries NO account
    # segment (foundation models are not account-scoped).
    assert model_arn == "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-6"


def test_bare_foundation_model_id_yields_no_profile_arn() -> None:
    """A non-geo id is a plain foundation model in the configured Region only."""
    profile_arn, model_arn = bma.model_resource_arns(
        "anthropic.claude-sonnet-4-6", _REGION, _ACCOUNT
    )

    assert profile_arn is None
    assert model_arn == (
        f"arn:aws:bedrock:{_REGION}::foundation-model/anthropic.claude-sonnet-4-6"
    )


@pytest.mark.parametrize(
    "prefix", ["us.", "eu.", "apac.", "au.", "jp.", "ca.", "global."]
)
def test_every_geo_prefix_is_recognised(prefix: str) -> None:
    """Each supported geography is split, not treated as part of the model id."""
    geo, model = bma.split_model_id(f"{prefix}anthropic.claude-sonnet-4-6")

    assert geo == prefix
    assert model == "anthropic.claude-sonnet-4-6"


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "   ",
        "*",
        "us.anthropic.*",
        "anthropic.claude?",
        "us.anthropic.claude sonnet",
    ],
)
def test_wildcard_or_blank_model_id_is_rejected(bad_id: str) -> None:
    """A wildcarded/blank config value must fail at synth, not widen the grant."""
    with pytest.raises(ValueError):
        bma.model_resource_arns(bad_id, _REGION, _ACCOUNT)


def test_profile_statement_pair_constrains_model_to_the_profile() -> None:
    """The foundation-model statement is conditioned on the profile ARN.

    That condition is what makes the Region wildcard on the foundation-model ARN
    safe: the model can only be invoked when routed through this one profile,
    never called directly in some other Region.
    """
    statements = bma.invoke_statements(
        "us.anthropic.claude-sonnet-4-6", _REGION, _ACCOUNT
    )
    assert len(statements) == 2

    rendered = [s.to_json() for s in statements]
    profile_stmt, model_stmt = rendered

    expected_profile = (
        f"arn:aws:bedrock:{_REGION}:{_ACCOUNT}:"
        "inference-profile/us.anthropic.claude-sonnet-4-6"
    )
    # CDK renders a single-element Resource/Action list as a bare string.
    assert _as_list(profile_stmt["Resource"]) == [expected_profile]
    assert "Condition" not in profile_stmt

    assert model_stmt["Condition"] == {
        "StringEquals": {"bedrock:InferenceProfileArn": expected_profile}
    }
    for statement in rendered:
        assert statement["Effect"] == "Allow"
        assert set(_as_list(statement["Action"])) == set(bma.INVOKE_ACTIONS)


def test_bare_model_yields_a_single_unconditioned_statement() -> None:
    """With no routing indirection one statement suffices."""
    statements = bma.invoke_statements(
        "anthropic.claude-sonnet-4-6", _REGION, _ACCOUNT
    )

    assert len(statements) == 1
    assert "Condition" not in statements[0].to_json()


# ---------------------------------------------------------------------------
# Synth: no wildcard Bedrock grant survives in the real template
# ---------------------------------------------------------------------------


def _bedrock_invoke_grants(template) -> list[tuple[str, dict | None]]:
    """Collect every ``(resource, condition)`` granted a ``bedrock:InvokeModel*`` action.

    Scans every ``AWS::IAM::Policy`` and every inline role policy in the
    template, so a wildcard re-introduced on ANY role (not just the Runtime's)
    fails these tests. The statement's ``Condition`` travels with each resource
    because a Region-wildcarded foundation-model ARN is only least-privilege
    while the ``bedrock:InferenceProfileArn`` condition is attached to it —
    dropping the condition alone would be a real widening.

    Args:
        template: The synthesized ``Template``.

    Returns:
        A flat list of ``(resource, condition)`` pairs, with CloudFormation
        intrinsics (``Ref``/``GetAtt`` inside an ``Fn::Join``) rendered to a
        placeholder.
    """
    documents = [
        resource["Properties"]["PolicyDocument"]
        for resource in template.find_resources("AWS::IAM::Policy").values()
    ]
    for role in template.find_resources("AWS::IAM::Role").values():
        for inline in role["Properties"].get("Policies", []):
            documents.append(inline["PolicyDocument"])

    grants: list[tuple[str, dict | None]] = []
    for document in documents:
        for statement in document.get("Statement", []):
            actions = _as_list(statement.get("Action", []))
            if not any(str(a).startswith("bedrock:InvokeModel") for a in actions):
                continue
            condition = statement.get("Condition")
            for value in _as_list(statement.get("Resource", [])):
                grants.append((join_to_str(value), condition))
    return grants


def test_synth_grants_bedrock_invoke_on_no_wildcard_resource() -> None:
    """No ``bedrock:InvokeModel*`` grant in the template may use a wildcard."""
    _, template = build_full_stack()

    grants = _bedrock_invoke_grants(template)
    assert grants, "expected at least one bedrock:InvokeModel grant in the template"

    for resource, _ in grants:
        assert not resource.endswith("foundation-model/*"), (
            f"wildcard foundation-model grant re-introduced: {resource}"
        )
        assert not resource.endswith("inference-profile/*"), (
            f"wildcard inference-profile grant re-introduced: {resource}"
        )
        assert "*" not in resource.rsplit("/", 1)[-1], (
            f"wildcard in the model portion of the grant: {resource}"
        )


def test_synth_region_wildcard_is_fenced_by_the_profile_condition() -> None:
    """Any Region-wildcarded grant must be confined to one inference profile.

    Keeping ``arn:aws:bedrock:*::foundation-model/<id>`` while dropping the
    ``bedrock:InferenceProfileArn`` condition would let the Runtime role invoke
    that model directly, in every Region, outside the profile — so the condition
    is asserted on the rendered template, not just on the helper's output.
    """
    _, template = build_full_stack()

    for resource, condition in _bedrock_invoke_grants(template):
        # ARN layout: arn:partition:service:region:account:resource
        region = resource.split(":")[3]
        if region != "*":
            continue
        assert condition, (
            f"Region-wildcarded grant carries no condition: {resource}"
        )
        fenced = condition.get("StringEquals", {}).get("bedrock:InferenceProfileArn")
        assert fenced, (
            "Region-wildcarded grant is not fenced by bedrock:InferenceProfileArn: "
            f"{resource} (condition={condition})"
        )
        assert ":inference-profile/" in join_to_str(fenced)


def test_synth_grants_bedrock_invoke_only_on_the_configured_model() -> None:
    """Every granted resource names the model id resolved from config."""
    from shared.config_loader import load_config

    _, template = build_full_stack()
    configured = load_config().bedrock_model_id
    _, foundation_model_id = bma.split_model_id(configured)

    for resource, _ in _bedrock_invoke_grants(template):
        assert resource.endswith(
            (f"inference-profile/{configured}", f"foundation-model/{foundation_model_id}")
        ), f"unexpected Bedrock resource granted: {resource}"
