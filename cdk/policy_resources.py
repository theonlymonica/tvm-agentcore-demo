"""
AgentCore Policy Engine and Cedar policy resources.

This module provisions:
- A PolicyEngine (alpha construct)
- Three Cedar policies (one per tool: read_document, search_documents,
  reply), each loaded from cedar/ with the gateway ARN substituted

The module is split into two functions to support proper dependency ordering:
1. create_policy_engine() — creates the engine (no gateway reference needed)
2. add_cedar_policies() — creates policies referencing the gateway ARN

The policy engine is attached to the gateway via the L1 escape hatch in
gateway_resources.py (the stable Gateway L2 does not accept
policy_engine_configuration directly). The escape hatch + explicit CDK
dependency ensures CloudFormation creates the IAM grants before the
Gateway resource.

AWS documentation references:
  - "Using Policy with Stable Gateway" (L1 escape hatch):
    https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_bedrock_agentcore_alpha/README.html
  - PolicyEngine, Policy, PolicyValidationMode (alpha):
    https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_bedrock_agentcore_alpha/README.html
"""

from __future__ import annotations

import os
from pathlib import Path

import aws_cdk as cdk
import aws_cdk.aws_bedrockagentcore as agentcore
import aws_cdk.aws_bedrock_agentcore_alpha as agentcore_alpha
from constructs import Construct


def create_policy_engine(scope: Construct) -> agentcore_alpha.PolicyEngine:
    """Create the Policy Engine (no policies, no gateway reference).

    This function creates only the PolicyEngine resource. Cedar policies
    are added separately via add_cedar_policies() after the gateway exists.

    Args:
        scope: The CDK Stack or Construct to attach resources to.

    Returns:
        The provisioned PolicyEngine construct.
    """
    policy_engine = agentcore_alpha.PolicyEngine(
        scope,
        "PolicyEngine",
        policy_engine_name="support_policy_engine",
        description=(
            "Cedar policy engine — one permit policy per tool, ENFORCE mode"
        ),
    )

    # Export the policy engine ARN for observability / debugging
    cdk.CfnOutput(
        scope,
        "PolicyEngineArn",
        value=policy_engine.policy_engine_arn,
        description="Policy Engine ARN",
    )

    return policy_engine


def add_cedar_policies(
    scope: Construct,
    policy_engine: agentcore_alpha.PolicyEngine,
    gateway: agentcore.Gateway,
) -> None:
    """Add the three Cedar policies that reference gateway.gateway_arn.

    Each policy is ordered after the gateway AND after every one of its tool
    targets: a policy whose ``action`` names a target should not be created
    before that target exists. This ordering was also a precondition for the
    abandoned ``FAIL_ON_ANY_FINDINGS`` attempt — see the ``validation_mode``
    comment in ``_create_cedar_policy`` for why that deploy-time check is gone.
    There is NO deploy-time validation of the action string; the only check is at
    synth, in ``tests/test_cedar_policy_actions.py``.

    Args:
        scope: The CDK Stack or Construct to attach resources to.
        policy_engine: The PolicyEngine to attach the policies to.
        gateway: The AgentCore Gateway (for gateway_arn substitution).

    Raises:
        ValueError: If the stack declares no gateway targets. Every policy names
            a target, so a target-less stack is a wiring error; failing at synth
            is cheaper and clearer than shipping policies whose actions can
            resolve to nothing.
    """
    cedar_dir = Path(os.path.dirname(__file__)).parent / "cedar"
    gateway_arn = gateway.gateway_arn

    # Create policies with explicit dependency on the gateway and all its
    # targets, so a policy is never created before the target its action names.
    # This is creation-order hygiene, NOT a validation gate: with
    # IGNORE_ALL_FINDINGS there is no deploy-time check of the action string.
    policies = []

    policies.append(_create_cedar_policy(
        scope=scope,
        construct_id="ReadDocumentPolicy",
        policy_engine=policy_engine,
        policy_name="read_document",
        cedar_file=cedar_dir / "read_document.cedar",
        gateway_arn=gateway_arn,
    ))

    policies.append(_create_cedar_policy(
        scope=scope,
        construct_id="SearchDocumentsPolicy",
        policy_engine=policy_engine,
        policy_name="search_documents",
        cedar_file=cedar_dir / "search_documents.cedar",
        gateway_arn=gateway_arn,
    ))

    policies.append(_create_cedar_policy(
        scope=scope,
        construct_id="ReplyPolicy",
        policy_engine=policy_engine,
        policy_name="reply",
        cedar_file=cedar_dir / "reply.cedar",
        gateway_arn=gateway_arn,
    ))

    # Add explicit dependencies: each Policy must wait for the gateway AND for
    # ALL of its tool targets.
    #
    # The targets are NOT children of the gateway construct — gateway_resources
    # creates them with the STACK as their scope and the gateway passed only as
    # the `gateway=` property (GatewayTarget.for_lambda(scope, ..., gateway=...)),
    # so the gateway's children are just its ServiceRole and Resource. Depending
    # on the gateway alone therefore orders a policy BEFORE the targets it names,
    # which is what forced validation to be suppressed. They are located in the
    # construct tree instead of being threaded through create_gateway's return
    # value so no existing call site or logical id changes.
    targets = [
        child
        for child in cdk.Stack.of(gateway).node.find_all()
        if isinstance(child, agentcore.GatewayTarget)
    ]
    if not targets:
        raise ValueError(
            "no AgentCore GatewayTarget found in the stack: every Cedar policy "
            "names a target, and a policy whose action names no registered "
            "target silently denies that tool at request time"
        )

    for policy in policies:
        policy.node.add_dependency(gateway)
        for target in targets:
            policy.node.add_dependency(target)


def _create_cedar_policy(
    scope: Construct,
    construct_id: str,
    policy_engine: agentcore_alpha.PolicyEngine,
    policy_name: str,
    cedar_file: Path,
    gateway_arn: str,
) -> agentcore_alpha.Policy:
    """Load a Cedar policy file, substitute the gateway ARN, and create the Policy.

    Args:
        scope: The CDK Stack or Construct.
        construct_id: Unique CDK construct id for this policy.
        policy_engine: The PolicyEngine to attach the policy to.
        policy_name: Human-readable name for the policy.
        cedar_file: Path to the .cedar file on disk.
        gateway_arn: The resolved gateway ARN to substitute into the policy.

    Returns:
        The provisioned Policy construct.
    """
    raw_cedar = cedar_file.read_text(encoding="utf-8")

    # Replace the synth-time placeholder with the actual gateway ARN.
    # The cedar files use <gateway-arn> as the placeholder.
    definition = raw_cedar.replace("<gateway-arn>", gateway_arn)

    return agentcore_alpha.Policy(
        scope,
        construct_id,
        policy_engine=policy_engine,
        policy_name=policy_name,
        definition=definition,
        description=f"Cedar permit policy for the {policy_name} tool",
        # IGNORE_ALL_FINDINGS is REQUIRED here, and the reason is structural
        # rather than a suppression of convenience.
        #
        # Every policy in cedar/ leaves `principal` unconstrained on purpose:
        # these are per-TOOL allow-lists, not the tenant boundary. The Cedar
        # policies are not the tenant boundary; the boundary is the scoped
        # credential, which lives in interceptor/handler.py,
        # interceptor/scoped_credentials.py and cdk/documents_roles.py. The
        # Policy Engine cannot know that, so it reports the unconstrained
        # principal as an "Overly Permissive" finding:
        #
        #   Overly Permissive: Policy Engine will allow every request for the
        #   specified principal (AgentCore::IamEntity), action (Reply___reply)
        #   and resource (...gateway/...) combination if the policy is added or
        #   updated
        #
        # The finding is CORRECT on its own terms — that is exactly what the
        # policy grants — and it is the documented design. But
        # PolicyValidationMode has only two values, so FAIL_ON_ANY_FINDINGS
        # cannot fail on a misspelled action while tolerating that advisory: it
        # makes the deliberate design decision fatal too, and NO deploy carrying
        # these policies can ever succeed. Verified live: a deploy with
        # FAIL_ON_ANY_FINDINGS aborts on ReplyPolicy with the message above and
        # rolls the stack back.
        #
        # The guard FAIL_ON_ANY_FINDINGS was reached for — catching an action
        # string that resolves to no registered gateway target, which would
        # silently deny the tool — is kept, and kept EARLIER: at synth, by
        # tests/test_cedar_policy_actions.py::test_actions_match_the_gateway_targets,
        # whose set comparison catches both an action naming no target and a
        # target named by no action. Losing the deploy-time check is a real
        # reduction in defence-in-depth; the enum leaves no way to keep both, and
        # a check that runs before the deploy is the better half to keep.
        #
        # Do NOT "fix" this by constraining `principal` to satisfy the validator:
        # that would restate the tenant boundary in a second place with no added
        # enforcement, and contradicts the reasoning in every cedar/*.cedar file.
        validation_mode=(
            agentcore_alpha.PolicyValidationMode.IGNORE_ALL_FINDINGS
        ),
    )
