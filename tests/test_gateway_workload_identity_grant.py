"""The gateway execution role can mint a workload access token.

With a Policy Engine attached, the gateway propagates the caller's session
identity by minting a Workload Access Token (WAT) before it invokes a target. On
the IAM inbound flow that mint calls ``bedrock-agentcore:GetWorkloadAccessToken``
against the gateway's own workload identity, so the grant is required IN ADDITION
to the three policy permissions (``GetPolicyEngine``, ``AuthorizeAction``,
``PartiallyAuthorizeActions``).

Why this file exists
--------------------
The grant was missing from the CDK entirely and the stack only worked because an
equivalent statement had been added to the role BY HAND, outside CloudFormation.
The first deploy that rewrote the role's managed ``DefaultPolicy`` silently
removed it, and every ``tools/call`` then failed — with a failure mode that is
close to undiagnosable from the outside:

* the caller sees only ``An internal error occurred. Please retry later.``;
* the Cedar evaluation still returns ``ALLOW``, so the policy engine looks fine;
* the target Lambda is never invoked, so its CloudWatch log group stays EMPTY;
* the request interceptor succeeds, so the vending path looks healthy too.

The real error surfaces only in the gateway's own ``APPLICATION_LOGS`` delivery,
which is not enabled by default. A test is therefore the cheapest place to keep
this from regressing.

Citation:
  - AgentCore Gateway and Policy IAM permissions, "IAM permissions for temporal
    policies" — gives this statement verbatim (sid
    ``PolicySessionWorkloadIdentity``) and states that without it tool
    invocations fail at the token-mint step:
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-permissions.html
"""

from __future__ import annotations

import json

import pytest
from aws_cdk.assertions import Template

from synth_helpers import build_full_stack

#: The action the gateway needs to mint a workload access token.
_WAT_ACTION = "bedrock-agentcore:GetWorkloadAccessToken"


@pytest.fixture(scope="module")
def synth_template() -> Template:
    """Synthesize the full stack once.

    Returns:
        The synthesized ``aws_cdk.assertions.Template``.
    """
    _stack, template = build_full_stack()
    return template


def _gateway_role_statements_granting(
    template: Template, action: str
) -> list[dict]:
    """Return statements granting ``action`` on the GATEWAY's role only.

    Scoping to the gateway role is essential rather than cosmetic: the agent
    runtime's execution role also grants ``GetWorkloadAccessToken`` (plus the
    ``ForJWT`` / ``ForUserId`` variants), so an unscoped search finds that one
    and passes even when the gateway grant is absent — a vacuous test that would
    not have caught the very regression this file exists for.

    Args:
        template: The synthesized template.
        action: The IAM action to look for.

    Returns:
        The matching statement dicts from policies attached to the gateway role.
    """
    found: list[dict] = []
    for resource in template.find_resources("AWS::IAM::Policy").values():
        props = resource["Properties"]
        roles = json.dumps(props.get("Roles", []))
        if "SupportGatewayServiceRole" not in roles:
            continue
        for stmt in props["PolicyDocument"].get("Statement", []):
            actions = stmt.get("Action")
            actions = [actions] if isinstance(actions, str) else (actions or [])
            if action in actions:
                found.append(stmt)
    return found


class TestGatewayWorkloadIdentityGrant:
    """The WAT mint is granted to the gateway role, and scoped, not wildcarded."""

    def test_the_grant_is_present_on_the_gateway_role(
        self, synth_template: Template
    ) -> None:
        stmts = _gateway_role_statements_granting(synth_template, _WAT_ACTION)
        assert stmts, (
            f"the gateway execution role does not grant {_WAT_ACTION}; with a "
            "policy engine attached EVERY tools/call fails at the token-mint "
            "step and the caller sees only a generic internal error (see module "
            "docstring)"
        )

    def test_the_grant_is_not_wildcarded(self, synth_template: Template) -> None:
        # A "*" resource would work, but this stack's posture is to scope every
        # grant. The docs give two concrete ARNs.
        for stmt in _gateway_role_statements_granting(synth_template, _WAT_ACTION):
            resources = stmt.get("Resource")
            resources = (
                [resources] if not isinstance(resources, list) else resources
            )
            assert "*" not in resources, (
                f"{_WAT_ACTION} must not be granted on '*'; scope it to the "
                "workload-identity directory and the gateway's identity"
            )

    def test_the_grant_targets_the_workload_identity_directory(
        self, synth_template: Template
    ) -> None:
        # The rendered resources carry CloudFormation intrinsics (the gateway id
        # is a deploy-time token), so assert on the serialized form.
        blobs = [
            json.dumps(stmt.get("Resource"))
            for stmt in _gateway_role_statements_granting(
                synth_template, _WAT_ACTION
            )
        ]
        assert any("workload-identity-directory/default" in b for b in blobs), (
            f"{_WAT_ACTION} is granted, but not on the workload-identity "
            f"directory: {blobs!r}"
        )
        assert any("workload-identity/" in b for b in blobs), (
            f"{_WAT_ACTION} must also cover the gateway's own child identity "
            f"(workload-identity/<gatewayId>*): {blobs!r}"
        )
