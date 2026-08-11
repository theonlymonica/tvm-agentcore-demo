"""
Lambda IAM wiring for per-request tenant scope enforcement.

This module finalizes the least-privilege IAM wiring for the tool Lambdas. It
implements the SECOND half of the two-step scoped-role contract begun in
``cdk/documents_roles.py`` (which creates ``DocumentsAccessRole`` /
``DocumentsWriteRole`` with a temporary ``AccountPrincipal`` placeholder trust
because the tool execution roles do not exist yet at data-layer synthesis time).

What this module does:

- **Identity side — grant the exec roles ``sts:AssumeRole``.** Each read tool's
  execution role (``read_document`` / ``search_documents``) is granted
  ``sts:AssumeRole`` on ``DocumentsAccessRole``; the ``reply`` execution role is
  granted ``sts:AssumeRole`` on ``DocumentsWriteRole``. This uses
  ``iam.Role.grant_assume_role(...)``, which per the CDK docs "grants permissions
  to the given principal to assume this role" — i.e. it adds an identity-based
  ``sts:AssumeRole`` statement to the GRANTEE (the exec role), scoped to the target
  role's ARN. It does NOT modify the target role's trust policy.

- **Trust side — narrow each scoped role's trust to ONLY its intended exec
  roles.** Because ``grant_assume_role`` only touches the grantee's identity policy
  (verified in the CDK docs, see reference below), the scoped roles still carry the
  temporary ``AccountPrincipal`` placeholder trust that ``documents_roles.py`` set
  via ``assumed_by``. That placeholder is broader than the intended
  named-principal trust and MUST NOT ship. We therefore REPLACE each role's
  ``AssumeRolePolicyDocument`` wholesale via the L1 (``CfnRole``) property override
  so the shipped trust policy names EXACTLY the intended exec roles and nothing
  else. A property override fully replaces the constructor-generated trust document,
  so the ``AccountPrincipal`` placeholder does not appear in the synthesized
  template.

Both sides are required for an ``AssumeRole`` to succeed: the exec role must hold
the ``sts:AssumeRole`` permission (identity side) AND the scoped role's trust policy
must name it as a principal (trust side) — see the AWS troubleshooting note that an
assume fails "because no role trust policy allows the sts:AssumeRole action".

The tool execution roles (read AND write) are granted NO direct DynamoDB read or
write permission on ``DocumentsTable`` — every data-plane access happens only
through the vended scoped session. This module adds only the ``sts:AssumeRole``
grants and the trust narrowing; it adds no DynamoDB grants.

AWS documentation references:
    - ``aws_cdk.aws_iam.Role.grant_assume_role`` ("Grant permissions to the given
      principal to assume this role" — grants the GRANTEE identity ``sts:AssumeRole``
      on this role; does not edit the role's trust policy):
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_iam/Role.html
    - ``sts:AssumeRole`` IAM action + the requirement that the role trust policy
      allow it for the principal (both the identity grant and the trust principal
      are needed):
      https://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html
    - IAM ``Principal`` element — role ARNs as trust principals:
      https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_principal.html
    - ``aws_cdk.aws_lambda.Function.grant_principal`` / ``role`` (the execution role
      used as the assume-role grantee / trust principal):
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_lambda/Function.html
    - L1 escape hatch (``CfnRole.AssumeRolePolicyDocument`` property override to
      replace the trust document; property overrides take precedence over
      constructor-generated values):
      https://docs.aws.amazon.com/cdk/v2/guide/cfn_layer.html

Functions:
    wire_lambda_iam: Grant the exec roles ``sts:AssumeRole`` on the scoped roles and
        narrow each scoped role's trust policy to only those exec roles.
"""

from __future__ import annotations

from typing import Any

import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as lambda_

from data_resources import DataResources
from documents_roles import SCOPE_TAG_KEY

#: IAM action for assuming a role (verified action string; used in the trust
#: documents built below). The identity-side grant uses the same action via
#: ``grant_assume_role``.
_ASSUME_ROLE_ACTION = "sts:AssumeRole"

#: IAM action for PASSING SESSION TAGS on that assume. This is a SEPARATE action
#: from ``sts:AssumeRole`` and must be allowed on BOTH sides: in the caller's
#: identity policy (granted in ``wire_lambda_iam``) and in the target role's TRUST
#: policy (``_trust_document``). AWS: "trust policies for all roles connected to
#: the identity provider (IdP) passing tags must have the sts:TagSession
#: permission. For roles without this permission in the trust policy, the
#: AssumeRole operation fails"
#: (https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html).
_TAG_SESSION_ACTION = "sts:TagSession"

#: Trust-policy condition requiring that a ``scope`` session tag be PRESENT on
#: the assume request — the gate that makes the ABAC identity-policy condition in
#: ``documents_roles.py`` unskippable. Without it a caller could assume the role
#: with no tag at all; the identity policy's ``Null`` guard would then deny every
#: data request, which fails closed but only AFTER a credential was minted.
#: Requiring the tag at the trust boundary rejects the assume itself.
#:
#: Shape follows the AWS ABAC tutorial, which mandates required tags on the
#: ``sts:TagSession`` statement using ``StringLike`` with ``"*"``
#: (https://docs.aws.amazon.com/IAM/latest/UserGuide/tutorial_abac-saml.html).
#: ``Null: {"aws:RequestTag/scope": "false"}`` is an equivalent documented
#: formulation; the tutorial's form is used because it is the shape AWS publishes
#: for exactly this purpose.
_REQUIRE_SCOPE_TAG_CONDITION: dict[str, Any] = {
    "StringLike": {f"aws:RequestTag/{SCOPE_TAG_KEY}": "*"}
}


def _trust_document(exec_role_arns: list[str]) -> dict[str, Any]:
    """Build an ``AssumeRolePolicyDocument`` trusting exactly the given exec roles.

    Produces the intended trust-policy shape: a single ``Allow`` statement whose
    ``Principal.AWS`` lists the tool execution-role ARNs and whose ``Action`` is
    ``sts:AssumeRole`` plus ``sts:TagSession``. Used to REPLACE (via L1 override)
    the temporary ``AccountPrincipal`` placeholder trust so the shipped policy
    names only the intended principals.

    The statement is CONDITIONED on a ``scope`` session tag being present
    (``_REQUIRE_SCOPE_TAG_CONDITION``). Two consequences, both intended:

    - An ``AssumeRole`` WITHOUT ``Tags=[{"Key": "scope", ...}]`` is denied at the
      trust boundary and no credential is minted at all. This is what stops the
      "future refactor / added caller / exception path skips policy construction"
      scenario from silently yielding table-wide cross-tenant access.
    - Both actions live in ONE statement (the shape AWS publishes in its ABAC
      tutorial) so the condition governs the ASSUME itself, not only the tagging.
      A separate, unconditioned ``sts:AssumeRole`` statement would leave the
      original hole wide open.

    Args:
        exec_role_arns: The execution-role ARNs allowed to assume the scoped role.

    Returns:
        The trust policy document as a plain dict (for a ``CfnRole`` property
        override).
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": exec_role_arns},
                "Action": [_ASSUME_ROLE_ACTION, _TAG_SESSION_ACTION],
                "Condition": _REQUIRE_SCOPE_TAG_CONDITION,
            }
        ],
    }


def _replace_trust_policy(role: Any, exec_role_arns: list[str]) -> None:
    """Replace a role's trust document, dropping the placeholder ``AccountPrincipal``.

    Overrides the L1 ``CfnRole.AssumeRolePolicyDocument`` property. A property
    override fully replaces the constructor-generated trust document (the
    ``assumed_by`` placeholder), guaranteeing the temporary ``AccountPrincipal`` set
    in ``documents_roles.py`` does NOT ship — the synthesized trust names exactly
    ``exec_role_arns``.

    Args:
        role: The scoped ``iam.Role`` construct whose trust to narrow.
        exec_role_arns: The execution-role ARNs to name as the sole trust principals.
    """
    cfn_role = role.node.default_child
    cfn_role.add_property_override(
        "AssumeRolePolicyDocument",
        _trust_document(exec_role_arns),
    )


def wire_lambda_iam(
    data: DataResources,
    interceptor_fn: lambda_.Function,
) -> None:
    """Wire the scoped-role assumption grants and finalize the scoped-role trust.

    The REQUEST INTERCEPTOR — not the tool Lambdas — assumes the scoped roles and
    vends credentials to the tools. Therefore:

    1. **Identity side** — grant ``sts:AssumeRole`` on BOTH ``DocumentsAccessRole``
       (read) and ``DocumentsWriteRole`` (write) to the INTERCEPTOR exec role (via
       ``Role.grant_assume_role``), PLUS ``sts:TagSession`` on the same two role
       ARNs (a separate action that ``grant_assume_role`` does not cover; without
       it the tagged assume fails ``AccessDenied``).
    2. **Trust side** — replace each scoped role's ``AssumeRolePolicyDocument`` so
       it trusts ONLY the interceptor exec role, removing the temporary
       ``AccountPrincipal`` placeholder that ``documents_roles.py`` created, and
       allows ``sts:AssumeRole``/``sts:TagSession`` ONLY when a ``scope`` session
       tag is present — so an untagged assume mints nothing.

    The TOOL execution roles (read_document / search_documents / reply) are granted
    NOTHING here: they hold NO ``sts:AssumeRole`` and NO DynamoDB permission (only
    the default Lambda logging role). They can reach the table ONLY with the
    credentials the interceptor vends into their event — a compromised tool cannot
    mint or widen any credential (verified by the compromised-tool test:
    ``sts:AssumeRole`` from a tool exec role -> AccessDenied).

    Both sides are required for an ``AssumeRole`` to succeed: the interceptor role
    must hold the ``sts:AssumeRole`` permission (identity side) AND each scoped
    role's trust policy must name it as a principal (trust side).

    Args:
        data: The data-layer resources, exposing the scoped role constructs
            (``documents_access_role`` / ``documents_write_role``).
        interceptor_fn: The REQUEST interceptor Lambda (the sole assumer of the
            scoped roles).
    """
    # 1. Identity side: grant the INTERCEPTOR exec role sts:AssumeRole on BOTH
    #    scoped roles. grant_assume_role adds the sts:AssumeRole permission to the
    #    GRANTEE (interceptor exec role) scoped to the target role ARN.
    data.documents_access_role.grant_assume_role(interceptor_fn.grant_principal)
    data.documents_write_role.grant_assume_role(interceptor_fn.grant_principal)

    # 1b. Identity side: sts:TagSession is a SEPARATE action from sts:AssumeRole
    #     and grant_assume_role does NOT include it, so passing Tags on the assume
    #     would fail AccessDenied without this statement. It is scoped to exactly
    #     the two scoped-role ARNs — the interceptor cannot tag a session on any
    #     other role. The trust side of the same requirement is in _trust_document
    #     (both sides are mandatory).
    interceptor_fn.add_to_role_policy(
        iam.PolicyStatement(
            sid="TagScopedDocumentsSessions",
            effect=iam.Effect.ALLOW,
            actions=[_TAG_SESSION_ACTION],
            resources=[
                data.documents_access_role_arn,
                data.documents_write_role_arn,
            ],
        )
    )

    # 2. Trust side: narrow each scoped role's trust to ONLY the interceptor exec
    #    role, replacing the temporary AccountPrincipal placeholder so it never
    #    ships. The tool exec roles are intentionally NOT trust principals.
    interceptor_role_arn = interceptor_fn.role.role_arn
    _replace_trust_policy(data.documents_access_role, [interceptor_role_arn])
    _replace_trust_policy(data.documents_write_role, [interceptor_role_arn])
