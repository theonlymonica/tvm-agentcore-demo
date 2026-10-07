"""Credential-header allowlist for the Lambda tool targets (F9 header channel).

The REQUEST interceptor propagates the authoritative scope and the vended STS
credentials as four custom request headers. The Gateway forwards a propagated
header to a target ONLY if that target allowlists it in
``metadataConfiguration.allowedRequestHeaders``; anything else is dropped. This
module owns that allowlist and the escape hatch that applies it.

Extracted from ``cdk/gateway_resources.py`` to keep that module small.

Constants:
    CREDENTIAL_REQUEST_HEADERS: the four header names every tool target allowlists.

Functions:
    allowlist_credential_headers: apply the allowlist to one L2 GatewayTarget.
"""

from __future__ import annotations

from typing import Any

#: The propagated request headers EVERY Lambda tool target must allowlist, so the
#: REQUEST interceptor's credential headers reach the tool.
#:
#: These literals MUST equal ``interceptor/credential_headers.CREDENTIAL_HEADERS``
#: and the tool-side names in ``tools/common/credentials_context.py``. They are
#: duplicated rather than imported because the CDK app runs from ``cdk/`` with its
#: own virtualenv and neither package is on its import path;
#: ``tests/test_header_contract_parity.py`` pins all three lists to each other, so
#: a header added on one side and forgotten here breaks the suite.
#:
#: Getting this wrong fails CLOSED but confusingly: the Gateway DROPS any
#: interceptor-supplied header a target has not allowlisted, so a typo produces a
#: tool that reports "propagated credential context is missing" rather than a
#: visible configuration error.
CREDENTIAL_REQUEST_HEADERS: tuple[str, ...] = (
    "x-tvm-served-scope",
    "x-tvm-access-key-id",
    "x-tvm-secret-access-key",
    "x-tvm-session-token",
)


def allowlist_credential_headers(target: Any, logical_name: str) -> None:
    """Allowlist the interceptor's credential headers on one Lambda tool target.

    Why an escape hatch: the L2 ``GatewayTarget.for_lambda`` props
    (``GatewayTargetLambdaProps``) do NOT include ``metadata_configuration`` —
    only ``for_api_gateway`` exposes it in this aws-cdk-lib build. The L1
    ``AWS::BedrockAgentCore::GatewayTarget`` DOES support it, so the property is
    written straight onto the underlying ``CfnGatewayTarget``. CloudFormation
    declares ``MetadataConfiguration`` as an UPDATE-WITH-NO-INTERRUPTION property,
    so adding it does not replace the target and the Cedar action names
    (``ReadDocument___read_document`` and its siblings) are untouched.

    Observed, and worth knowing before changing this: the SERVICE maintains its own
    ``x-amzn-bedrock-agentcore-policy-session-id`` entry on every target's
    allowlist, and this override COMPOSES with it rather than replacing it — the
    live allowlist after deploy carries that entry plus these four, while the
    synthesized template declares only these four. That behaviour is NOT
    documented; it was observed on the live gateway. It matters because an override
    that replaced the list would silently break the policy engine's session-id
    propagation. Budget: 4 here + 1 service entry = 5 of the Gateway's limit of 10
    allowlisted headers per target.

    Args:
        target: The L2 ``GatewayTarget`` whose underlying L1 resource is patched.
        logical_name: The construct id, used only in the error message.

    Raises:
        RuntimeError: If the L2 construct exposes no default child. Raised at
            SYNTH time rather than skipping the override silently, because a
            missing allowlist makes the Gateway drop the credential headers and
            every call to that tool fails closed for a reason that looks like a
            tool bug.

    References (AWS Documentation MCP server, per the ``aws-docs-lookup`` rule):
        - MetadataConfiguration property + no-interruption update behaviour:
          https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-bedrockagentcore-gatewaytarget.md
        - AllowedRequestHeaders (1-10 items, 1-100 chars each):
          https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-bedrockagentcore-gatewaytarget-metadataconfiguration.md
        - Header propagation, allowlist merge, interceptor-over-client precedence:
          https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-headers.html
    """
    cfn_target = target.node.default_child
    if cfn_target is None:  # pragma: no cover - synth-time guard
        raise RuntimeError(
            f"{logical_name} has no default child; cannot set "
            f"MetadataConfiguration.AllowedRequestHeaders, so the interceptor's "
            f"credential headers would be dropped by the Gateway"
        )
    cfn_target.add_property_override(
        "MetadataConfiguration.AllowedRequestHeaders",
        list(CREDENTIAL_REQUEST_HEADERS),
    )
