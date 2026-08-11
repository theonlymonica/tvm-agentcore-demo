"""
Post-creation AgentCore Gateway wiring via AwsCustomResource.

This module holds the two ``AwsCustomResource``-based helpers that wire a
Gateway AFTER it (and its IAM role policy) are fully created:

- ``attach_policy_engine`` — calls the ``bedrock-agentcore-control``
  ``UpdateGateway`` API to attach the PolicyEngine, avoiding the
  ``GetPolicyEngine`` "Access denied" race. Its ``UpdateGateway`` payload
  re-declares the authorizer, so it mirrors the CUSTOM_JWT authorizer set at
  creation time — otherwise the update would revert the gateway to ``AWS_IAM``.
  It also re-declares ``interceptorConfigurations`` in full, so it MUST
  re-declare the RESPONSE interceptor alongside the REQUEST entry whenever one
  is wired, or it would silently drop it. This is the second RESPONSE
  declaration point (the first is ``_build_interceptor_configs`` at creation);
  both are driven by the single ``response_interceptor_fn`` so they cannot
  diverge.
- ``enable_gateway_tracing`` — builds the CloudWatch Logs delivery pipeline
  (source -> destination -> delivery) so the gateway's Cedar decision spans
  reach ``aws/spans``.

These helpers live here rather than in ``gateway_resources.py`` to keep that
module within its line budget. Both are orthogonal to the inbound-auth
configuration; they are grouped together because both are single-SDK-call
custom-resource wiring performed against the gateway.

AWS documentation references:
  - UpdateGateway with Policy Engine (service docs):
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/update-gateway-with-policy.html
  - CUSTOM_JWT customJWTAuthorizer shape (discoveryUrl, allowedClients):
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-create-api.html
  - AwsCustomResource (CDK custom_resources):
    https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.custom_resources/AwsCustomResource.html
  - Gateway TRACES delivery / observability:
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html
"""

from __future__ import annotations

import aws_cdk.aws_bedrockagentcore as agentcore
import aws_cdk.aws_bedrock_agentcore_alpha as agentcore_alpha
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as lambda_
import aws_cdk.custom_resources as cr
from aws_cdk import ArnFormat, Stack
from constructs import Construct

# The managed Cognito identity references. The UpdateGateway payload
# re-declares the authorizer, so it consumes the SAME AuthResources object as the
# creation-time CUSTOM_JWT authorizer (single source of truth).
from auth_resources import AuthResources


def enable_gateway_tracing(
    scope: Construct,
    gateway: agentcore.Gateway,
) -> None:
    """Enable gateway TRACES delivery to CloudWatch aws/spans.

    Creates the CloudWatch Logs delivery pipeline (source -> destination ->
    delivery) so the gateway's Cedar authorization decision spans reach the
    aws/spans log group. Requires CloudWatch Transaction Search to be enabled
    (handled in runtime_resources.py).

    Implemented as three chained AwsCustomResource constructs because each
    AwsCustomResource performs a single SDK call.

    The ``service`` string ("CloudWatchLogs") matches the value used for the
    other CloudWatch Logs calls in runtime_resources.py for consistency.

    AWS documentation reference:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html

    Args:
        scope: The CDK Stack or Construct to attach resources to.
        gateway: The Gateway whose traces are delivered.
    """
    # -----------------------------------------------------------------------
    # CREDENTIAL-IN-PAYLOAD GUARD — DO NOT enable gateway APPLICATION_LOGS
    # request-body delivery.
    #
    # The REQUEST interceptor vends short-lived STS credentials to the tools
    # inside `params.arguments`, because a Lambda target has no header channel.
    # The gateway's vended APPLICATION_LOGS carry a `requestBody` field that
    # includes `params.arguments` VERBATIM
    # (observability-gateway-metrics.html), so enabling APPLICATION_LOGS
    # delivery would write those credentials into
    # /aws/vendedlogs/bedrock-agentcore/gateway/APPLICATION_LOGS/{gateway_id}.
    # This is not theoretical: a real run once created an APPLICATION_LOGS
    # delivery source and vended credentials reached CloudWatch.
    #
    # This module therefore delivers ONLY `TRACES` (below); vended spans do NOT
    # include arguments. Two mechanical layers back the prohibition up, so it
    # does not rest on this comment alone:
    #
    #   1. tests/test_synth_log_delivery.py asserts against the SYNTHESIZED
    #      template that every PutDeliverySource declares logType=TRACES, that
    #      exactly one source exists and targets this gateway, and that the
    #      literal APPLICATION_LOGS appears nowhere. Adding an APPLICATION_LOGS
    #      delivery here breaks the test suite.
    #   2. The `logs:PutDeliverySource` grant below is pinned to the single
    #      `source_name` delivery-source ARN, so this deploy-time role cannot
    #      create an ADDITIONAL, differently-named APPLICATION_LOGS source.
    #
    # IAM cannot express the constraint directly: `logs:PutDeliverySource`
    # supports only tag condition keys plus `logs:LogGeneratingResourceArns`
    # (service-authorization reference for CloudWatch Logs), so there is no
    # `logType` condition key and therefore no policy or SCP that denies
    # APPLICATION_LOGS while permitting TRACES. Two residual paths an operator
    # can still take are neither prevented nor detected here: a console change,
    # or a same-name PutDeliverySource that OVERWRITES this source's logType.
    # Credential-in-payload is inherent to the AWS reference "Design 2" for a
    # Lambda target; the exposure window is bounded by the session policy's
    # 60-second DateLessThan on aws:CurrentTime
    # (interceptor/scoped_credentials.py _SESSION_POLICY_TTL_SECONDS), NOT by
    # the 900s STS DurationSeconds — that is only the session floor.
    # -----------------------------------------------------------------------
    source_name = "scoped-credentials-gateway-traces-source"
    dest_name = "scoped-credentials-gateway-traces-dest"

    # The ARN of the ONE delivery source this stack may create or update.
    # PutDeliverySource takes the `delivery-source` resource type, so the grant
    # can be name-pinned:
    # arn:<partition>:logs:<region>:<account>:delivery-source:<name>
    traces_source_arn = Stack.of(scope).format_arn(
        service="logs",
        resource="delivery-source",
        resource_name=source_name,
        arn_format=ArnFormat.COLON_RESOURCE_NAME,
    )

    logs_policy = cr.AwsCustomResourcePolicy.from_statements([
        # Creating/updating a delivery SOURCE is the credential-relevant action
        # (it is what carries `logType`), so it is pinned to the single frozen
        # TRACES source rather than "*". Asserted by
        # tests/test_synth_log_delivery.py::TestPutDeliverySourceIsPinned.
        iam.PolicyStatement(
            actions=["logs:PutDeliverySource"],
            resources=[traces_source_arn],
        ),
        iam.PolicyStatement(
            actions=[
                # Read/delete of the source stay account-scoped: the
                # service-authorization reference lists DeleteDeliverySource
                # against the `delivery-destination` resource type, so pinning it
                # to a delivery-source ARN risks an AccessDenied on stack
                # teardown. Neither action can introduce a log type.
                "logs:GetDeliverySource",
                "logs:DeleteDeliverySource",
                # Destination + delivery APIs are account-scoped by nature here:
                # the delivery's name/id is generated by the service, so there is
                # no ARN to pin at synth time. None of them carries a `logType`.
                "logs:PutDeliveryDestination",
                "logs:CreateDelivery",
                "logs:DeleteDeliveryDestination",
                "logs:DeleteDelivery",
                "logs:GetDeliveryDestination",
                "logs:GetDelivery",
                "logs:PutDeliveryDestinationPolicy",
                "logs:GetDeliveryDestinationPolicy",
                "logs:DeleteDeliveryDestinationPolicy",
                "logs:UpdateDeliveryConfiguration",
            ],
            resources=["*"],
        ),
        iam.PolicyStatement(
            actions=["bedrock-agentcore:AllowVendedLogDeliveryForResource"],
            resources=["*"],
        ),
        # X-Ray resource-policy perms so AWS can auto-create the policy that lets
        # delivery.logs.amazonaws.com write trace segments (required for CreateDelivery to an XRAY destination).
        # https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AWS-logs-infrastructure-V2-XRayTraces.html
        iam.PolicyStatement(
            actions=[
                "xray:PutResourcePolicy",
                "xray:ListResourcePolicies",
                "xray:GetTraceSegmentDestination",
            ],
            resources=["*"],
        ),
    ])

    # Step 1: delivery source (TRACES, gateway ARN)
    delivery_source = cr.AwsCustomResource(
        scope,
        "GatewayTracesSource",
        on_create=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="PutDeliverySource",
            parameters={
                "name": source_name,
                "logType": "TRACES",
                "resourceArn": gateway.gateway_arn,
            },
            physical_resource_id=cr.PhysicalResourceId.of(source_name),
        ),
        on_update=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="PutDeliverySource",
            parameters={
                "name": source_name,
                "logType": "TRACES",
                "resourceArn": gateway.gateway_arn,
            },
            physical_resource_id=cr.PhysicalResourceId.of(source_name),
        ),
        on_delete=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="DeleteDeliverySource",
            parameters={"name": source_name},
            ignore_error_codes_matching="ResourceNotFoundException",
        ),
        # install latest SDK: bedrock-agentcore-control / recent APIs may be absent from the Lambda built-in SDK
        install_latest_aws_sdk=True,
        policy=logs_policy,
    )

    # Step 2: delivery destination (XRAY)
    delivery_dest = cr.AwsCustomResource(
        scope,
        "GatewayTracesDestination",
        on_create=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="PutDeliveryDestination",
            parameters={
                "name": dest_name,
                "deliveryDestinationType": "XRAY",
            },
            physical_resource_id=cr.PhysicalResourceId.of(dest_name),
        ),
        on_update=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="PutDeliveryDestination",
            parameters={
                "name": dest_name,
                "deliveryDestinationType": "XRAY",
            },
            physical_resource_id=cr.PhysicalResourceId.of(dest_name),
        ),
        on_delete=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="DeleteDeliveryDestination",
            parameters={"name": dest_name},
            ignore_error_codes_matching="ResourceNotFoundException",
        ),
        install_latest_aws_sdk=True,
        policy=logs_policy,
    )
    delivery_dest.node.add_dependency(delivery_source)

    # Step 3: create delivery (link source -> destination)
    # The destination ARN comes from the PutDeliveryDestination response.
    dest_arn = delivery_dest.get_response_field("deliveryDestination.arn")

    delivery = cr.AwsCustomResource(
        scope,
        "GatewayTracesDelivery",
        on_create=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="CreateDelivery",
            parameters={
                "deliverySourceName": source_name,
                "deliveryDestinationArn": dest_arn,
            },
            physical_resource_id=cr.PhysicalResourceId.from_response(
                "delivery.id"
            ),
        ),
        # Delete the delivery (the source->destination link) on teardown.
        # Without this, CloudFormation cannot delete the delivery
        # destination because a delivery still references it, causing
        # DELETE_FAILED. The delivery id is the physical resource id
        # captured from the CreateDelivery response (delivery.id).
        # DeleteDelivery takes { "id": <delivery id> } — confirmed against
        # https://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_DeleteDelivery.html
        on_delete=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="DeleteDelivery",
            parameters={
                "id": cr.PhysicalResourceIdReference(),
            },
            ignore_error_codes_matching="ResourceNotFoundException",
        ),
        install_latest_aws_sdk=True,
        policy=logs_policy,
    )
    delivery.node.add_dependency(delivery_dest)


def attach_policy_engine(
    scope: Construct,
    gateway: agentcore.Gateway,
    policy_engine: agentcore_alpha.PolicyEngine,
    interceptor_fn: lambda_.Function,
    auth: AuthResources,
    response_interceptor_fn: lambda_.Function | None = None,
) -> None:
    """Attach the policy engine to the gateway via UpdateGateway API.

    Uses an AwsCustomResource to call the bedrock-agentcore-control
    UpdateGateway API after the gateway and its IAM role policy are both
    fully created. This avoids the race condition where CloudFormation
    tries to call GetPolicyEngine before the IAM permission exists.

    The UpdateGateway payload re-declares the authorizer, so it MUST mirror
    the CUSTOM_JWT authorizer set at creation time — otherwise this call would
    revert the gateway to AWS_IAM.

    UpdateGateway parity (critical): this call re-declares
    ``interceptorConfigurations`` in full, so it MUST re-declare EVERY
    interceptor to keep. When ``response_interceptor_fn`` is provided the
    RESPONSE config is appended alongside the REQUEST entry; otherwise
    UpdateGateway would rewrite the list WITHOUT it and silently drop it (a
    split-brain vs. the CloudFormation template). This is the SECOND of the two
    RESPONSE declaration points — the first is ``_build_interceptor_configs`` in
    gateway_resources.py; both are driven by the single ``response_interceptor_fn``
    threaded through ``create_gateway`` so they cannot diverge. When it is None
    only the REQUEST interceptor is declared.
    ``pass_request_headers=True`` on the REQUEST entry is unchanged; the RESPONSE
    entry needs no ``inputConfiguration``.

    AWS documentation references:
      - UpdateGateway with Policy Engine:
        https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/update-gateway-with-policy.html
      - CUSTOM_JWT customJWTAuthorizer shape (discoveryUrl, allowedClients):
        https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-create-api.html
      - interceptorConfigurations entry shape (lambda.arn + interceptionPoints):
        https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-configuration.html

    Args:
        scope: The CDK Stack or Construct to attach resources to.
        gateway: The Gateway to attach the policy engine to.
        policy_engine: The PolicyEngine to attach.
        interceptor_fn: The REQUEST interceptor Lambda.
        auth: The managed Cognito identity references, mirroring the values the
            creation-time authorizer used.
        response_interceptor_fn: The RESPONSE interceptor Lambda, or None to
            re-declare only the REQUEST interceptor.
    """
    interceptor_configs = [
        {
            "interceptor": {
                "lambda": {
                    "arn": interceptor_fn.function_arn,
                }
            },
            "interceptionPoints": ["REQUEST"],
            "inputConfiguration": {
                "passRequestHeaders": True,
            },
        }
    ]

    # UpdateGateway parity: re-declare the RESPONSE interceptor here too so this
    # call cannot silently drop it (conditional — it may not be wired).
    if response_interceptor_fn is not None:
        interceptor_configs.append(
            {
                "interceptor": {
                    "lambda": {
                        "arn": response_interceptor_fn.function_arn,
                    }
                },
                "interceptionPoints": ["RESPONSE"],
            }
        )

    update_params = {
        "gatewayIdentifier": gateway.gateway_id,
        "name": "support-tools-gateway",
        "roleArn": gateway.role.role_arn,
        # CUSTOM_JWT must match the creation-time authorizer so this
        # UpdateGateway call does not revert the gateway to AWS_IAM.
        # Validate via allowedClients only (Cognito access token aud=null).
        "authorizerType": "CUSTOM_JWT",
        "authorizerConfiguration": {
            "customJWTAuthorizer": {
                "discoveryUrl": auth.discovery_url,
                "allowedClients": [auth.app_client_id],
            }
        },
        "protocolConfiguration": {
            "mcp": {
                "instructions": "Customer support tools",
                "supportedVersions": ["2025-03-26", "2025-06-18", "2025-11-25"],
                "sessionConfiguration": {
                    "sessionTimeoutInSeconds": 3600,
                },
            }
        },
        "policyEngineConfiguration": {
            "arn": policy_engine.policy_engine_arn,
            "mode": "ENFORCE",
        },
        "interceptorConfigurations": interceptor_configs,
    }

    attach = cr.AwsCustomResource(
        scope,
        "AttachPolicyEngine",
        on_create=cr.AwsSdkCall(
            service="bedrock-agentcore-control",
            action="UpdateGateway",
            parameters=update_params,
            physical_resource_id=cr.PhysicalResourceId.of(
                "attach-policy-engine"
            ),
        ),
        on_update=cr.AwsSdkCall(
            service="bedrock-agentcore-control",
            action="UpdateGateway",
            parameters=update_params,
            physical_resource_id=cr.PhysicalResourceId.of(
                "attach-policy-engine"
            ),
        ),
        install_latest_aws_sdk=True,
        policy=cr.AwsCustomResourcePolicy.from_statements(
            [
                iam.PolicyStatement(
                    actions=[
                        "bedrock-agentcore:UpdateGateway",
                        "bedrock-agentcore:GetGateway",
                    ],
                    # Both actions operate on this ONE gateway, and its ARN is
                    # already in scope, so the deploy-time custom-resource role
                    # does not carry account-wide read/modify rights over every
                    # AgentCore gateway. Pinning bedrock-agentcore actions to
                    # `gateway.gateway_arn` is already proven in this stack:
                    # gateway_resources.py scopes AuthorizeAction /
                    # PartiallyAuthorizeActions to the same ARN.
                    #
                    # If a future deploy ever fails here with AccessDenied on
                    # GetGateway, the cause would be that action not accepting a
                    # resource ARN — revert THAT action to "*" with a reference
                    # to the AWS docs rather than re-widening both.
                    resources=[gateway.gateway_arn],
                ),
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[gateway.role.role_arn],
                ),
            ]
        ),
    )

    # Ensure the custom resource runs after the gateway AND its role
    # policy both exist. The gateway node includes the role as a child,
    # and CDK will ensure the DefaultPolicy is created as part of the
    # gateway's dependency tree.
    attach.node.add_dependency(gateway)

    # Also wait for the gateway role's inline DefaultPolicy.
    # The UpdateGateway call re-runs the gateway's GetPolicyEngine
    # check, so the role grants (GetPolicyEngine, AuthorizeAction) must
    # already be attached when the custom resource runs. Depending only on
    # the gateway does not guarantee the DefaultPolicy is in place.
    default_policy = gateway.role.node.try_find_child("DefaultPolicy")
    if default_policy is not None:
        attach.node.add_dependency(default_policy)
