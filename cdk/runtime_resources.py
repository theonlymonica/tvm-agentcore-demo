"""
AgentCore Runtime resources.

This module provisions:
- AgentCore Runtime with ARM64 container built from the agent/ directory
- Environment variables (BEDROCK_MODEL_ID, AWS_REGION, GATEWAY_URL)
- Tracing enabled for observability (spans → CloudWatch aws/spans)
- CloudWatch Transaction Search enablement (two custom resources)
- IAM grants: runtime can invoke the ONE configured Bedrock model
  (bedrock:InvokeModel + bedrock:InvokeModelWithResponseStream, pinned by
  cdk/bedrock_model_access.py); console principal can invoke
  the runtime (no gateway IAM invoke grant — CUSTOM_JWT gateway uses a Bearer
  JWT, not IAM; see the note near the end of create_runtime)

Transaction Search enablement (two-step):
  Step 1: Create a CloudWatch Logs resource policy allowing X-Ray to
          write to the aws/spans log group (via logs:PutResourcePolicy).
  Step 2: Call UpdateTraceSegmentDestination with Destination=CloudWatchLogs
          so that X-Ray trace segments route to CloudWatch Logs.

  These are modelled as AwsCustomResource constructs so they execute at
  deploy time and are cleaned up on stack deletion.

  AWS documentation references:
    - Transaction Search IAM requirements:
      https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Enable-TransactionSearch.html
    - CDK AwsCustomResource:
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.custom_resources/AwsCustomResource.html

Gateway observability (tracing) for policy decision spans:
  Cedar authorization decision spans only reach the CloudWatch aws/spans log
  group once a TRACES delivery pipeline exists for the gateway. That pipeline
  is not configurable via a CDK prop on the Gateway L2, so it is built at
  deploy time by gateway_wiring.enable_gateway_tracing — three chained
  AwsCustomResource constructs (delivery source, delivery destination, and the
  delivery that joins them) over the CloudWatch Logs
  put_delivery_source / put_delivery_destination / create_delivery APIs. No
  manual console or CLI step is required, and Transaction Search itself is
  enabled by the two custom resources in this module.

  What to inspect after a deploy: invoke a tool through the gateway, then look
  at
    1. the CloudWatch aws/spans log group — span attribute
       aws.agentcore.policy.authorization_decision (values: ALLOW / DENY)
    2. the AWS/Bedrock-AgentCore CloudWatch metrics
       AllowDecisions / DenyDecisions (dimensions: ToolName, Mode)

  AWS documentation references:
    - Policy observability data:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-policy-metrics.html
    - Enabling gateway tracing:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html
    - CDK Runtime construct (tracing_enabled, environment_variables):
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_bedrockagentcore/README.html
"""

from __future__ import annotations

import json
import os

import aws_cdk as cdk
import aws_cdk.aws_bedrockagentcore as agentcore
import aws_cdk.aws_iam as iam
import aws_cdk.custom_resources as cr
from constructs import Construct

from asset_packaging import ASSET_EXCLUDE
from bedrock_model_access import invoke_statements


def create_runtime(
    scope: Construct,
    gateway: agentcore.Gateway,
    config,
) -> agentcore.Runtime:
    """Create the AgentCore Runtime for the support agent.

    Provisions the Runtime with:
    - ARM64 container built from agent/ (Dockerfile targets linux/arm64)
    - Environment variables: BEDROCK_MODEL_ID, AWS_REGION, GATEWAY_URL
    - Tracing enabled for observability
    - Grant: runtime execution role → bedrock:InvokeModel and
      bedrock:InvokeModelWithResponseStream on the single configured model
      (one statement for a bare foundation model, two for a cross-Region
      inference profile — see cdk/bedrock_model_access.py).

    Also creates two custom resources to enable CloudWatch Transaction
    Search so that X-Ray trace segments are routed to CloudWatch Logs:
      1. CreateXRayLogsResourcePolicy — logs:PutResourcePolicy
      2. EnableTransactionSearch — xray:UpdateTraceSegmentDestination

    Note on protocol_configuration:
      Defaults to ProtocolType.HTTP which is correct for the Strands agent
      HTTP contract (POST /invocations, GET /ping). Confirmed in CDK docs:
      "Protocol configuration for the agent runtime. Defaults to
      ProtocolType.HTTP"
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_bedrockagentcore/README.html

    Note on RuntimeEndpoint:
      A DEFAULT endpoint is created automatically when the Runtime is
      provisioned. No named RuntimeEndpoint is required for the initial
      deploy.

    Args:
        scope: The CDK Stack or Construct to attach resources to.
        gateway: The AgentCore Gateway (for constructing the GATEWAY_URL env).
        config: The AppConfig instance with bedrock_model_id, aws_region.

    Returns:
        The provisioned Runtime construct.
    """
    # ------------------------------------------------------------------
    # Step 1: Create resource policy allowing X-Ray to write to aws/spans.
    # This must exist before UpdateTraceSegmentDestination can succeed.
    # Uses cdk.Fn.sub so ${AWS::AccountId} and ${AWS::Region} resolve at
    # deploy time (not synth time).
    # ------------------------------------------------------------------
    logs_resource_policy_doc = cdk.Fn.sub(
        json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Sid": "TransactionSearchXRayAccess",
                "Effect": "Allow",
                "Principal": {"Service": "xray.amazonaws.com"},
                "Action": "logs:PutLogEvents",
                "Resource": [
                    "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:aws/spans:*",
                    "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/application-signals/data:*",
                ],
                "Condition": {
                    "ArnLike": {
                        "aws:SourceArn": "arn:aws:xray:${AWS::Region}:${AWS::AccountId}:*"
                    },
                    "StringEquals": {
                        "aws:SourceAccount": "${AWS::AccountId}"
                    },
                },
            }],
        })
    )

    create_logs_policy = cr.AwsCustomResource(
        scope,
        "CreateXRayLogsResourcePolicy",
        on_create=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="PutResourcePolicy",
            parameters={
                "policyName": "ScopedCredentialsXRayTransactionSearch",
                "policyDocument": logs_resource_policy_doc,
            },
            physical_resource_id=cr.PhysicalResourceId.of(
                "xray-logs-resource-policy"
            ),
        ),
        on_delete=cr.AwsSdkCall(
            service="CloudWatchLogs",
            action="DeleteResourcePolicy",
            parameters={
                "policyName": "ScopedCredentialsXRayTransactionSearch",
            },
        ),
        install_latest_aws_sdk=True,
        # WILDCARD ON PURPOSE. The CloudWatch Logs
        # resource-policy APIs are ACCOUNT-level: a log-group resource policy is
        # a single per-account object, not a per-ARN resource, so
        # logs:PutResourcePolicy / DeleteResourcePolicy /
        # DescribeResourcePolicies accept no resource ARN and "*" is the only
        # policy that works. Narrowing was attempted and rejected — recording it
        # here is more useful than a future reviewer re-deriving it. Access is
        # further bounded by this being a DEPLOY-TIME custom-resource role, and
        # by the policy document above, whose aws:SourceAccount / aws:SourceArn
        # conditions confine the trust it creates to this account.
        policy=cr.AwsCustomResourcePolicy.from_statements([
            iam.PolicyStatement(
                actions=[
                    "logs:PutResourcePolicy",
                    "logs:DeleteResourcePolicy",
                    "logs:DescribeResourcePolicies",
                ],
                resources=["*"],
            ),
        ]),
    )

    # ------------------------------------------------------------------
    # Step 2: Enable CloudWatch Logs as X-Ray trace segment destination.
    # Per docs, this requires ALL of the following permissions:
    # https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Enable-TransactionSearch.html
    # ------------------------------------------------------------------
    enable_transaction_search = cr.AwsCustomResource(
        scope,
        "EnableTransactionSearch",
        on_create=cr.AwsSdkCall(
            service="XRay",
            action="UpdateTraceSegmentDestination",
            parameters={"Destination": "CloudWatchLogs"},
            physical_resource_id=cr.PhysicalResourceId.of(
                "xray-transaction-search"
            ),
            ignore_error_codes_matching=".*PENDING.*|.*InvalidRequest.*|.*already.*",
        ),
        on_update=cr.AwsSdkCall(
            service="XRay",
            action="UpdateTraceSegmentDestination",
            parameters={"Destination": "CloudWatchLogs"},
            physical_resource_id=cr.PhysicalResourceId.of(
                "xray-transaction-search"
            ),
            ignore_error_codes_matching=".*PENDING.*|.*InvalidRequest.*|.*already.*",
        ),
        install_latest_aws_sdk=True,
        # WILDCARDS ON PURPOSE. Transaction Search is an
        # ACCOUNT-level setting, and the APIs that configure it take no resource
        # ARN: xray:{Get,Update}TraceSegmentDestination and
        # xray:{GetIndexingRules,UpdateIndexingRule} act on the account's single
        # trace configuration, application-signals:StartDiscovery creates the
        # account's service-linked discovery, and the logs resource-policy calls
        # are account-level as above. The two statements that CAN be pinned
        # already are: the log-group grant is scoped to the two
        # application-signals / spans group ARNs, and iam:CreateServiceLinkedRole
        # / iam:GetRole are scoped to the service-linked role ARN. This is a
        # deploy-time custom-resource role, not a runtime one.
        policy=cr.AwsCustomResourcePolicy.from_statements([
            iam.PolicyStatement(
                sid="TransactionSearchXRayPermissions",
                actions=[
                    "xray:GetTraceSegmentDestination",
                    "xray:UpdateTraceSegmentDestination",
                    "xray:GetIndexingRules",
                    "xray:UpdateIndexingRule",
                ],
                resources=["*"],
            ),
            iam.PolicyStatement(
                sid="TransactionSearchLogGroupPermissions",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutRetentionPolicy",
                ],
                resources=[
                    "arn:aws:logs:*:*:log-group:/aws/application-signals/data:*",
                    "arn:aws:logs:*:*:log-group:aws/spans:*",
                ],
            ),
            iam.PolicyStatement(
                sid="TransactionSearchLogsPermissions",
                actions=[
                    "logs:PutResourcePolicy",
                    "logs:DescribeResourcePolicies",
                ],
                resources=["*"],
            ),
            iam.PolicyStatement(
                sid="TransactionSearchApplicationSignalsPermissions",
                actions=["application-signals:StartDiscovery"],
                resources=["*"],
            ),
            iam.PolicyStatement(
                sid="CloudWatchApplicationSignalsCreateServiceLinkedRolePermissions",
                actions=["iam:CreateServiceLinkedRole"],
                resources=[
                    "arn:aws:iam::*:role/aws-service-role/"
                    "application-signals.cloudwatch.amazonaws.com/"
                    "AWSServiceRoleForCloudWatchApplicationSignals"
                ],
                conditions={
                    "StringLike": {
                        "iam:AWSServiceName": "application-signals.cloudwatch.amazonaws.com"
                    }
                },
            ),
            iam.PolicyStatement(
                sid="CloudWatchApplicationSignalsGetRolePermissions",
                actions=["iam:GetRole"],
                resources=[
                    "arn:aws:iam::*:role/aws-service-role/"
                    "application-signals.cloudwatch.amazonaws.com/"
                    "AWSServiceRoleForCloudWatchApplicationSignals"
                ],
            ),
            iam.PolicyStatement(
                sid="CloudWatchApplicationSignalsCloudTrailPermissions",
                actions=["cloudtrail:CreateServiceLinkedChannel"],
                resources=[
                    "arn:aws:cloudtrail:*:*:channel/"
                    "aws-service-channel/application-signals/*"
                ],
            ),
        ]),
    )

    # Ordering: logs policy must exist before transaction search is enabled
    enable_transaction_search.node.add_dependency(create_logs_policy)

    # ------------------------------------------------------------------
    # Runtime provisioning
    # ------------------------------------------------------------------

    # Resolve the agent/ directory relative to cdk/
    agent_dir = os.path.join(os.path.dirname(__file__), "..", "agent")

    # Build the ARM64 container from the local Dockerfile.
    # AgentRuntimeArtifact.from_asset builds from a local Docker context.
    # The Dockerfile already targets linux/arm64.
    #
    # exclude=: the asset hash fingerprints the whole staged context, so a
    # local pytest run writing agent/__pycache__ moved the ECR tag. See
    # cdk/asset_packaging.py for why not an agent/.dockerignore.
    agent_runtime_artifact = agentcore.AgentRuntimeArtifact.from_asset(
        agent_dir,
        exclude=ASSET_EXCLUDE,
    )

    # Construct the gateway URL from the gateway ID.
    # Format: https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
    # The gateway ID is accessible from the underlying CfnGateway Ref.
    cfn_gateway = gateway.node.default_child
    gateway_url = cdk.Fn.join("", [
        "https://",
        cfn_gateway.ref,
        ".gateway.bedrock-agentcore.",
        config.aws_region,
        ".amazonaws.com/mcp",
    ])

    runtime_env = {
        "BEDROCK_MODEL_ID": config.bedrock_model_id,
        "AWS_REGION": config.aws_region,
        "GATEWAY_URL": gateway_url,
    }

    runtime = agentcore.Runtime(
        scope,
        "SupportAgentRuntime",
        runtime_name="support_agent",
        agent_runtime_artifact=agent_runtime_artifact,
        description=(
            "Strands support agent runtime — ARM64 container, HTTP protocol"
        ),
        environment_variables=runtime_env,
        # Tracing enabled so Runtime spans flow to CloudWatch aws/spans.
        # Transaction Search custom resources above ensure the destination
        # and resource policy are in place before the Runtime deploys.
        tracing_enabled=True,
    )

    # Ensure transaction search is fully enabled before the Runtime creates
    # its AWS::Logs::Delivery with X-Ray destination.
    runtime.node.add_dependency(enable_transaction_search)

    # Grant the runtime's execution role permission to invoke EXACTLY the one
    # configured Bedrock model — nothing else. The agent has no per-invocation
    # model override (the model comes solely from bedrock_model_id / the
    # BEDROCK_MODEL_ID env var), so the grant no longer needs to cover every
    # foundation model and every inference profile in the account.
    #
    # For a cross-Region inference profile id this yields two statements: the
    # profile ARN in the source Region, and the underlying foundation model
    # constrained by bedrock:InferenceProfileArn to requests routed through that
    # profile. See cdk/bedrock_model_access.py for the ARN rules and citations.
    for statement in invoke_statements(
        config.bedrock_model_id,
        config.aws_region,
        cdk.Stack.of(scope).account,
    ):
        runtime.role.add_to_principal_policy(statement)

    # NOTE: No gateway IAM invoke grant under CUSTOM_JWT.
    # The gateway is CUSTOM_JWT, so the agent's MCP call is authenticated by a
    # forwarded user JWT (Authorization: Bearer <token>), NOT SigV4/IAM. The
    # former ``gateway.grant_invoke(runtime.role)`` granted
    # ``bedrock-agentcore:InvokeGateway`` (the AWS_IAM/SigV4 inbound action),
    # which does nothing for a CUSTOM_JWT gateway — removed as dead IAM. The
    # Bearer token is acquired and forwarded by the console and attached to
    # outbound MCP calls by the agent code; GATEWAY_URL is kept.
    # See the AgentCore gateway inbound-auth documentation
    # (gateway-inbound-auth.html): grant_invoke grants InvokeGateway, the
    # AWS_IAM/SigV4 inbound action, while a CUSTOM_JWT gateway uses a Bearer
    # JWT instead.

    cfn_runtime = runtime.node.default_child
    cdk.CfnOutput(
        scope,
        "SupportAgentRuntimeArn",
        value=cfn_runtime.attr_agent_runtime_arn,
        description="AgentCore Runtime ARN for console invocation",
    )

    return runtime
