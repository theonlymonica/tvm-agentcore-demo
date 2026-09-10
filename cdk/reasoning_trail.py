"""Destination for Bedrock model invocation logging — the reasoning trail.

WHAT THIS ANSWERS.

Work item 1 made the table access attributable and work item 2 recorded what the
interceptor was asked for and what it granted. One question is still unanswerable from
either: **why did the model ask for that tool?** The interceptor sees a finished
`tools/call` and has no idea what reasoning produced it. That lives only in the model
invocation — the prompt that went in and the completion that came out.

Bedrock records it only if model invocation logging is enabled, which is off by
default: "Model invocation logging is disabled by default"
(https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html).

WHY THE SWITCH ITSELF IS NOT IN THIS STACK, and this is the important part.

Model invocation logging is configured **per account and Region, for every Bedrock
invocation in it** — the documentation is explicit: "collect invocation logs, model
input data, and model output data for all invocations in your AWS account used in
Amazon Bedrock in a Region". It is NOT scoped to this agent, this stack, or this role.

So the toggle has a blast radius larger than the stack that would own it. A
`cdk deploy` of this project must not silently begin recording the full prompts and
completions of unrelated Bedrock workloads in the same account — and there is no
CloudFormation resource for it anyway: AWS's own prescriptive-guidance pattern
provisions "an AWS Lambda function that configures logging settings in Amazon Bedrock"
(https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/configure-bedrock-invocation-logging-cloudformation.html),
i.e. a custom resource, because the setting is only reachable through the API.

The split adopted here:

- **In the stack (this module):** the log group and the IAM role Bedrock assumes to
  write to it. Both are ordinary, scoped, reproducible resources that record nothing
  on their own — creating them changes no behaviour anywhere.
- **Out of the stack:** the account-wide switch, flipped deliberately with
  `aws bedrock put-model-invocation-logging-configuration` and recorded in
  `notes/`, with its previous value (no configuration = disabled) and the command to
  put it back.

That way the destination is version-controlled and the account-wide effect stays an
explicit human act rather than a side effect of a deploy.

COST. Logging writes the FULL request and response of every invocation, so the volume
is prompt + completion per call — far larger than anything else this project logs, and
the reason this is the most expensive item in the audit chain. Charged as CloudWatch
Logs ingestion and storage (https://aws.amazon.com/cloudwatch/pricing/).

PRIVACY, stated because it is the sharpest edge in the whole design. These logs contain
the model's prompts, which in this system include retrieved document content. The
project's standing rule is that no enriched body and no enriched tool arguments may be
written to any log — that rule is about OUR log lines, and it holds. This log group is
written by Bedrock, not by us, and it necessarily contains what was sent to the model.
Anyone enabling it is accepting that document content reaches CloudWatch Logs. That is
why the group carries audit retention and is created in this stack rather than being
left to Bedrock's default, and why the switch is a documented, reversible act.

Reference (AWS Documentation MCP server, per the ``aws-docs-lookup`` rule):
    - Destination setup, required trust policy and role policy, and the
      ``aws/bedrock/modelinvocations`` log-stream name:
      https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html

Functions:
    create_reasoning_trail_destination: Provision the log group and Bedrock's role.
"""

from __future__ import annotations

from typing import NamedTuple

import aws_cdk as cdk
import aws_cdk.aws_iam as iam
import aws_cdk.aws_logs as logs
from constructs import Construct

from observability import AUDIT_LOG_RETENTION

#: Log group the reasoning trail is delivered to. Named under /aws/bedrock/ so it is
#: recognisable next to the service's own conventions, and prefixed with the project
#: name so it is obviously this stack's.
REASONING_LOG_GROUP_NAME = "/aws/bedrock/scoped-credentials-modelinvocations"

#: The log stream Bedrock writes model invocations to. Fixed by the service and named
#: verbatim in the role policy below, so the grant cannot be wider than one stream.
_BEDROCK_LOG_STREAM = "aws/bedrock/modelinvocations"

#: The service principal that assumes the delivery role.
_BEDROCK_SERVICE_PRINCIPAL = "bedrock.amazonaws.com"


class ReasoningTrailDestination(NamedTuple):
    """The two resources the account-wide switch will be pointed at.

    Attributes:
        log_group: Log group receiving the model invocations.
        delivery_role: Role Bedrock assumes to write to it.
    """

    log_group: logs.LogGroup
    delivery_role: iam.Role


def create_reasoning_trail_destination(
    scope: Construct,
) -> ReasoningTrailDestination:
    """Provision the log group and the role Bedrock assumes to write to it.

    Creating these changes no behaviour: nothing is logged until the account-wide
    configuration is pointed at them (see the module docstring). They are in the stack
    so the destination is reproducible and its retention is deliberate.

    Args:
        scope: The CDK Stack or Construct to attach the resources to.

    Returns:
        The log group and delivery role, for the stack to output.
    """
    stack = cdk.Stack.of(scope)

    log_group = logs.LogGroup(
        scope,
        "ReasoningTrailLogGroup",
        log_group_name=REASONING_LOG_GROUP_NAME,
        # Audit retention: this group answers "why did the model ask for that tool",
        # which is one of the auditor's six questions, so it must not expire before
        # the records it is joined to.
        retention=AUDIT_LOG_RETENTION,
        removal_policy=cdk.RemovalPolicy.DESTROY,
    )

    # The trust policy is confused-deputy hardened exactly as the documentation
    # prescribes: `bedrock.amazonaws.com` may assume this role ONLY when acting for
    # this account and for a Bedrock ARN in this account. Without the two conditions
    # any account's Bedrock could name this role.
    delivery_role = iam.Role(
        scope,
        "ReasoningTrailDeliveryRole",
        assumed_by=iam.ServicePrincipal(
            _BEDROCK_SERVICE_PRINCIPAL,
            conditions={
                "StringEquals": {"aws:SourceAccount": stack.account},
                "ArnLike": {
                    "aws:SourceArn": (
                        f"arn:aws:bedrock:{stack.region}:{stack.account}:*"
                    )
                },
            },
        ),
        description=(
            "Role Bedrock assumes to deliver model invocation logs (the reasoning "
            "trail) to this project's log group"
        ),
    )

    # Narrowest grant the service will work with: two actions, on ONE log stream in
    # ONE group. Not the group's wildcard, and not logs:* — the documented stream
    # name is fixed, so there is no reason to allow more.
    #
    # The ARN is assembled here rather than from `log_group.log_group_arn`, and that
    # is not stylistic: CDK's `log_group_arn` returns the group ARN WITH ":*"
    # appended. Interpolating it before ":log-stream:..." yields
    # "...:log-group:/name:*:log-stream:aws/bedrock/modelinvocations", which is not a
    # valid ARN. Bedrock rejects the whole configuration with a ValidationException
    # ("Failed to validate permissions for log group ... Verify the IAM role
    # permissions are correct") — a message that points at the permissions rather
    # than at the malformed resource, so this is worth not rediscovering.
    log_stream_arn = (
        f"arn:aws:logs:{stack.region}:{stack.account}:log-group:"
        f"{REASONING_LOG_GROUP_NAME}:log-stream:{_BEDROCK_LOG_STREAM}"
    )
    delivery_role.add_to_policy(
        iam.PolicyStatement(
            sid="WriteModelInvocationsToOneStream",
            effect=iam.Effect.ALLOW,
            actions=["logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[log_stream_arn],
        )
    )

    cdk.CfnOutput(
        scope,
        "ReasoningTrailLogGroupName",
        value=REASONING_LOG_GROUP_NAME,
        description=(
            "Destination for Bedrock model invocation logging. Creating it logs "
            "nothing; the account-wide switch is a documented manual act"
        ),
    )
    cdk.CfnOutput(
        scope,
        "ReasoningTrailDeliveryRoleArn",
        value=delivery_role.role_arn,
        description="Role ARN to pass as loggingConfig.roleArn when enabling logging",
    )

    return ReasoningTrailDestination(
        log_group=log_group,
        delivery_role=delivery_role,
    )
