"""Shared Lambda log-group construction.

Every first-party Lambda in this stack gets an EXPLICIT ``logs.LogGroup`` with a
finite retention, instead of relying on the log group Lambda creates implicitly
on first invocation (which never expires and is not CloudFormation-managed).

Why an explicit LogGroup rather than ``Function(log_retention=...)``
-------------------------------------------------------------------
``log_retention`` is deprecated in aws-cdk-lib and synthesizes a
``Custom::LogRetention`` resource backed by a CDK-generated Lambda whose role
holds ``logs:PutRetentionPolicy`` / ``logs:DeleteRetentionPolicy`` on ``"*"``.
Removing over-wide grants is precisely the point of this stack, so paying for
retention with a new account-wide wildcard grant would be self-defeating. An
explicit ``logs.LogGroup`` passed as ``log_group=`` sets the function's
``LoggingConfig.LogGroup`` directly: no custom resource, no extra Lambda, no
wildcard, and ``RetentionInDays`` is visible in the synthesized template (so
``tests/test_synth_operational_posture.py`` can assert it).

The log group is created with ``RemovalPolicy.DESTROY`` so the groups go away on
teardown; RETAIN would leave orphaned groups behind on every ``cdk destroy`` /
redeploy cycle.

Retention choice: ONE_MONTH. These are demo/experiment logs whose value is
short-lived — a run is analysed the same day. The purpose is cost containment,
not forensics: nothing here is an audit trail, so a longer window would buy
nothing.

Documentation references:
  - aws_cdk.aws_lambda.FunctionOptions.log_group (sets LoggingConfig.LogGroup;
    log_retention is documented as deprecated in favour of it):
    https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_lambda/FunctionOptions.html
  - aws_cdk.aws_logs.LogGroup / RetentionDays:
    https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_logs/LogGroup.html
  - Lambda advanced logging controls (a function may target a named log group):
    https://docs.aws.amazon.com/lambda/latest/dg/monitoring-cloudwatchlogs.html
"""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_logs as logs
from constructs import Construct

# Retention applied to every first-party Lambda log group (see module docstring).
LAMBDA_LOG_RETENTION = logs.RetentionDays.ONE_MONTH

# Reserved concurrency for the three tool Lambdas.
#
# Reserved concurrency is both a CAP and a carve-out from the account pool: a
# runaway or abusive caller can consume at most this many concurrent executions
# per tool, so it cannot exhaust account concurrency and starve other workloads.
# 10 is ample for the single-user demo, which drives tens of sequential requests
# rather than hundreds of parallel ones; 3 x 10 = 30 leaves the account's
# 100-execution unreserved floor untouched.
#
# Deliberately NOT applied to the two interceptors or the seed function:
#   - the interceptors sit on EVERY gateway request, so a reservation there
#     converts a burst into gateway-visible throttling for all tenants at once
#     — the opposite of the isolation this cap is meant to provide;
#   - the seed function runs once, invoked by the CloudFormation provider.
TOOL_RESERVED_CONCURRENCY = 10


def lambda_log_group(
    scope: Construct,
    construct_id: str,
    *,
    function_name: str,
) -> logs.LogGroup:
    """Create a retention-bounded, teardown-friendly Lambda log group.

    Args:
        scope: The CDK Stack or Construct to attach the log group to.
        construct_id: The construct id for the log group (conventionally the
            owning function's construct id plus ``LogGroup``).
        function_name: The Lambda function name whose conventional log-group
            path (``/aws/lambda/<function_name>``) this group takes. For a
            function with an auto-generated physical name, pass the stable
            ``scoped-credentials-*`` label the group should carry instead.

    Returns:
        The ``logs.LogGroup`` to pass as the function's ``log_group=``.
    """
    return logs.LogGroup(
        scope,
        construct_id,
        log_group_name=f"/aws/lambda/{function_name}",
        retention=LAMBDA_LOG_RETENTION,
        removal_policy=cdk.RemovalPolicy.DESTROY,
    )
