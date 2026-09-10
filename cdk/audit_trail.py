"""CloudTrail data-plane audit trail for the Documents table.

WHY THIS EXISTS, and why it is a separate module.

Work item 1 makes every vended session carry two identity parameters: a
``SourceIdentity`` (the Cognito ``sub`` of the human) and a ``RoleSessionName``
derived from the Gateway request. Those land in the ``AssumeRole`` event, which is a
MANAGEMENT event and therefore visible in CloudTrail Event history for 90 days at no
cost, with no trail required.

The DynamoDB read or write performed WITH those credentials is a DATA-plane event.
CloudTrail does not record data events unless a trail explicitly selects them:
"To enable logging of the following API actions in CloudTrail files, you must enable
logging of data plane API activity in CloudTrail"
(https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/logging-using-cloudtrail.html).
Without this module the audit chain stops at the moment the credential is minted,
and the actual table access is recorded nowhere.

WHAT A DATA EVENT ADDS, verified against the documentation rather than assumed:

- The item-level operations this system performs are all covered: ``GetItem``,
  ``Query`` and ``UpdateItem`` are on the documented list of DynamoDB data-plane
  actions CloudTrail records.
- ``requestParameters`` carries ``tableName`` AND ``key`` — so the ITEM KEYS appear
  in the event. AWS's own worked example shows ``"key": {"key": "<partitionKey>"}``
  (https://aws.amazon.com/blogs/database/amazon-dynamodb-now-supports-audit-logging-and-monitoring-using-aws-cloudtrail/).
  That is worth stating plainly: turning this on means the partition key — here the
  TEAM NAME — and the document id are written into the trail. The document BODY is
  not; only the keys and the request metadata.
- ``userIdentity.sessionContext`` carries the assumed-role session, which is where
  the work-item-1 ``sourceIdentity`` and the derived role session name become
  readable on the data access itself.
- ``readOnly`` distinguishes a read from a write, and ``resources`` names the table
  ARN.

COST, stated because switching this on is the one change here that bills per use:
data events are charged at $0.10 per 100,000 events delivered to S3, plus S3 storage
for the trail's own objects (https://aws.amazon.com/cloudtrail/pricing/,
https://aws.amazon.com/s3/pricing/). Charging is PER EVENT DELIVERED, so a trail with
no matching activity costs nothing beyond the storage already written. Two deliberate
narrowings keep that true:

1. Data events only, and only two selectors: the Documents table by ARN, and
   AgentCore Runtime invocations. Not "all DynamoDB tables in the account".
2. Management events are NOT selected. Note the reason, because the obvious one is
   wrong: the first copy of management events delivered to S3 is FREE
   ("You can deliver one copy of your ongoing management events to your Amazon S3
   bucket for free by creating trails" — the pricing page above), so excluding them
   saves no per-event charge. They are excluded because they are account-wide and
   constant — every deploy, every console action, every service-linked role — which
   would bury the handful of events this trail exists to show under unrelated noise
   and grow the bucket continuously whether or not anyone is testing. The AssumeRole
   half of the chain is separately readable in Event history for 90 days at no cost.

The runtime selector adds one event per user invocation, which is negligible next to
the three table events the same request already produces.

A note on a documented trap that does not apply here but would if the table changed:
specifying ``AWS::DynamoDB::Table`` logs table AND DynamoDB Streams events by
default, and internal ``GetRecords`` calls from replication are billable as data
events even though DynamoDB does not charge for them. The Documents table has no
stream, so there is nothing to exclude today — but adding a stream later would start
billing stream events through this selector.

Teardown: the trail and its bucket are ``RemovalPolicy.DESTROY`` with
``auto_delete_objects``, so a ``cdk destroy`` leaves nothing behind and does not fail
on a non-empty bucket.

Functions:
    create_audit_trail: Provision the trail and select the Documents table's data
        events.
"""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_cloudtrail as cloudtrail
import aws_cdk.aws_s3 as s3
from constructs import Construct

#: Physical trail name. Deliberately NOT set as a fixed physical name on the bucket
#: (see the redeployability note in cdk/documents_roles.py): a CloudTrail trail name
#: is unique per account, and a fixed bucket name is unique GLOBALLY, which is the
#: worse of the two constraints. The bucket therefore gets a generated name.
TRAIL_NAME = "scoped-credentials-documents-audit"


def create_audit_trail(
    scope: Construct,
    documents_table_arn: str,
) -> cloudtrail.Trail:
    """Create the trail that records data events on the Documents table.

    Args:
        scope: The CDK Stack or Construct to attach the trail to.
        documents_table_arn: ARN of the Documents table — the ONLY data resource
            selected, so the trail cannot silently start billing for other tables.

    Returns:
        The provisioned ``Trail`` construct.
    """
    bucket = s3.Bucket(
        scope,
        "AuditTrailBucket",
        block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        encryption=s3.BucketEncryption.S3_MANAGED,
        enforce_ssl=True,
        # The trail is a temporary verification aid, so teardown must be clean:
        # without auto_delete_objects a `cdk destroy` fails on a non-empty bucket
        # and leaves both the bucket and its log objects behind.
        removal_policy=cdk.RemovalPolicy.DESTROY,
        auto_delete_objects=True,
    )

    trail = cloudtrail.Trail(
        scope,
        "DocumentsAuditTrail",
        trail_name=TRAIL_NAME,
        bucket=bucket,
        # `management_events` is deliberately LEFT AT ITS DEFAULT here, and that is
        # not the same as management events being logged — read on, because the two
        # statements only look contradictory.
        #
        # Passing `ReadWriteType.NONE` is what this wants to say, but the L2 refuses
        # it: "At least one event selector must be added when management event
        # recording is set to None", and it decides that at synth-time validation,
        # before the L1 override below is applied — so the override cannot satisfy it.
        #
        # What actually ships is the override: it REPLACES the EventSelectors array
        # with a single selector carrying `IncludeManagementEvents: False`. A trail
        # logs what its selectors say, so no management event is delivered by this
        # trail regardless of this prop. tests/test_audit_trail_scope.py asserts that
        # on the SYNTHESIZED template rather than trusting this comment.
        management_events=cloudtrail.ReadWriteType.ALL,
    )

    # Two data-event selectors, declared as ADVANCED selectors.
    #
    # Advanced rather than basic is forced by the second resource type:
    # `AWS::DynamoDB::Table` is one of the three BASIC types, but
    # `AWS::BedrockAgentCore::Runtime` is not, and a trail cannot carry both forms —
    # "You cannot apply both event selectors and advanced event selectors to a trail"
    # (https://docs.aws.amazon.com/awscloudtrail/latest/APIReference/API_AdvancedEventSelector.html).
    # So the table selector is re-expressed in advanced form alongside the new one.
    #
    # WHY THE RUNTIME SELECTOR EXISTS. `InvokeAgentRuntime` is a DATA event under
    # `resources.type = AWS::BedrockAgentCore::Runtime`
    # (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-operations.html),
    # which is why it appears nowhere in Event history: `lookup-events` returns
    # management events only, and a trail that selects just the table discards
    # everything else. AgentCore's own security guidance asks for exactly this —
    # "Enable CloudTrail logging — AWS CloudTrail records API calls including
    # InvokeAgentRuntime ... Each record includes caller identity" and "Correlate
    # logs using request IDs"
    # (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-security-best-practices.html).
    # That gives an AWS-WRITTEN record of who invoked the agent, which is the one
    # class of evidence the agent itself cannot forge.
    #
    # Management events are simply not selected: with advanced selectors there is no
    # IncludeManagementEvents flag to set false, so their absence is structural
    # rather than a suppressed default.
    #
    # NOT INCLUDED, deliberately: a Gateway data-event resource type. No
    # `AWS::BedrockAgentCore::Gateway` selector is documented in the sources above,
    # and naming an unsupported resources.type would be a guess — the trail would be
    # rejected or, worse, silently select nothing. Flagged rather than invented.
    cfn_trail = trail.node.default_child
    # The L2 always emits EventSelectors; the two forms are mutually exclusive, so it
    # has to be removed rather than merely left unused.
    cfn_trail.add_deletion_override("Properties.EventSelectors")
    cfn_trail.add_property_override(
        "AdvancedEventSelectors",
        [
            {
                "Name": "DocumentsTableDataEvents",
                "FieldSelectors": [
                    {"Field": "eventCategory", "Equals": ["Data"]},
                    {"Field": "resources.type", "Equals": ["AWS::DynamoDB::Table"]},
                    {"Field": "resources.ARN", "Equals": [documents_table_arn]},
                ],
            },
            {
                "Name": "AgentRuntimeInvocationDataEvents",
                "FieldSelectors": [
                    {"Field": "eventCategory", "Equals": ["Data"]},
                    {
                        "Field": "resources.type",
                        "Equals": ["AWS::BedrockAgentCore::Runtime"],
                    },
                ],
            },
        ],
    )

    cdk.CfnOutput(
        scope,
        "DocumentsAuditTrailName",
        value=TRAIL_NAME,
        description=(
            "CloudTrail trail recording DynamoDB data events for the Documents "
            "table (work item 1 verification; billed per data event)"
        ),
    )

    return trail
