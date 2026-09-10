"""The reasoning trail's destination must be narrow, and must log nothing by itself.

``cdk/reasoning_trail.py`` creates a log group and a role Bedrock assumes to write to
it. Two risks worth pinning, and neither is about correctness of the happy path:

1. **The role is assumed by an AWS service**, so without the documented
   ``aws:SourceAccount`` / ``aws:SourceArn`` conditions it is a confused-deputy hole:
   any account's Bedrock could name this role and write into this account's logs.
2. **Creating the destination must change no behaviour.** Model invocation logging is
   an account-and-Region-wide switch covering every Bedrock invocation in the account.
   If a deploy could flip it, deploying this demo would start recording the full
   prompts and completions of unrelated workloads. There is no CloudFormation resource
   for the switch, so the test that matters is the ABSENCE of any custom resource
   sneaking one in.

AWS documentation reference:
    https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from aws_cdk.assertions import Template

import synth_helpers as sh
from reasoning_trail import REASONING_LOG_GROUP_NAME

BEDROCK_PRINCIPAL = "bedrock.amazonaws.com"
EXPECTED_STREAM = "aws/bedrock/modelinvocations"


@pytest.fixture(scope="module")
def template() -> Template:
    """Synthesize the real stack."""
    _stack, tmpl = sh.build_full_stack()
    return tmpl


def _reasoning_role(template: Template) -> dict[str, Any]:
    """Return the delivery role's properties."""
    roles = {
        lid: resource["Properties"]
        for lid, resource in template.find_resources("AWS::IAM::Role").items()
        if lid.startswith("ReasoningTrailDeliveryRole")
    }
    assert len(roles) == 1, f"expected one delivery role, found {list(roles)}"
    return next(iter(roles.values()))


class TestTheDestinationExists:
    def test_log_group_is_created_with_the_expected_name(
        self, template: Template
    ) -> None:
        """A stack-owned group, so its retention is deliberate rather than default."""
        groups = [
            resource["Properties"]
            for resource in template.find_resources("AWS::Logs::LogGroup").values()
            if resource["Properties"].get("LogGroupName")
            == REASONING_LOG_GROUP_NAME
        ]
        assert len(groups) == 1, (
            f"exactly one log group named {REASONING_LOG_GROUP_NAME}"
        )

    def test_log_group_is_removed_on_teardown(self, template: Template) -> None:
        """A destroy must not orphan a group holding prompts and completions."""
        matching = [
            resource
            for resource in template.find_resources("AWS::Logs::LogGroup").values()
            if resource["Properties"].get("LogGroupName")
            == REASONING_LOG_GROUP_NAME
        ]
        assert matching[0].get("DeletionPolicy") == "Delete"


class TestTheRoleCannotBeAConfusedDeputy:
    def test_only_bedrock_can_assume_it(self, template: Template) -> None:
        """One service principal, no accounts, no wildcards."""
        statements = _reasoning_role(template)["AssumeRolePolicyDocument"]["Statement"]
        assert len(statements) == 1, "one trust statement"

        principal = statements[0]["Principal"]
        assert principal == {"Service": BEDROCK_PRINCIPAL}, (
            f"only {BEDROCK_PRINCIPAL} may assume this role; got {principal!r}"
        )

    def test_the_trust_is_fenced_to_this_account(self, template: Template) -> None:
        """Without aws:SourceAccount, another account's Bedrock could use this role."""
        statement = _reasoning_role(template)["AssumeRolePolicyDocument"][
            "Statement"
        ][0]
        condition = statement.get("Condition", {})

        assert "aws:SourceAccount" in condition.get("StringEquals", {}), (
            "the trust policy must pin aws:SourceAccount — a service principal "
            "without it is assumable on behalf of any account (confused deputy); "
            f"got {condition!r}"
        )

    def test_the_trust_is_fenced_to_a_bedrock_arn(self, template: Template) -> None:
        """The second half of the documented confused-deputy guard."""
        statement = _reasoning_role(template)["AssumeRolePolicyDocument"][
            "Statement"
        ][0]
        arn_like = statement.get("Condition", {}).get("ArnLike", {})

        assert "aws:SourceArn" in arn_like, (
            f"the trust policy must pin aws:SourceArn; got {arn_like!r}"
        )
        assert "bedrock" in json.dumps(arn_like["aws:SourceArn"]), (
            "aws:SourceArn must name a Bedrock ARN"
        )


class TestTheGrantIsOneStream:
    def test_it_can_only_write_and_only_to_one_stream(
        self, template: Template
    ) -> None:
        """Two actions, one stream. Not the group wildcard, not logs:*."""
        policies = [
            resource["Properties"]
            for resource in template.find_resources("AWS::IAM::Policy").values()
            if any(
                "ReasoningTrailDeliveryRole" in json.dumps(role)
                for role in resource["Properties"].get("Roles", [])
            )
        ]
        assert len(policies) == 1, "one identity policy on the delivery role"

        statements = policies[0]["PolicyDocument"]["Statement"]
        assert len(statements) == 1, "one statement"

        assert sorted(statements[0]["Action"]) == [
            "logs:CreateLogStream",
            "logs:PutLogEvents",
        ], f"only the two documented write actions; got {statements[0]['Action']!r}"

        resource_json = json.dumps(statements[0]["Resource"])
        assert f"log-stream:{EXPECTED_STREAM}" in resource_json, (
            "the grant must name the single documented log stream, so it cannot "
            f"write elsewhere in the group; got {resource_json}"
        )
        assert "logs:*" not in resource_json

    def test_the_stream_arn_is_not_malformed_by_a_wildcard(
        self, template: Template
    ) -> None:
        """A ":*" before ":log-stream:" makes the ARN invalid and Bedrock refuse.

        CDK's `log_group_arn` returns the group ARN with ":*" appended, so building
        the stream ARN from it yields "...:log-group:/name:*:log-stream:...". Bedrock
        rejects the whole configuration with a ValidationException that blames the
        role's permissions rather than the ARN shape, which makes this an expensive
        thing to rediscover.
        """
        policies = [
            resource["Properties"]
            for resource in template.find_resources("AWS::IAM::Policy").values()
            if any(
                "ReasoningTrailDeliveryRole" in json.dumps(role)
                for role in resource["Properties"].get("Roles", [])
            )
        ]
        resource_json = json.dumps(
            policies[0]["PolicyDocument"]["Statement"][0]["Resource"]
        )

        assert ":*:log-stream:" not in resource_json, (
            "the log-stream ARN carries a wildcard segment before ':log-stream:' — "
            "this is the malformed shape CDK's log_group_arn produces; assemble the "
            f"ARN from the group NAME instead. Got {resource_json}"
        )

    def test_the_role_holds_no_managed_policy(self, template: Template) -> None:
        """A managed policy would widen the grant invisibly to this test."""
        assert not _reasoning_role(template).get("ManagedPolicyArns"), (
            "the delivery role must carry only its inline statement"
        )


class TestCreatingTheDestinationLogsNothing:
    def test_no_custom_resource_enables_account_wide_logging(
        self, template: Template
    ) -> None:
        """The account-wide switch must never be flipped by a deploy.

        Model invocation logging covers EVERY Bedrock invocation in the account and
        Region. There is no CloudFormation resource for it, so the only way a deploy
        could enable it is a custom resource calling the API — which is exactly what
        this asserts is absent. Deploying this project must not begin recording
        unrelated workloads' prompts.
        """
        rendered = json.dumps(template.to_json())

        for forbidden in (
            "PutModelInvocationLoggingConfiguration",
            "putModelInvocationLoggingConfiguration",
        ):
            assert forbidden not in rendered, (
                f"{forbidden} appears in the template: a deploy would flip an "
                "account-wide logging switch covering every Bedrock invocation in "
                "the account. The switch must stay a deliberate manual act."
            )

    def test_no_bedrock_logging_config_resource_type(
        self, template: Template
    ) -> None:
        """Guard against a future native resource type being adopted silently."""
        types = {
            resource.get("Type", "")
            for resource in template.to_json().get("Resources", {}).values()
        }
        offenders = {
            resource_type
            for resource_type in types
            if "Bedrock" in resource_type and "Logging" in resource_type
        }
        assert not offenders, (
            f"a Bedrock logging resource appeared in the stack: {offenders}. If AWS "
            "has added one, the account-wide blast radius must be re-argued before "
            "adopting it."
        )
