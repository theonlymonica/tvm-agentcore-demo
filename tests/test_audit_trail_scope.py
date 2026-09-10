"""The audit trail must stay narrow, because it bills per event.

``cdk/audit_trail.py`` exists so the DynamoDB access performed with a vended
credential is recorded at all, and so AgentCore's own data-plane invocations are
recorded by AWS rather than by the agent. Unlike everything else in this stack it
costs money per use, so the risk it carries is not a security hole but a bill: a
selector that widens from one table to every table, or a trail that starts duplicating
management events already free in Event history, would be invisible in review and
visible only on the invoice.

These tests pin the narrowings that bound the cost, the read+write coverage without
which half the audit chain would be missing, and the mutual exclusivity of the two
selector forms — which, if broken, makes CloudFormation reject the trail.

AWS documentation references:
    - DynamoDB data-plane events must be explicitly enabled:
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/logging-using-cloudtrail.html
    - InvokeAgentRuntime is a DATA event under AWS::BedrockAgentCore::Runtime:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-operations.html
    - The two selector forms cannot coexist on one trail:
      https://docs.aws.amazon.com/awscloudtrail/latest/APIReference/API_AdvancedEventSelector.html
    - Data-event pricing: https://aws.amazon.com/cloudtrail/pricing/
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from aws_cdk.assertions import Template

import synth_helpers as sh

DYNAMODB_TYPE = "AWS::DynamoDB::Table"
RUNTIME_TYPE = "AWS::BedrockAgentCore::Runtime"


@pytest.fixture(scope="module")
def template() -> Template:
    """Synthesize the real stack."""
    _stack, tmpl = sh.build_full_stack()
    return tmpl


def _trail(template: Template) -> dict[str, Any]:
    """Return the single trail's properties."""
    trails = [
        resource["Properties"]
        for resource in template.find_resources("AWS::CloudTrail::Trail").values()
    ]
    assert len(trails) == 1, (
        "exactly one CloudTrail trail — a second would double the data-event bill "
        f"for the same access; found {len(trails)}"
    )
    return trails[0]


def _selectors(template: Template) -> list[dict[str, Any]]:
    """Return the trail's advanced event selectors."""
    return _trail(template)["AdvancedEventSelectors"]


def _field(selector: dict[str, Any], field: str) -> list[str]:
    """Return the Equals values for one field of a selector, or []."""
    for field_selector in selector["FieldSelectors"]:
        if field_selector["Field"] == field:
            return field_selector.get("Equals", [])
    return []


class TestOnlyOneSelectorFormIsUsed:
    def test_basic_event_selectors_are_absent(self, template: Template) -> None:
        """Both forms on one trail is rejected by CloudFormation.

        The L2 Trail construct always emits EventSelectors, so it has to be actively
        removed. If it comes back, the deploy fails rather than degrading quietly —
        but it fails after a 60-second round trip, so catch it here.
        """
        assert "EventSelectors" not in _trail(template), (
            "EventSelectors must be deleted when AdvancedEventSelectors are used: "
            "a trail cannot carry both forms"
        )

    def test_advanced_selectors_are_present(self, template: Template) -> None:
        """Two selectors, no more: each one is a separate line on the bill."""
        assert len(_selectors(template)) == 2


class TestTheTableSelectorIsScopedToOneTable:
    def _table_selector(self, template: Template) -> dict[str, Any]:
        matching = [
            selector
            for selector in _selectors(template)
            if DYNAMODB_TYPE in _field(selector, "resources.type")
        ]
        assert len(matching) == 1, "exactly one DynamoDB selector"
        return matching[0]

    def test_it_selects_data_events(self, template: Template) -> None:
        """eventCategory Data — a management category here would be a cost bug."""
        assert _field(self._table_selector(template), "eventCategory") == ["Data"]

    def test_it_names_exactly_one_table_arn(self, template: Template) -> None:
        """One ARN, so the trail cannot silently start billing for other tables."""
        arns = _field(self._table_selector(template), "resources.ARN")
        assert len(arns) == 1, "one table ARN, not a list that can grow"

    def test_that_arn_is_this_stack_s_documents_table(
        self, template: Template
    ) -> None:
        """Asserted structurally: a CloudFormation reference to our own table."""
        rendered = json.dumps(
            _field(self._table_selector(template), "resources.ARN")
        )
        assert "DocumentsTable" in rendered, (
            f"must reference this stack's DocumentsTable; got {rendered}"
        )
        assert "*" not in rendered, "a wildcard would select every table in the account"

    def test_reads_and_writes_are_both_covered(self, template: Template) -> None:
        """No readOnly field means both, which is what the audit chain needs.

        read_document and search_documents read; reply writes. A readOnly:true
        selector would make the write half of the chain invisible, so its ABSENCE is
        the assertion.
        """
        assert _field(self._table_selector(template), "readOnly") == [], (
            "the selector must not restrict readOnly: reply's UpdateItem has to be "
            "recorded as well as the reads"
        )


class TestTheRuntimeSelectorRecordsInvocations:
    def _runtime_selector(self, template: Template) -> dict[str, Any]:
        matching = [
            selector
            for selector in _selectors(template)
            if RUNTIME_TYPE in _field(selector, "resources.type")
        ]
        assert len(matching) == 1, "exactly one AgentCore Runtime selector"
        return matching[0]

    def test_it_selects_data_events(self, template: Template) -> None:
        """InvokeAgentRuntime is a DATA event, so the category must say so.

        This is why it appears nowhere in Event history: lookup-events returns
        management events only.
        """
        assert _field(self._runtime_selector(template), "eventCategory") == ["Data"]

    def test_it_uses_the_documented_resource_type(self, template: Template) -> None:
        """The type is copied from the docs, not guessed.

        An unsupported resources.type either fails the deploy or selects nothing at
        all, and "selects nothing" is the failure mode that looks like success.
        """
        assert _field(self._runtime_selector(template), "resources.type") == [
            RUNTIME_TYPE
        ]


class TestNoGatewaySelectorIsInvented:
    def test_no_undocumented_agentcore_resource_type_appears(
        self, template: Template
    ) -> None:
        """Only the ONE AgentCore type that is documented may be selected.

        A Gateway data-event resource type is not documented. Naming one would be a
        guess, and a selector matching nothing is worse than no selector because it
        reads as coverage.
        """
        selected = {
            value
            for selector in _selectors(template)
            for value in _field(selector, "resources.type")
        }
        agentcore = {value for value in selected if "BedrockAgentCore" in value}
        assert agentcore == {RUNTIME_TYPE}, (
            "only the documented AWS::BedrockAgentCore::Runtime type may be "
            f"selected; found {agentcore}"
        )


class TestManagementEventsAreNotDuplicated:
    def test_no_selector_asks_for_management_events(
        self, template: Template
    ) -> None:
        """Management events are free in Event history; paying twice buys nothing.

        With advanced selectors there is no IncludeManagementEvents flag to set
        false — their absence is structural, so this asserts no selector names the
        Management category.
        """
        for selector in _selectors(template):
            assert "Management" not in _field(selector, "eventCategory"), (
                "this trail exists for DATA events; the AssumeRole event that "
                "carries SourceIdentity is already readable in Event history for 90 "
                "days at no cost"
            )


class TestTeardownLeavesNothing:
    def test_trail_bucket_is_destroyed_with_the_stack(
        self, template: Template
    ) -> None:
        """The trail is a verification aid, so it must not outlive a destroy."""
        buckets = {
            logical_id: resource
            for logical_id, resource in template.find_resources(
                "AWS::S3::Bucket"
            ).items()
            if logical_id.startswith("AuditTrailBucket")
        }
        assert len(buckets) == 1, "one bucket for the trail"
        assert next(iter(buckets.values())).get("DeletionPolicy") == "Delete"

        # auto_delete_objects empties the bucket on delete; without it the Delete
        # policy alone fails on a non-empty bucket.
        assert template.find_resources("Custom::S3AutoDeleteObjects"), (
            "auto_delete_objects must be enabled, or destroying the stack fails on "
            "the non-empty trail bucket"
        )
