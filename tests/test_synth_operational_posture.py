"""Synth assertions for the CDK operational posture.

This module pins five independent infrastructure concerns — none of them a live
vulnerability, all of them unsafe as a reused pattern — against the SYNTHESIZED
template of the real ``ToxicFlowStack``:

1. ``TestDocumentsTableDurability`` — point-in-time recovery and KMS encryption
   are on, and deletion protection stays OFF (it would deadlock teardown, which
   requires the table to stay destroyable; see ``cdk/data_resources.py``).
2. ``TestLambdaLogGroups`` — every first-party Lambda writes to a stack-owned
   log group with a finite retention that is removed on teardown, and the
   deprecated ``log_retention`` custom resource is not reintroduced.
3. ``TestToolReservedConcurrency`` — the three tool Lambdas carry a concurrency
   cap; the request-path interceptors deliberately do not.
4. ``TestSeedGrantIsMinimal`` — the seed function holds exactly the two write
   actions it performs, on the one table.
5. ``TestGatewayGrantIsPinned`` — no ``bedrock-agentcore`` gateway action is
   granted on ``"*"``.

These are pure synth assertions: no AWS call, no Docker build, no credentials.
"""

from __future__ import annotations

from typing import Any

import pytest
from aws_cdk.assertions import Template

from synth_helpers import (
    build_full_stack,
    function_by_name,
    function_role_logical_id,
    iam_actions_targeting_role,
)

# Mirrors observability.LAMBDA_LOG_RETENTION (RetentionDays.ONE_MONTH). Written
# out literally so the test fails if the constant is changed without thought.
EXPECTED_RETENTION_DAYS = 30

# Mirrors observability.TOOL_RESERVED_CONCURRENCY.
EXPECTED_TOOL_RESERVED_CONCURRENCY = 10

# The three tool Lambdas: the functions a tenant's requests actually fan out to,
# and therefore the ones a runaway caller could use to exhaust account
# concurrency.
TOOL_FUNCTION_NAMES = (
    "toxic-flow-read-document",
    "toxic-flow-search-documents",
    "toxic-flow-reply",
)

# Every first-party function that declares an explicit physical name.
NAMED_FIRST_PARTY_FUNCTIONS = (
    "toxic-flow-documents-seed",
    *TOOL_FUNCTION_NAMES,
    "toxic-flow-response-interceptor",
)

# The REQUEST interceptor's physical name is deliberately auto-generated (the
# Zip->Image package-type switch requires replacement), so it is located by its
# construct id prefix instead.
SESSION_GUARD_ID_PREFIX = "SessionGuardFunction"

# Actions ``grant_write_data`` used to confer on the seeder and that it never
# performs. Named individually so a regression report says which one came back.
FORBIDDEN_SEED_ACTIONS = frozenset({
    "dynamodb:UpdateItem",
    "dynamodb:DeleteItem",
    "dynamodb:DescribeTable",
    "dynamodb:GetItem",
    "dynamodb:Query",
    "dynamodb:Scan",
})

# The gateway control-plane actions that must stay scoped to the gateway ARN
# rather than ``"*"``.
PINNED_GATEWAY_ACTIONS = frozenset({
    "bedrock-agentcore:UpdateGateway",
    "bedrock-agentcore:GetGateway",
})


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth() -> tuple[Template, dict[str, Any]]:
    """Synthesize the real stack once for every assertion in this module.

    Returns:
        A ``(Template, resources_dict)`` tuple. The raw resources dict is
        returned alongside the ``Template`` because several assertions here read
        resource-level attributes (``DeletionPolicy``) and scan across resource
        types, which the ``Template`` matcher API does not express directly.
    """
    _stack, template = build_full_stack()
    return template, template.to_json()["Resources"]


def _as_list(value: Any) -> list[Any]:
    """Normalize a CloudFormation string-or-list field to a list.

    Args:
        value: A single value, a list of values, or ``None``.

    Returns:
        The value as a list (empty when ``value`` is ``None``).
    """
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _documents_table(resources: dict[str, Any]) -> dict[str, Any]:
    """Return the sole ``AWS::DynamoDB::Table`` resource.

    Args:
        resources: The template's ``Resources`` dict.

    Returns:
        The table resource dict (including ``DeletionPolicy``).

    Raises:
        AssertionError: If there is not exactly one table.
    """
    tables = [
        r for r in resources.values() if r["Type"] == "AWS::DynamoDB::Table"
    ]
    assert len(tables) == 1, f"expected exactly one table, got {len(tables)}"
    return tables[0]


def _log_groups(resources: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return every ``AWS::Logs::LogGroup`` keyed by logical id.

    Args:
        resources: The template's ``Resources`` dict.

    Returns:
        Mapping of logical id to log-group resource dict.
    """
    return {
        lid: r
        for lid, r in resources.items()
        if r["Type"] == "AWS::Logs::LogGroup"
    }


def _session_guard(resources: dict[str, Any]) -> dict[str, Any]:
    """Return the REQUEST interceptor function (auto-generated physical name).

    Args:
        resources: The template's ``Resources`` dict.

    Returns:
        The Lambda function resource dict.

    Raises:
        AssertionError: If it is not found exactly once.
    """
    matches = [
        r
        for lid, r in resources.items()
        if r["Type"] == "AWS::Lambda::Function"
        and lid.startswith(SESSION_GUARD_ID_PREFIX)
    ]
    assert len(matches) == 1, (
        f"expected one {SESSION_GUARD_ID_PREFIX}* Lambda, got {len(matches)}"
    )
    return matches[0]


def _policy_statements(resources: dict[str, Any]) -> list[tuple[str, dict]]:
    """Collect every IAM policy statement in the template with its owner id.

    Covers standalone ``AWS::IAM::Policy`` and ``AWS::IAM::ManagedPolicy``
    resources as well as inline ``Policies`` on ``AWS::IAM::Role``, so a
    statement cannot escape the wildcard scan by changing where it is attached.

    Args:
        resources: The template's ``Resources`` dict.

    Returns:
        A list of ``(logical_id, statement)`` tuples.
    """
    collected: list[tuple[str, dict]] = []
    for lid, resource in resources.items():
        kind = resource["Type"]
        props = resource["Properties"]
        if kind in ("AWS::IAM::Policy", "AWS::IAM::ManagedPolicy"):
            collected.extend(
                (lid, s) for s in props["PolicyDocument"].get("Statement", [])
            )
        elif kind == "AWS::IAM::Role":
            for inline in props.get("Policies", []):
                collected.extend(
                    (lid, s)
                    for s in inline["PolicyDocument"].get("Statement", [])
                )
    return collected


# ---------------------------------------------------------------------------
# Item 1 — DocumentsTable durability and protection
# ---------------------------------------------------------------------------


class TestDocumentsTableDurability:
    """The table declares PITR and KMS encryption, and stays deletable."""

    def test_point_in_time_recovery_is_enabled(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        _template, resources = synth
        spec = _documents_table(resources)["Properties"].get(
            "PointInTimeRecoverySpecification", {}
        )
        assert spec.get("PointInTimeRecoveryEnabled") is True, (
            f"PITR must be enabled, got {spec!r}"
        )

    def test_encryption_uses_a_kms_key(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        # TableEncryption.AWS_MANAGED synthesizes SSESpecification.SSEEnabled:
        # true — i.e. the aws/dynamodb managed key rather than the invisible
        # AWS-OWNED default. Absent SSESpecification means the owned key.
        _template, resources = synth
        spec = _documents_table(resources)["Properties"].get(
            "SSESpecification", {}
        )
        assert spec.get("SSEEnabled") is True, (
            f"table must use a KMS key, not the AWS-owned default, got {spec!r}"
        )

    def test_deletion_protection_stays_off_so_teardown_works(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        # Deletion protection and DeletionPolicy: Delete are INCOMPATIBLE —
        # DynamoDB refuses to delete a protected table, so enabling it would
        # break the teardown that test_synth_config.py pins: the table must stay
        # destroyable. Both halves are asserted together so the pair cannot
        # silently drift apart.
        _template, resources = synth
        table = _documents_table(resources)
        assert table.get("DeletionPolicy") == "Delete", (
            "DeletionPolicy must stay Delete so the table is destroyed on "
            f"teardown, got {table.get('DeletionPolicy')!r}"
        )
        assert not table["Properties"].get("DeletionProtectionEnabled"), (
            "DeletionProtectionEnabled must stay absent/false: it deadlocks "
            "stack teardown against DeletionPolicy: Delete"
        )


# ---------------------------------------------------------------------------
# Item 2 — Lambda log groups
# ---------------------------------------------------------------------------


class TestLambdaLogGroups:
    """Every first-party Lambda logs to a bounded, stack-owned group."""

    @pytest.mark.parametrize("function_name", NAMED_FIRST_PARTY_FUNCTIONS)
    def test_named_function_targets_a_stack_owned_log_group(
        self, synth: tuple[Template, dict[str, Any]], function_name: str
    ) -> None:
        template, resources = synth
        _lid, function = function_by_name(template, function_name)
        ref = function["Properties"].get("LoggingConfig", {}).get("LogGroup")
        assert isinstance(ref, dict) and "Ref" in ref, (
            f"{function_name} must declare LoggingConfig.LogGroup, got {ref!r}"
        )
        assert ref["Ref"] in _log_groups(resources), (
            f"{function_name} references {ref['Ref']!r}, which is not an "
            "AWS::Logs::LogGroup in this template"
        )

    def test_request_interceptor_targets_a_stack_owned_log_group(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        _template, resources = synth
        ref = (
            _session_guard(resources)["Properties"]
            .get("LoggingConfig", {})
            .get("LogGroup")
        )
        assert isinstance(ref, dict) and ref.get("Ref") in _log_groups(
            resources
        ), f"REQUEST interceptor log group is not stack-owned: {ref!r}"

    def test_every_log_group_has_the_expected_finite_retention(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        _template, resources = synth
        groups = _log_groups(resources)
        assert groups, "no AWS::Logs::LogGroup found — retention is unbounded"
        for lid, group in groups.items():
            assert (
                group["Properties"].get("RetentionInDays")
                == EXPECTED_RETENTION_DAYS
            ), (
                f"{lid} must retain {EXPECTED_RETENTION_DAYS} days, got "
                f"{group['Properties'].get('RetentionInDays')!r}"
            )

    def test_log_groups_are_removed_on_teardown(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        # RETAIN would orphan a group per function on every destroy/redeploy
        # cycle, and the next deploy would then collide with its own leftovers.
        _template, resources = synth
        groups = _log_groups(resources)
        assert groups, "no AWS::Logs::LogGroup found — nothing to tear down"
        for lid, group in groups.items():
            assert group.get("DeletionPolicy") == "Delete", (
                f"{lid} must be deleted on teardown, got "
                f"{group.get('DeletionPolicy')!r}"
            )

    def test_deprecated_log_retention_helper_is_not_used(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        # Function(log_retention=...) is deprecated and pays for retention with
        # a Custom::LogRetention resource whose role holds
        # logs:PutRetentionPolicy on "*" — a new account-wide wildcard, exactly
        # what the gateway grant assertions below rule out (control-plane
        # actions belong on a specific ARN, never on "*").
        _template, resources = synth
        offenders = [
            lid
            for lid, r in resources.items()
            if r["Type"] == "Custom::LogRetention"
        ]
        assert not offenders, (
            f"Custom::LogRetention reintroduced by {offenders!r}; use an "
            "explicit logs.LogGroup (cdk/observability.py) instead"
        )


# ---------------------------------------------------------------------------
# Item 3 — reserved concurrency
# ---------------------------------------------------------------------------


class TestToolReservedConcurrency:
    """The tool Lambdas are capped; the request-path interceptors are not."""

    @pytest.mark.parametrize("function_name", TOOL_FUNCTION_NAMES)
    def test_tool_function_declares_the_cap(
        self, synth: tuple[Template, dict[str, Any]], function_name: str
    ) -> None:
        template, _resources = synth
        _lid, function = function_by_name(template, function_name)
        assert (
            function["Properties"].get("ReservedConcurrentExecutions")
            == EXPECTED_TOOL_RESERVED_CONCURRENCY
        ), (
            f"{function_name} must reserve "
            f"{EXPECTED_TOOL_RESERVED_CONCURRENCY} concurrent executions, got "
            f"{function['Properties'].get('ReservedConcurrentExecutions')!r}"
        )

    def test_interceptors_are_deliberately_uncapped(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        # Both interceptors sit on EVERY gateway request/response, so a
        # reservation there converts a burst into gateway-visible throttling for
        # all tenants at once — the opposite of the isolation the tool caps buy.
        # Asserted rather than merely commented so the exclusion reads as a
        # decision, not an oversight.
        template, resources = synth
        _lid, response_fn = function_by_name(
            template, "toxic-flow-response-interceptor"
        )
        for label, function in (
            ("REQUEST interceptor", _session_guard(resources)),
            ("RESPONSE interceptor", response_fn),
        ):
            assert (
                "ReservedConcurrentExecutions" not in function["Properties"]
            ), (
                f"{label} must stay uncapped: reserving concurrency on the "
                "request path throttles every tenant at once"
            )


# ---------------------------------------------------------------------------
# Item 4 — the seed function's table grant
# ---------------------------------------------------------------------------


class TestSeedGrantIsMinimal:
    """The seeder holds exactly the two write actions it performs."""

    def test_seed_role_holds_only_batch_write_and_put(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        template, _resources = synth
        _lid, seed_fn = function_by_name(template, "toxic-flow-documents-seed")
        role = function_role_logical_id(seed_fn)
        granted = {
            action
            for action in iam_actions_targeting_role(template, role)
            if action.startswith("dynamodb:")
        }
        assert granted == {"dynamodb:BatchWriteItem", "dynamodb:PutItem"}, (
            "seed role must hold exactly BatchWriteItem + PutItem, got "
            f"{sorted(granted)!r}"
        )

    def test_seed_role_cannot_mutate_or_remove_documents(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        # The specific regression grant_write_data caused: a seeder that could
        # rewrite or destroy seeded documents rather than only (re)writing them.
        template, _resources = synth
        _lid, seed_fn = function_by_name(template, "toxic-flow-documents-seed")
        role = function_role_logical_id(seed_fn)
        leaked = FORBIDDEN_SEED_ACTIONS.intersection(
            iam_actions_targeting_role(template, role)
        )
        assert not leaked, f"seed role regained unused action(s): {sorted(leaked)!r}"

    def test_seed_grant_is_scoped_to_the_one_table(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        _template, resources = synth
        statements = [
            (lid, s)
            for lid, s in _policy_statements(resources)
            if any(
                a.startswith("dynamodb:BatchWriteItem")
                for a in _as_list(s.get("Action"))
            )
        ]
        assert statements, "no statement grants dynamodb:BatchWriteItem"
        for lid, statement in statements:
            resource = _as_list(statement.get("Resource"))
            assert "*" not in resource, (
                f"{lid} grants BatchWriteItem on '*' — pin it to the table ARN"
            )
            assert all(isinstance(entry, dict) for entry in resource), (
                f"{lid} must reference the table ARN intrinsic, got {resource!r}"
            )


# ---------------------------------------------------------------------------
# Item 5 — the gateway control-plane grant
# ---------------------------------------------------------------------------


class TestGatewayGrantIsPinned:
    """No gateway control-plane action is granted account-wide."""

    def test_no_gateway_action_is_granted_on_wildcard(
        self, synth: tuple[Template, dict[str, Any]]
    ) -> None:
        _template, resources = synth
        checked = 0
        for lid, statement in _policy_statements(resources):
            actions = set(_as_list(statement.get("Action")))
            matched = actions.intersection(PINNED_GATEWAY_ACTIONS)
            if not matched:
                continue
            checked += 1
            resource = _as_list(statement.get("Resource"))
            assert "*" not in resource, (
                f"{lid} grants {sorted(matched)!r} on '*' — pin it to "
                "gateway.gateway_arn"
            )
        assert checked, (
            "no statement grants UpdateGateway/GetGateway — the AttachPolicyEngine "
            "custom resource should still need them; this assertion has gone stale"
        )
