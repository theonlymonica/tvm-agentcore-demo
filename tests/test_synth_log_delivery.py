"""Synth guard: the gateway MUST NEVER deliver APPLICATION_LOGS.

Why this file exists
--------------------
The REQUEST interceptor vends short-lived tenant STS credentials to the tools
inside ``params.arguments["context"]`` (``interceptor/handler.py``), because a
Lambda target has no header channel. The gateway's vended ``APPLICATION_LOGS``
carry a ``requestBody`` field that includes ``params.arguments`` VERBATIM, so
enabling that delivery writes live ``secret_access_key`` + ``session_token``
values into CloudWatch.

This is not theoretical: a real run once had ``APPLICATION_LOGS`` delivery
temporarily enabled, and the vended request body DID carry
``tenant_credentials`` and all three credential field names (the values
themselves were never printed), after which the delivery was removed.

Until this file existed, the only thing standing between that log group and a
live cross-tenant credential leak was a ~25-line comment in
``cdk/gateway_wiring.py``. A comment cannot fail a build. These assertions turn
the prohibition into a mechanical one: the constraint now breaks the test suite
instead of breaking silently.

What is asserted
----------------
* Every ``PutDeliverySource`` call the stack synthesizes declares
  ``logType == "TRACES"`` — on Create AND on Update.
* Exactly ONE delivery source is declared, and it targets the sole gateway, so a
  second (differently named) source cannot be added unnoticed.
* The literal ``APPLICATION_LOGS`` appears NOWHERE in the synthesized template —
  a coarse net that also catches an L1 ``AWS::Logs::DeliverySource``, an inline
  Lambda body, or a parameter default that the structured checks do not model.
* The deploy-time custom-resource role's ``logs:PutDeliverySource`` permission is
  pinned to the single named delivery-source ARN rather than ``"*"``, so the
  deploy path itself cannot mint an extra APPLICATION_LOGS source.

Guard-has-teeth tests
---------------------
A prohibition test that cannot fail is worse than no test, because it manufactures
confidence. :class:`TestGuardHasTeeth` therefore runs the SAME check functions
against hand-built templates that deliberately violate each rule and asserts they
are flagged. If someone later weakens a check, those tests go red.

Note: the ``logType`` value cannot be constrained by IAM. ``logs:PutDeliverySource``
supports only tag condition keys plus ``logs:LogGeneratingResourceArns`` (IAM
service-authorization reference for CloudWatch Logs), so no policy or SCP can
express "deny logType=APPLICATION_LOGS". Synth-time assertion is therefore the
only enforceable layer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from aws_cdk.assertions import Template

import synth_helpers as sh

#: The delivery source logType the stack is allowed to declare. Vended spans do
#: not include ``params.arguments``, which is why TRACES is safe while
#: APPLICATION_LOGS is not.
ALLOWED_LOG_TYPE = "TRACES"

#: The prohibited logType (Amazon Bedrock AgentCore Gateway request-body logs).
FORBIDDEN_LOG_TYPE = "APPLICATION_LOGS"

#: The frozen delivery-source name created by ``enable_gateway_tracing``. The
#: deploy-time IAM statement is pinned to exactly this name.
TRACES_SOURCE_NAME = "toxic-flow-gateway-traces-source"


# ---------------------------------------------------------------------------
# Check functions — shared by the real-stack assertions and the teeth tests
# ---------------------------------------------------------------------------


def offending_delivery_sources(template: Template) -> list[str]:
    """Return a description of every ``PutDeliverySource`` call that is not TRACES.

    Sweeps EVERY ``Custom::AWS`` resource and all three lifecycle properties, so
    a second delivery source — or a Create that is TRACES paired with an Update
    that is not — cannot hide behind the first match.

    Args:
        template: The synthesized template.

    Returns:
        A list of human-readable descriptions of the offending calls. Empty means
        the template is clean.
    """
    offenders: list[str] = []
    for sdk_call in sh.custom_resource_sdk_calls(template, "PutDeliverySource"):
        log_type = sdk_call.call.get("parameters", {}).get("logType")
        if log_type != ALLOWED_LOG_TYPE:
            offenders.append(
                f"{sdk_call.logical_id}.{sdk_call.lifecycle} declares "
                f"logType={log_type!r}"
            )
    return offenders


def forbidden_log_type_occurrences(template: Template) -> int:
    """Return how many times the forbidden logType literal occurs in the template.

    Args:
        template: The synthesized template.

    Returns:
        The number of occurrences of ``APPLICATION_LOGS`` in the rendered
        template JSON. Zero is the only acceptable value.
    """
    return sh.template_text(template).count(FORBIDDEN_LOG_TYPE)


def offending_native_delivery_sources(template: Template) -> list[str]:
    """Return every native ``AWS::Logs::DeliverySource`` that is not TRACES.

    Why this exists alongside :func:`offending_delivery_sources`
    ------------------------------------------------------------
    That function walks ``Custom::AWS`` SDK calls, because the gateway's delivery
    source is wired with an ``AwsCustomResource``. It is blind to a delivery
    source declared as a *native* CloudFormation resource — and this stack has
    one: ``agentcore.Runtime(tracing_enabled=True)`` synthesizes
    ``AWS::Logs::DeliverySource`` / ``DeliveryDestination`` / ``Delivery`` for the
    agent runtime.

    Without this function that native resource is caught only incidentally, by
    :func:`forbidden_log_type_occurrences` sweeping the template for the literal.
    That is a single coarse check, and a future refinement should narrow it if a
    legitimate runtime APPLICATION_LOGS delivery is ever wanted. Such a narrowing
    would lean on the structured checks above staying intact — and they would not
    have been, because they never saw native resources at all. This closes that
    gap, so narrowing the coarse check later stays safe.

    Args:
        template: The synthesized template.

    Returns:
        Human-readable descriptions of the offending resources. Empty means clean.
    """
    offenders: list[str] = []
    resources = template.find_resources("AWS::Logs::DeliverySource")
    for logical_id, resource in resources.items():
        log_type = resource.get("Properties", {}).get("LogType")
        if log_type != ALLOWED_LOG_TYPE:
            offenders.append(f"{logical_id} declares LogType={log_type!r}")
    return offenders


def put_delivery_source_statements(template: Template) -> list[dict]:
    """Return every ``Allow`` statement that grants ``logs:PutDeliverySource``.

    Includes statements that grant it via a wildcard action (``logs:*`` / ``*``),
    since those confer the permission just as effectively as naming it.

    Args:
        template: The synthesized template.

    Returns:
        The matching statement dicts.
    """
    statements: list[dict] = []
    policies = {
        **template.find_resources("AWS::IAM::Policy"),
        **template.find_resources("AWS::IAM::ManagedPolicy"),
        **template.find_resources("AWS::IAM::Role"),
    }
    for resource in policies.values():
        for statement in _iter_statements(resource.get("Properties", {})):
            if statement.get("Effect") != "Allow":
                continue
            actions = _as_list(statement.get("Action"))
            if any(
                action in ("logs:PutDeliverySource", "logs:*", "*")
                for action in actions
            ):
                statements.append(statement)
    return statements


def unpinned_put_delivery_source_statements(template: Template) -> list[Any]:
    """Return ``logs:PutDeliverySource`` grants that are not name-pinned.

    A grant is "unpinned" when its ``Resource`` does not name the single frozen
    delivery source. ``logs:PutDeliverySource`` creates-or-updates by name, and
    the action supports resource-level permissions on
    ``arn:<partition>:logs:<region>:<account>:delivery-source:<name>``, so
    pinning the name is the tightest form IAM can express (there is no
    ``logType`` condition key — see the module docstring).

    Args:
        template: The synthesized template.

    Returns:
        The offending statement dicts. Empty means every grant is pinned.
    """
    return [
        statement
        for statement in put_delivery_source_statements(template)
        if TRACES_SOURCE_NAME not in json.dumps(statement.get("Resource"))
    ]


def _iter_statements(properties: dict) -> list[dict]:
    """Return every IAM statement reachable from a resource's properties.

    Covers the inline ``PolicyDocument`` of ``AWS::IAM::Policy`` /
    ``AWS::IAM::ManagedPolicy`` and the ``Policies`` list of an inline role
    policy.

    Args:
        properties: The resource's ``Properties`` dict.

    Returns:
        A flat list of statement dicts.
    """
    statements: list[dict] = []
    document = properties.get("PolicyDocument")
    if isinstance(document, dict):
        statements.extend(_as_list(document.get("Statement")))
    for policy in _as_list(properties.get("Policies")):
        inner = policy.get("PolicyDocument") if isinstance(policy, dict) else None
        if isinstance(inner, dict):
            statements.extend(_as_list(inner.get("Statement")))
    return [s for s in statements if isinstance(s, dict)]


def _as_list(value: Any) -> list:
    """Return ``value`` as a list (IAM allows scalar or list in most fields).

    Args:
        value: A scalar, a list, or ``None``.

    Returns:
        ``[]`` for ``None``, ``value`` unchanged if already a list, else
        ``[value]``.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# ---------------------------------------------------------------------------
# The real stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_stack_template() -> Template:
    """Synthesize the real ``ToxicFlowStack`` once and return its template.

    Returns:
        The ``Template`` for the production stack (the same synth path used by
        ``tests/test_synth_config_wiring.py``).
    """
    _, template = sh.build_full_stack()
    return template


class TestGatewayDeliversTracesOnly:
    """The credential-in-payload channel must never be logged."""

    def test_every_put_delivery_source_declares_traces(
        self, full_stack_template: Template
    ) -> None:
        offenders = offending_delivery_sources(full_stack_template)
        assert not offenders, (
            "every gateway delivery source must declare logType=TRACES; "
            f"vended APPLICATION_LOGS would capture params.arguments (and the "
            f"tenant credentials riding in it) verbatim. Offending calls: "
            f"{offenders}"
        )

    def test_exactly_one_delivery_source_is_declared(
        self, full_stack_template: Template
    ) -> None:
        # Count Create calls only: Update re-declares the same logical source.
        creates = [
            call
            for call in sh.custom_resource_sdk_calls(
                full_stack_template, "PutDeliverySource"
            )
            if call.lifecycle == "Create"
        ]
        assert len(creates) == 1, (
            "the stack must declare exactly ONE gateway delivery source (the "
            f"TRACES one); found {len(creates)}: "
            f"{[c.logical_id for c in creates]}"
        )
        assert creates[0].call["parameters"]["name"] == TRACES_SOURCE_NAME

    def test_delivery_source_targets_the_sole_gateway(
        self, full_stack_template: Template
    ) -> None:
        # The guard is only meaningful if the source it constrains is THIS
        # gateway's. resourceArn is a CloudFormation intrinsic, which
        # join_to_str flattens to a placeholder, so inspect the raw value.
        gateways = full_stack_template.find_resources(
            "AWS::BedrockAgentCore::Gateway"
        )
        assert len(gateways) == 1, (
            f"expected exactly one gateway, got {len(gateways)}"
        )
        gateway_logical_id = next(iter(gateways))

        creates = [
            call
            for call in sh.custom_resource_sdk_calls(
                full_stack_template, "PutDeliverySource"
            )
            if call.lifecycle == "Create"
        ]
        raw_text = json.dumps(creates[0].raw)
        assert gateway_logical_id in raw_text, (
            "the TRACES delivery source must target the sole gateway "
            f"({gateway_logical_id}); its resourceArn references none of it: "
            f"{raw_text}"
        )

    def test_forbidden_log_type_appears_nowhere_in_template(
        self, full_stack_template: Template
    ) -> None:
        occurrences = forbidden_log_type_occurrences(full_stack_template)
        assert occurrences == 0, (
            f"the literal {FORBIDDEN_LOG_TYPE!r} must not appear anywhere in the "
            f"synthesized template (found {occurrences} occurrence(s)); it would "
            "mean some resource can deliver gateway request bodies"
        )

    def test_native_delivery_source_resources_exist(
        self, full_stack_template: Template
    ) -> None:
        # Non-vacuity for the assertion below: the runtime's tracing wiring
        # synthesizes AWS::Logs::DeliverySource natively, so an empty set would
        # mean the check has nothing to constrain (and would pass silently if the
        # runtime construct changed shape).
        resources = full_stack_template.find_resources("AWS::Logs::DeliverySource")
        assert resources, (
            "expected at least one native AWS::Logs::DeliverySource (the agent "
            "runtime's TRACES delivery from tracing_enabled=True); found none, so "
            "the native-resource assertion below would pass vacuously"
        )

    def test_every_native_delivery_source_declares_traces(
        self, full_stack_template: Template
    ) -> None:
        offenders = offending_native_delivery_sources(full_stack_template)
        assert not offenders, (
            "every native AWS::Logs::DeliverySource must declare LogType=TRACES; "
            "the Custom::AWS sweep above cannot see these resources, so this is "
            f"their only structural check. Offending resources: {offenders}"
        )


class TestPutDeliverySourceIsPinned:
    """Defence in depth — the deploy role cannot mint an extra delivery source."""

    def test_the_grant_exists_at_all(self, full_stack_template: Template) -> None:
        # Non-vacuity: the pinning assertion below is satisfied by an empty set,
        # so prove the permission is actually present in the template first.
        # (If the tracing wiring is ever removed, this test is the one to delete.)
        statements = put_delivery_source_statements(full_stack_template)
        assert statements, (
            "expected the deploy-time custom-resource role to grant "
            "logs:PutDeliverySource; found no such statement, so the pinning "
            "assertion below would pass vacuously"
        )

    def test_put_delivery_source_permission_names_the_traces_source(
        self, full_stack_template: Template
    ) -> None:
        offenders = unpinned_put_delivery_source_statements(full_stack_template)
        assert not offenders, (
            "logs:PutDeliverySource must be pinned to the single "
            f"{TRACES_SOURCE_NAME!r} delivery-source ARN, otherwise the "
            "deploy-time custom-resource role can create an additional "
            "APPLICATION_LOGS source for the gateway. Offending statements: "
            f"{offenders}"
        )


# ---------------------------------------------------------------------------
# Guard-has-teeth: the checks above must actually reject a bad template
# ---------------------------------------------------------------------------


def _custom_resource_template(sdk_call: dict, lifecycle: str = "Create") -> Template:
    """Build a minimal template holding one ``Custom::AWS`` SDK call.

    Args:
        sdk_call: The SDK-call dict to embed (serialized as JSON, exactly as
            ``AwsCustomResource`` does).
        lifecycle: Which lifecycle property to put it in.

    Returns:
        A ``Template`` over the synthetic CloudFormation body.
    """
    return Template.from_json({
        "Resources": {
            "FakeCustomResource": {
                "Type": "Custom::AWS",
                "Properties": {lifecycle: json.dumps(sdk_call)},
            }
        }
    })


def _put_delivery_source_call(log_type: str) -> dict:
    """Return a ``PutDeliverySource`` SDK-call dict with the given logType.

    Args:
        log_type: The ``logType`` parameter value.

    Returns:
        The SDK-call dict.
    """
    return {
        "service": "CloudWatchLogs",
        "action": "PutDeliverySource",
        "parameters": {
            "name": TRACES_SOURCE_NAME,
            "logType": log_type,
            "resourceArn": "arn:aws:bedrock-agentcore:us-east-1:1234:gateway/g",
        },
    }


def _native_delivery_source_template(log_type: str) -> Template:
    """Build a minimal template holding one native ``AWS::Logs::DeliverySource``.

    Args:
        log_type: The ``LogType`` property value.

    Returns:
        A ``Template`` over the synthetic CloudFormation body.
    """
    return Template.from_json({
        "Resources": {
            "RuntimeTracesDeliverySource": {
                "Type": "AWS::Logs::DeliverySource",
                "Properties": {
                    "Name": "runtime-source",
                    "LogType": log_type,
                    "ResourceArn": (
                        "arn:aws:bedrock-agentcore:us-east-1:1234:runtime/r"
                    ),
                },
            }
        }
    })


class TestGuardHasTeeth:
    """The checks reject deliberately-bad templates (not vacuously passing)."""

    def test_native_application_logs_source_is_flagged(self) -> None:
        # The gap the native check closes: a delivery source declared as a
        # NATIVE CloudFormation resource, which the Custom::AWS sweep cannot see.
        # The runtime construct already emits one of these, so this is a
        # reachable shape, not a hypothetical.
        template = _native_delivery_source_template(FORBIDDEN_LOG_TYPE)
        assert offending_native_delivery_sources(template)
        # Proof that the Custom::AWS sweep is genuinely blind to it — this is
        # WHY the native check had to be added rather than assumed covered.
        assert offending_delivery_sources(template) == []

    def test_native_traces_source_is_not_flagged(self) -> None:
        template = _native_delivery_source_template(ALLOWED_LOG_TYPE)
        assert offending_native_delivery_sources(template) == []

    def test_application_logs_create_is_flagged(self) -> None:
        template = _custom_resource_template(
            _put_delivery_source_call(FORBIDDEN_LOG_TYPE)
        )
        assert offending_delivery_sources(template)
        assert forbidden_log_type_occurrences(template) == 1

    def test_application_logs_hidden_in_update_is_flagged(self) -> None:
        # The realistic drift: Create stays TRACES, Update is edited. A check
        # that read only Create would pass here.
        template = Template.from_json({
            "Resources": {
                "GatewayTracesSource": {
                    "Type": "Custom::AWS",
                    "Properties": {
                        "Create": json.dumps(
                            _put_delivery_source_call(ALLOWED_LOG_TYPE)
                        ),
                        "Update": json.dumps(
                            _put_delivery_source_call(FORBIDDEN_LOG_TYPE)
                        ),
                    },
                }
            }
        })
        offenders = offending_delivery_sources(template)
        assert len(offenders) == 1
        assert "Update" in offenders[0]

    def test_second_application_logs_source_is_flagged(self) -> None:
        # The bypass the singular custom_resource_sdk_call helper allows: the
        # FIRST PutDeliverySource is clean, a second one is not.
        template = Template.from_json({
            "Resources": {
                "GatewayTracesSource": {
                    "Type": "Custom::AWS",
                    "Properties": {
                        "Create": json.dumps(
                            _put_delivery_source_call(ALLOWED_LOG_TYPE)
                        )
                    },
                },
                "GatewayAppLogsSource": {
                    "Type": "Custom::AWS",
                    "Properties": {
                        "Create": json.dumps(
                            _put_delivery_source_call(FORBIDDEN_LOG_TYPE)
                        )
                    },
                },
            }
        })
        assert len(offending_delivery_sources(template)) == 1

    def test_clean_template_is_not_flagged(self) -> None:
        template = _custom_resource_template(
            _put_delivery_source_call(ALLOWED_LOG_TYPE)
        )
        assert offending_delivery_sources(template) == []
        assert forbidden_log_type_occurrences(template) == 0

    def test_wildcard_put_delivery_source_statement_is_flagged(self) -> None:
        template = Template.from_json({
            "Resources": {
                "FakePolicy": {
                    "Type": "AWS::IAM::Policy",
                    "Properties": {
                        "PolicyDocument": {
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": ["logs:PutDeliverySource"],
                                    "Resource": "*",
                                }
                            ]
                        }
                    },
                }
            }
        })
        assert len(unpinned_put_delivery_source_statements(template)) == 1

    def test_wildcard_action_covering_put_delivery_source_is_flagged(self) -> None:
        # logs:* grants PutDeliverySource just as effectively as naming it.
        template = Template.from_json({
            "Resources": {
                "FakeRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "Policies": [
                            {
                                "PolicyDocument": {
                                    "Statement": [
                                        {
                                            "Effect": "Allow",
                                            "Action": "logs:*",
                                            "Resource": "*",
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                }
            }
        })
        assert len(unpinned_put_delivery_source_statements(template)) == 1

    def test_pinned_statement_is_not_flagged(self) -> None:
        template = Template.from_json({
            "Resources": {
                "FakePolicy": {
                    "Type": "AWS::IAM::Policy",
                    "Properties": {
                        "PolicyDocument": {
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": ["logs:PutDeliverySource"],
                                    "Resource": (
                                        "arn:aws:logs:us-east-1:1234:"
                                        f"delivery-source:{TRACES_SOURCE_NAME}"
                                    ),
                                }
                            ]
                        }
                    },
                }
            }
        })
        assert unpinned_put_delivery_source_statements(template) == []
