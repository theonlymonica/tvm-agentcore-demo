"""Shared CDK synth/config test helpers.

This module was extracted from ``tests/test_synth_config.py`` to keep that file
under the 400-line limit. It holds two groups of helpers shared by the
synth/config assertions:

* ``build_full_stack`` — synthesizes the REAL ``ToxicFlowStack`` (the full
  production stack, packaging the container-image REQUEST interceptor and the
  agent Runtime image) so the interceptor-wiring, frozen-name and
  all-Lambda-runtime assertions run against the exact template the deployed
  stack produces — not the minimal stub stack used by the data/tool-schema
  assertions.
* CloudFormation-template inspection helpers — locate the sole gateway, its
  create-time ``InterceptorConfigurations``, the ``AttachPolicyEngine``
  ``UpdateGateway`` SDK-call payload (embedded as an ``Fn::Join`` JSON string),
  the Lambda functions, and the IAM actions granted to a given execution role.
* Prohibition helpers — ``custom_resource_sdk_calls`` (EVERY ``Custom::AWS`` SDK
  call, across the Create/Update/Delete lifecycle properties) and
  ``template_text``. Assertions that a configuration must NEVER appear need to
  sweep the whole template rather than the first match, so they cannot use the
  singular ``custom_resource_sdk_call``: a second custom resource repeating the
  action — an ``APPLICATION_LOGS`` delivery source alongside the permitted one —
  would sit behind the first match and never be inspected (that prohibition is
  asserted in ``tests/test_synth_log_delivery.py``).

Import-path note
----------------
The ``cdk/`` modules import each other flat (``from toxic_flow_stack import
...``), so ``cdk/`` must be on ``sys.path``. This mirrors the setup already used
by ``tests/test_synth_config.py`` and ``tests/test_seed.py``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Import-path setup: put cdk/ on sys.path so the flat cdk imports resolve.
# ---------------------------------------------------------------------------

_CDK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cdk"
)
if _CDK_DIR not in sys.path:
    sys.path.insert(0, _CDK_DIR)

import aws_cdk as cdk  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402

# The frozen stack name.
FROZEN_STACK_NAME = "ToxicFlowStack"

# The frozen Lambda function-name prefix and the four function names pinned
# explicitly.
FROZEN_FUNCTION_PREFIX = "toxic-flow-"
REQUIRED_FUNCTION_NAMES = {
    "toxic-flow-response-interceptor",
    "toxic-flow-read-document",
    "toxic-flow-search-documents",
    "toxic-flow-reply",
}

# The frozen Python runtime for every Python Lambda.
FROZEN_PYTHON_RUNTIME = "python3.14"


# ---------------------------------------------------------------------------
# Full-stack synthesis
# ---------------------------------------------------------------------------


def build_full_stack() -> tuple[cdk.Stack, Template]:
    """Synthesize the real ``ToxicFlowStack`` and return ``(stack, template)``.

    Instantiates the production stack (not the minimal stub used by the
    data/tool-schema assertions) so the interceptor wiring, RESPONSE-interceptor
    execution role, frozen names, and per-Lambda runtimes are exactly those the
    deployed stack emits. A fixed ``env`` (account/region) is supplied so no
    environment lookup is required at synth time.

    Returns:
        A ``(stack, Template)`` tuple. The ``stack`` is returned alongside the
        ``Template`` so the frozen stack-name assertion can read
        ``stack.stack_name`` (the stack name is cloud-assembly metadata, not part
        of the template body).
    """
    from toxic_flow_stack import ToxicFlowStack

    app = cdk.App()
    stack = ToxicFlowStack(
        app,
        FROZEN_STACK_NAME,
        env=cdk.Environment(account="123456789012", region="us-east-1"),
    )
    return stack, Template.from_stack(stack)


# ---------------------------------------------------------------------------
# Fn::Join reconstruction (for the AwsCustomResource SDK-call payload)
# ---------------------------------------------------------------------------


def _fragment_to_str(part: Any) -> str:
    """Render one ``Fn::Join`` array element to a string for JSON reconstruction.

    Literal string fragments pass through unchanged. CloudFormation intrinsics
    (``Fn::GetAtt`` / ``Ref`` for the interceptor Lambda ARNs, the gateway id,
    the role ARN) always appear *inside* a JSON string value — between the
    surrounding quote characters — so replacing each with a fixed placeholder
    preserves JSON validity while discarding only the ARN/id value (which these
    assertions never inspect).

    Args:
        part: One element of an ``Fn::Join`` value array.

    Returns:
        The string to substitute for this fragment.
    """
    if isinstance(part, str):
        return part
    return "INTRINSIC"


def join_to_str(value: Any) -> str:
    """Reconstruct a possibly ``Fn::Join`` template value into a single string.

    Args:
        value: Either a plain string (all-literal SDK call) or an
            ``{"Fn::Join": [sep, [parts...]]}`` structure (the usual case, where
            ARNs/ids are CloudFormation intrinsics).

    Returns:
        The reconstructed string, with intrinsics replaced by a placeholder.

    Raises:
        AssertionError: If ``value`` is neither a string nor an ``Fn::Join``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "Fn::Join" in value:
        separator, parts = value["Fn::Join"]
        return separator.join(_fragment_to_str(p) for p in parts)
    raise AssertionError(f"unexpected non-string, non-Fn::Join value: {value!r}")


def custom_resource_sdk_call(template: Template, action: str) -> dict:
    """Return the parsed SDK-call dict for the ``Custom::AWS`` resource whose
    on-create call invokes ``action``.

    ``AwsCustomResource`` serializes its ``AwsSdkCall`` (service/action/
    parameters) into the ``Create`` property as a JSON string, with any ARNs/ids
    spliced in via ``Fn::Join``. This reconstructs that string and parses it.

    Args:
        template: The synthesized template.
        action: The SDK action to match (e.g. ``"UpdateGateway"``).

    Returns:
        The parsed SDK-call dict, including its ``"parameters"`` sub-dict.

    Raises:
        AssertionError: If no matching ``Custom::AWS`` resource is found.
    """
    for resource in template.find_resources("Custom::AWS").values():
        create = resource["Properties"].get("Create")
        if create is None:
            continue
        call = json.loads(join_to_str(create))
        if call.get("action") == action:
            return call
    raise AssertionError(f"no Custom::AWS resource invokes action {action!r}")


#: The three ``AwsCustomResource`` lifecycle properties that each hold a
#: serialized ``AwsSdkCall``. An assertion that inspects only ``Create`` would
#: miss a call reachable on stack UPDATE, so guard tests must sweep all three.
SDK_CALL_LIFECYCLES = ("Create", "Update", "Delete")


class SdkCall(NamedTuple):
    """One ``AwsSdkCall`` found in a synthesized ``Custom::AWS`` resource.

    Attributes:
        logical_id: The ``Custom::AWS`` resource's logical id (for failure
            messages that point at the offending construct).
        lifecycle: Which lifecycle property held the call — ``"Create"``,
            ``"Update"`` or ``"Delete"``.
        call: The parsed SDK-call dict (``service`` / ``action`` /
            ``parameters``).
        raw: The UNPARSED template value for the lifecycle property (a string or
            an ``Fn::Join`` structure). Assertions that must inspect the
            CloudFormation intrinsics themselves — which ``join_to_str``
            deliberately flattens to a placeholder — read this instead of
            ``call``.
    """

    logical_id: str
    lifecycle: str
    call: dict
    raw: Any


def custom_resource_sdk_calls(
    template: Template, action: str | None = None
) -> list[SdkCall]:
    """Return EVERY ``Custom::AWS`` SDK call in the template, all lifecycles.

    The singular :func:`custom_resource_sdk_call` returns the FIRST ``Create``
    call matching an action, which is the wrong shape for a *prohibition*: a
    second custom resource invoking the same action (for example a second
    ``PutDeliverySource``, this time for ``APPLICATION_LOGS``) would never be
    inspected and the assertion would pass while the template was unsafe. Guard
    tests therefore enumerate with this function and assert over the whole list.

    Args:
        template: The synthesized template.
        action: Optional SDK action to filter on (e.g. ``"PutDeliverySource"``).
            ``None`` returns every call found.

    Returns:
        A list of :class:`SdkCall` tuples, in template iteration order. Empty if
        nothing matches — callers that require presence must assert on the
        length themselves.
    """
    calls: list[SdkCall] = []
    for logical_id, resource in template.find_resources("Custom::AWS").items():
        properties = resource.get("Properties", {})
        for lifecycle in SDK_CALL_LIFECYCLES:
            raw = properties.get(lifecycle)
            if raw is None:
                continue
            parsed = json.loads(join_to_str(raw))
            if action is None or parsed.get("action") == action:
                calls.append(SdkCall(logical_id, lifecycle, parsed, raw))
    return calls


def template_text(template: Template) -> str:
    """Return the whole synthesized template as one JSON string.

    Used for belt-and-braces "this literal must appear nowhere" assertions,
    which catch an occurrence in a resource type the structured helpers do not
    model (a raw ``AWS::Logs::DeliverySource`` L1, an inline Lambda body, a
    CloudFormation parameter default).

    Args:
        template: The synthesized template.

    Returns:
        The template rendered as JSON text.
    """
    return json.dumps(template.to_json())


# ---------------------------------------------------------------------------
# Gateway / Lambda / IAM inspection
# ---------------------------------------------------------------------------


def gateway_interceptor_configs(template: Template) -> list[dict]:
    """Return the sole gateway's create-time ``InterceptorConfigurations`` list.

    Args:
        template: The synthesized template.

    Returns:
        The ``Properties.InterceptorConfigurations`` list of the single
        ``AWS::BedrockAgentCore::Gateway`` resource.

    Raises:
        AssertionError: If there is not exactly one gateway resource.
    """
    gateways = template.find_resources("AWS::BedrockAgentCore::Gateway")
    assert len(gateways) == 1, f"expected exactly one gateway, got {len(gateways)}"
    gateway = next(iter(gateways.values()))
    return gateway["Properties"]["InterceptorConfigurations"]


def lambda_functions(template: Template) -> dict[str, dict]:
    """Return all ``AWS::Lambda::Function`` resources keyed by logical id.

    Args:
        template: The synthesized template.

    Returns:
        Mapping of logical id to the resource dict.
    """
    return template.find_resources("AWS::Lambda::Function")


def function_by_name(template: Template, function_name: str) -> tuple[str, dict]:
    """Return ``(logical_id, resource)`` for the Lambda with ``FunctionName``.

    Args:
        template: The synthesized template.
        function_name: The explicit ``FunctionName`` to match.

    Returns:
        A ``(logical_id, resource_dict)`` tuple.

    Raises:
        AssertionError: If no Lambda declares that ``FunctionName``.
    """
    for logical_id, resource in lambda_functions(template).items():
        if resource["Properties"].get("FunctionName") == function_name:
            return logical_id, resource
    raise AssertionError(f"no Lambda function named {function_name!r}")


def function_role_logical_id(function_resource: dict) -> str:
    """Return the logical id of a Lambda's execution role.

    The ``Role`` property of a Lambda whose role CDK auto-creates is an
    ``{"Fn::GetAtt": ["<RoleLogicalId>", "Arn"]}`` reference.

    Args:
        function_resource: A Lambda function resource dict.

    Returns:
        The referenced IAM role's logical id.

    Raises:
        AssertionError: If the ``Role`` property is not a ``Fn::GetAtt`` ref.
    """
    role = function_resource["Properties"]["Role"]
    if isinstance(role, dict) and "Fn::GetAtt" in role:
        return role["Fn::GetAtt"][0]
    raise AssertionError(f"unexpected Role reference shape: {role!r}")


def _statement_actions(policy_document: dict) -> list[str]:
    """Flatten every ``Action`` (string or list) across a policy document.

    Args:
        policy_document: An IAM policy document with a ``Statement`` list.

    Returns:
        A flat list of action strings.
    """
    actions: list[str] = []
    for statement in policy_document.get("Statement", []):
        action = statement.get("Action", [])
        if isinstance(action, str):
            actions.append(action)
        else:
            actions.extend(action)
    return actions


def _roles_reference(roles: list, role_logical_id: str) -> bool:
    """Return True if a policy ``Roles`` list references ``role_logical_id``.

    Args:
        roles: The ``Properties.Roles`` list of an ``AWS::IAM::Policy``.
        role_logical_id: The role logical id to look for.

    Returns:
        True if any entry is ``{"Ref": role_logical_id}``.
    """
    return any(
        isinstance(entry, dict) and entry.get("Ref") == role_logical_id
        for entry in roles
    )


def iam_actions_targeting_role(
    template: Template, role_logical_id: str
) -> list[str]:
    """Collect every IAM action granted to ``role_logical_id``.

    Gathers actions from both the role's own inline ``Policies`` and every
    ``AWS::IAM::Policy`` resource whose ``Roles`` reference the role. (Managed
    policy ARNs such as ``AWSLambdaBasicExecutionRole`` carry no DynamoDB or
    ``sts:AssumeRole`` action and are not expanded here.)

    Args:
        template: The synthesized template.
        role_logical_id: The execution role's logical id.

    Returns:
        A flat list of every action string granted to that role.
    """
    actions: list[str] = []

    for resource in template.find_resources("AWS::IAM::Policy").values():
        roles = resource["Properties"].get("Roles", [])
        if _roles_reference(roles, role_logical_id):
            actions.extend(_statement_actions(resource["Properties"]["PolicyDocument"]))

    role = template.find_resources("AWS::IAM::Role").get(role_logical_id)
    if role is not None:
        for inline in role["Properties"].get("Policies", []):
            actions.extend(_statement_actions(inline["PolicyDocument"]))

    return actions
