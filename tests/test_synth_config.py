"""CDK synth/config tests for the data model + published tool schemas.

These assertions validate the *synthesized CloudFormation template* on two
surfaces: the ``scope`` / ``doc_id`` data model, and the tool schemas the
gateway publishes to the agent.

* The DocumentsTable ``KeySchema`` / ``AttributeDefinitions`` are ``scope``
  (HASH / S) and ``doc_id`` (RANGE / S) — NOT ``served_scope`` /
  ``document_id`` — and its ``DeletionPolicy`` is ``Delete`` (from
  ``RemovalPolicy.DESTROY``).
* Each tool ``input_schema`` declares EXACTLY its allowed properties and no
  others: ``doc_id`` for ``read_document``; ``query`` for ``search_documents``;
  ``doc_id`` + ``body`` for ``reply``. No ``served_scope``, no ``context``, no
  retired ``document_id``.
* The gateway target names (``ReadDocument`` / ``SearchDocuments`` / ``Reply``),
  tool names (``read_document`` / ``search_documents`` / ``reply``), and
  composite names (``ReadDocument___read_document`` etc.) are frozen and
  unchanged.

Synthesis approach (minimal stack)
----------------------------------
The full ``ToxicFlowStack`` packages the REQUEST interceptor as a Docker *image*
Lambda and the agent as a Docker *image* Runtime; synthesizing it triggers two
container builds. Neither Docker image influences the DocumentsTable schema or
the published tool schemas — those come entirely from ``create_data_resources``
(``cdk/data_resources.py``) and ``create_gateway`` (``cdk/gateway_resources.py``).

This module therefore builds a minimal ``cdk.Stack`` that invokes those SAME
production factory functions (plus ``create_policy_engine``, which
``create_gateway`` requires), substituting a zip-packaged stub Lambda for the
Docker-image interceptor and stub zip Lambdas for the three tool targets. The
assertions are NOT weakened: the DynamoDB table and the three
``AWS::BedrockAgentCore::GatewayTarget`` tool schemas are produced by the exact
production code the deployed stack runs.

Full-stack interceptor wiring
-----------------------------
The interceptor-wiring assertions (the RESPONSE interceptor's execution role,
the frozen ``toxic-flow-*`` function names, the per-Lambda runtimes) need the
REAL ``ToxicFlowStack``, because those resources do not exist on this file's
stub stack. They live in the companion module
``tests/test_synth_config_wiring.py``, sharing the extracted helpers in
``tests/synth_helpers.py``. Run both files together (e.g. ``pytest
tests/test_synth_config.py tests/test_synth_config_wiring.py``).

Import-path note
----------------
The ``cdk/`` modules import each other flat (``from data_resources import ...``),
so ``cdk/`` must be on ``sys.path`` — the repository ``conftest.py`` adds the
repo root and ``tools/`` but not ``cdk/``. This module prepends ``cdk/`` exactly
as ``tests/test_seed.py`` does.
"""

from __future__ import annotations

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Import-path setup: put cdk/ on sys.path so the flat cdk imports resolve.
# ---------------------------------------------------------------------------

_CDK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cdk"
)
if _CDK_DIR not in sys.path:
    sys.path.insert(0, _CDK_DIR)

import aws_cdk as cdk  # noqa: E402
import aws_cdk.aws_lambda as lambda_  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402

from auth_resources import create_auth_resources  # noqa: E402
from data_resources import create_data_resources  # noqa: E402
from gateway_resources import create_gateway  # noqa: E402
from policy_resources import create_policy_engine  # noqa: E402


# ---------------------------------------------------------------------------
# Expected (frozen) values — the article's wire contract.
# ---------------------------------------------------------------------------

# DocumentsTable composite key (partition + sort).
_EXPECTED_KEY_SCHEMA = {"HASH": "scope", "RANGE": "doc_id"}
_EXPECTED_ATTR_DEFS = {"scope": "S", "doc_id": "S"}
_RETIRED_KEY_NAMES = {"served_scope", "document_id"}

# Each tool declares EXACTLY these properties, keyed by the frozen gateway
# target name.
_EXPECTED_TOOLS = {
    "ReadDocument": {
        "tool_name": "read_document",
        "properties": {"doc_id"},
        "required": {"doc_id"},
    },
    "SearchDocuments": {
        "tool_name": "search_documents",
        "properties": {"query"},
        "required": {"query"},
    },
    "Reply": {
        "tool_name": "reply",
        "properties": {"doc_id", "body"},
        "required": {"doc_id", "body"},
    },
}

# The frozen composite tool names ({target}___{tool}).
_EXPECTED_COMPOSITE = {
    "ReadDocument": "ReadDocument___read_document",
    "SearchDocuments": "SearchDocuments___search_documents",
    "Reply": "Reply___reply",
}

# Property names that must never appear in any tool schema.
_FORBIDDEN_PROPERTIES = {"served_scope", "context", "document_id"}


# ---------------------------------------------------------------------------
# Minimal-stack synthesis (module-scoped: synthesize once, assert many).
# ---------------------------------------------------------------------------


def _stub_lambda(scope: cdk.Stack, construct_id: str) -> lambda_.Function:
    """Create a zip-packaged PYTHON_3_14 stub Lambda (no Docker build).

    The stub stands in for the Docker-image REQUEST interceptor and for the
    three tool Lambdas. Its packaging is irrelevant to the DocumentsTable
    schema and the published tool schemas, which are the subject under test.

    Args:
        scope: The stack to attach the function to.
        construct_id: Unique construct id for the function.

    Returns:
        A minimal ``lambda_.Function`` construct.
    """
    return lambda_.Function(
        scope,
        construct_id,
        runtime=lambda_.Runtime.PYTHON_3_14,  # PYTHON_3_14 only
        handler="index.handler",
        code=lambda_.Code.from_inline("def handler(event, context):\n    return {}\n"),
    )


@pytest.fixture(scope="module")
def synth_template() -> Template:
    """Synthesize the minimal stack once and return the CloudFormation template.

    Invokes the real ``create_data_resources`` + ``create_policy_engine`` +
    ``create_gateway`` production factories on a fresh stack, then returns
    ``Template.from_stack`` for assertions.

    Returns:
        The synthesized ``aws_cdk.assertions.Template``.
    """
    app = cdk.App()
    stack = cdk.Stack(
        app,
        "TestSynthConfigStack",
        env=cdk.Environment(account="123456789012", region="us-east-1"),
    )

    # Real data layer: DocumentsTable + scoped roles + seed custom resource.
    create_data_resources(stack)

    # Real managed Cognito identity layer: create_gateway needs the CUSTOM_JWT
    # discovery URL + app client id from it.
    auth = create_auth_resources(stack)

    # create_gateway requires a PolicyEngine; create the real one (no Docker).
    policy_engine = create_policy_engine(scope=stack)

    interceptor_fn = _stub_lambda(stack, "StubInterceptor")
    tool_fns = {
        "read_document": _stub_lambda(stack, "StubReadDocument"),
        "search_documents": _stub_lambda(stack, "StubSearchDocuments"),
        "reply": _stub_lambda(stack, "StubReply"),
    }

    # Real gateway + real tool-target schema construction.
    create_gateway(
        scope=stack,
        interceptor_fn=interceptor_fn,
        policy_engine=policy_engine,
        auth=auth,
        tool_fns=tool_fns,
    )

    return Template.from_stack(stack)


# ---------------------------------------------------------------------------
# Helpers to reach into the synthesized resources.
# ---------------------------------------------------------------------------


def _single_documents_table(template: Template) -> dict:
    """Return the sole ``AWS::DynamoDB::Table`` resource dict.

    Args:
        template: The synthesized template.

    Returns:
        The resource dict (with ``Type``/``Properties``/``DeletionPolicy``).
    """
    tables = template.find_resources("AWS::DynamoDB::Table")
    assert len(tables) == 1, f"expected exactly one DynamoDB table, got {len(tables)}"
    return next(iter(tables.values()))


def _gateway_targets_by_name(template: Template) -> dict[str, dict]:
    """Return the gateway-target resources keyed by their ``Properties.Name``.

    Args:
        template: The synthesized template.

    Returns:
        Mapping of target name (e.g. ``ReadDocument``) to the resource dict.
    """
    targets = template.find_resources("AWS::BedrockAgentCore::GatewayTarget")
    by_name: dict[str, dict] = {}
    for resource in targets.values():
        name = resource["Properties"]["Name"]
        by_name[name] = resource
    return by_name


def _tool_payload(target_resource: dict) -> dict:
    """Return the single inline tool definition from a gateway-target resource.

    Args:
        target_resource: A gateway-target resource dict.

    Returns:
        The first (and only) ``ToolSchema.InlinePayload`` entry.
    """
    inline = (
        target_resource["Properties"]["TargetConfiguration"]["Mcp"]["Lambda"][
            "ToolSchema"
        ]["InlinePayload"]
    )
    assert len(inline) == 1, f"expected one inline tool definition, got {len(inline)}"
    return inline[0]


# ===========================================================================
# DocumentsTable schema + deletion policy
# ===========================================================================


class TestDocumentsTableSchema:
    """The DocumentsTable uses the ``scope`` / ``doc_id`` composite key."""

    def test_key_schema_is_scope_and_doc_id(self, synth_template: Template) -> None:
        props = _single_documents_table(synth_template)["Properties"]
        key_map = {e["KeyType"]: e["AttributeName"] for e in props["KeySchema"]}
        assert key_map == _EXPECTED_KEY_SCHEMA, (
            f"KeySchema must be scope (HASH) / doc_id (RANGE), got {key_map!r}"
        )

    def test_attribute_definitions_are_scope_and_doc_id(
        self, synth_template: Template
    ) -> None:
        props = _single_documents_table(synth_template)["Properties"]
        attr_map = {
            e["AttributeName"]: e["AttributeType"]
            for e in props["AttributeDefinitions"]
        }
        assert attr_map == _EXPECTED_ATTR_DEFS, (
            f"AttributeDefinitions must be scope (S) / doc_id (S), got {attr_map!r}"
        )

    def test_no_retired_key_attribute_names(self, synth_template: Template) -> None:
        props = _single_documents_table(synth_template)["Properties"]
        names = {e["AttributeName"] for e in props["KeySchema"]}
        names |= {e["AttributeName"] for e in props["AttributeDefinitions"]}
        leaked = names & _RETIRED_KEY_NAMES
        assert not leaked, f"retired key attribute name(s) present: {leaked!r}"

    def test_deletion_policy_is_delete(self, synth_template: Template) -> None:
        # RemovalPolicy.DESTROY synthesizes to DeletionPolicy: Delete.
        table = _single_documents_table(synth_template)
        assert table.get("DeletionPolicy") == "Delete", (
            f"DeletionPolicy must be Delete, got {table.get('DeletionPolicy')!r}"
        )


# ===========================================================================
# Published tool schemas declare EXACTLY their allowed properties
# ===========================================================================


class TestToolSchemas:
    """Each tool ``input_schema`` declares exactly its allowed properties."""

    def test_exactly_three_targets_with_frozen_names(
        self, synth_template: Template
    ) -> None:
        by_name = _gateway_targets_by_name(synth_template)
        assert set(by_name) == set(_EXPECTED_TOOLS), (
            f"gateway target names must be frozen {set(_EXPECTED_TOOLS)}, "
            f"got {set(by_name)}"
        )

    @pytest.mark.parametrize("target_name", sorted(_EXPECTED_TOOLS))
    def test_tool_declares_exactly_allowed_properties(
        self, synth_template: Template, target_name: str
    ) -> None:
        expected = _EXPECTED_TOOLS[target_name]
        payload = _tool_payload(_gateway_targets_by_name(synth_template)[target_name])
        declared = set(payload["InputSchema"]["Properties"].keys())
        assert declared == expected["properties"], (
            f"{target_name} must declare exactly {expected['properties']}, "
            f"got {declared}"
        )

    @pytest.mark.parametrize("target_name", sorted(_EXPECTED_TOOLS))
    def test_tool_required_matches_allowed_properties(
        self, synth_template: Template, target_name: str
    ) -> None:
        expected = _EXPECTED_TOOLS[target_name]
        payload = _tool_payload(_gateway_targets_by_name(synth_template)[target_name])
        required = set(payload["InputSchema"]["Required"])
        assert required == expected["required"], (
            f"{target_name} required must be {expected['required']}, got {required}"
        )

    def test_no_forbidden_properties_in_any_tool_schema(
        self, synth_template: Template
    ) -> None:
        # No served_scope, no document_id, no context — in any tool schema.
        by_name = _gateway_targets_by_name(synth_template)
        for target_name, resource in by_name.items():
            declared = set(
                _tool_payload(resource)["InputSchema"]["Properties"].keys()
            )
            leaked = declared & _FORBIDDEN_PROPERTIES
            assert not leaked, (
                f"{target_name} schema declares forbidden propert(y/ies) {leaked}"
            )


# ===========================================================================
# Frozen target / tool / composite names
# ===========================================================================


class TestFrozenNames:
    """Target names, tool names, and composite names are unchanged."""

    @pytest.mark.parametrize("target_name", sorted(_EXPECTED_TOOLS))
    def test_tool_name_matches_frozen_value(
        self, synth_template: Template, target_name: str
    ) -> None:
        payload = _tool_payload(_gateway_targets_by_name(synth_template)[target_name])
        expected_tool = _EXPECTED_TOOLS[target_name]["tool_name"]
        assert payload["Name"] == expected_tool, (
            f"{target_name} tool name must be {expected_tool!r}, "
            f"got {payload['Name']!r}"
        )

    @pytest.mark.parametrize("target_name", sorted(_EXPECTED_TOOLS))
    def test_composite_name_is_frozen(
        self, synth_template: Template, target_name: str
    ) -> None:
        # The gateway forms the composite tool name as {targetName}___{toolName}.
        payload = _tool_payload(_gateway_targets_by_name(synth_template)[target_name])
        composite = f"{target_name}___{payload['Name']}"
        assert composite == _EXPECTED_COMPOSITE[target_name], (
            f"composite name must be {_EXPECTED_COMPOSITE[target_name]!r}, "
            f"got {composite!r}"
        )
