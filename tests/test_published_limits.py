"""The published tool contract must state the limits the handlers enforce.

Extracted from ``tests/test_tool_result_bounds.py`` when the reply conversation cap
moved into the published tool description: these are statements about the
model-facing CONTRACT rather than about handler behaviour, and the host file was
over the repo's 400-line module limit.

Why the guard exists. The published ``Description`` / ``InputSchema`` is the only
contract the model writes its call from, so a limit that is enforced but not
published is discoverable only by making a call and failing it -- which costs one of
the agent's finite turns. Publishing it duplicates the number in
``cdk/gateway_resources.py``, and these tests are what stops the two copies
drifting: change a handler constant without updating the description and the guard
fails, naming the file to fix.

**Read against the synthesized template, not the CDK source.** The earlier version
of this guard substring-searched ``cdk/gateway_resources.py`` as text. That checks
the wrong artifact in both directions, and the first direction was hit in practice:
adding the conversation cap as an implicitly-concatenated string split
``"at most 50 "`` from ``"entries"``, so the guard failed while the published
description was exactly right -- a source-formatting question reported as a contract
violation. The converse is worse: a phrase sitting in a comment, a docstring, or a
different tool's description satisfies a file-wide text search while the contract
the model actually receives says something else. Resolving the tool payload out of
the synthesized ``AWS::BedrockAgentCore::GatewayTarget`` resource asserts the string
the gateway will publish, for the tool it belongs to.

Each assertion still pins the whole PHRASE rather than the bare number, because a
number alone can match coincidentally -- ``str(25)`` appears inside the ``2025`` of
the MCP protocol-version strings -- and a guard that can pass by accident is not a
guard.

Nothing here is a deployed artifact; these tests synthesize the stack in-process.
"""

from __future__ import annotations

import pytest
import synth_helpers as sh
from aws_cdk.assertions import Template

import reply.handler as reply_module
import search_documents.handler as search_module

_TARGET_TYPE = "AWS::BedrockAgentCore::GatewayTarget"


@pytest.fixture(scope="module")
def template() -> Template:
    """Synthesize the real stack once for the whole module.

    Returns:
        The synthesized ``aws_cdk.assertions.Template``.
    """
    _stack, synthesized = sh.build_full_stack()
    return synthesized


def _tool_payload(template: Template, target_name: str) -> dict:
    """Return the single inline tool definition published by a gateway target.

    Args:
        template: The synthesized template.
        target_name: The gateway target's ``Name`` (e.g. ``Reply``).

    Returns:
        The sole ``ToolSchema.InlinePayload`` entry for that target.

    Raises:
        AssertionError: If the target is absent or publishes more than one tool.
    """
    for resource in template.find_resources(_TARGET_TYPE).values():
        if resource["Properties"]["Name"] != target_name:
            continue
        inline = resource["Properties"]["TargetConfiguration"]["Mcp"]["Lambda"][
            "ToolSchema"
        ]["InlinePayload"]
        assert len(inline) == 1, (
            f"expected one inline tool definition on {target_name}, "
            f"got {len(inline)}"
        )
        return inline[0]
    raise AssertionError(f"no {_TARGET_TYPE} named {target_name!r} in the template")


def _tool_description(template: Template, target_name: str) -> str:
    """Return the published description of a target's tool.

    Args:
        template: The synthesized template.
        target_name: The gateway target's ``Name``.

    Returns:
        The tool-level ``Description`` string.
    """
    return _tool_payload(template, target_name)["Description"]


def _property_description(
    template: Template, target_name: str, property_name: str
) -> str:
    """Return the published description of one input-schema property.

    Args:
        template: The synthesized template.
        target_name: The gateway target's ``Name``.
        property_name: The ``InputSchema`` property to read.

    Returns:
        That property's ``Description`` string.
    """
    payload = _tool_payload(template, target_name)
    return payload["InputSchema"]["Properties"][property_name]["Description"]


class TestPublishedLimitsMatchEnforcedLimits:
    """Every enforced bound appears verbatim in the published tool contract."""

    def test_search_result_cap_is_published(self, template: Template) -> None:
        phrase = f"at most {search_module._MAX_RESULTS} matches"
        description = _tool_description(template, "SearchDocuments")
        assert phrase in description, (
            "the search_documents tool description must state the result cap as "
            f"{phrase!r}; published description is {description!r}"
        )

    def test_search_truncation_flag_is_published(self, template: Template) -> None:
        # The model can only act on `truncated` if it is told the field exists.
        description = _tool_description(template, "SearchDocuments")
        assert "truncated" in description, (
            "the search_documents tool description must tell the model it reports "
            f"a `truncated` flag; published description is {description!r}"
        )

    def test_reply_body_limit_is_published(self, template: Template) -> None:
        phrase = f"maximum {reply_module._MAX_BODY_BYTES} bytes"
        description = _property_description(template, "Reply", "body")
        assert phrase in description, (
            "the reply `body` property description must state the body limit as "
            f"{phrase!r}; published description is {description!r}"
        )

    def test_reply_conversation_cap_is_published(self, template: Template) -> None:
        # This limit used to be stated in the error the model received AFTER a
        # rejected write. It moved here when the same ConditionExpression gained a
        # second cause (the target document not existing), which made a message
        # naming the cap false in that case. The rule is constant, so the tool
        # description -- read BEFORE the model chooses to call -- is its proper home,
        # and this assertion is what keeps it from silently going stale.
        phrase = f"at most {reply_module._MAX_CONVERSATION_LEN} entries"
        description = _tool_description(template, "Reply")
        assert phrase in description, (
            "the reply tool description must state the conversation cap as "
            f"{phrase!r}; published description is {description!r}"
        )

    def test_reply_states_that_it_cannot_create_documents(
        self, template: Template
    ) -> None:
        # The handler's ConditionExpression refuses a write to a document that does
        # not exist. Publishing that keeps the model from spending turns inventing
        # ids, and keeps the contract honest about what the tool does.
        description = _tool_description(template, "Reply")
        assert "cannot create a document" in description, (
            "the reply tool description must tell the model it cannot create a "
            f"document; published description is {description!r}"
        )
