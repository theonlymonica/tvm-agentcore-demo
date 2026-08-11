"""Synth assertion: the interceptor's client pinning matches the gateway's.

``interceptor/jwt_claims.py`` re-checks the access token's ``client_id`` against
the ``COGNITO_ALLOWED_CLIENT_IDS`` env var so its trust boundary does not depend
on the gateway ``CUSTOM_JWT`` authorizer's ``allowedClients`` staying correct.
That defence in depth only holds while the two agree, and they are wired in two
different files — so this module asserts the two template values are the SAME
reference. Real drift therefore fails the suite: point the gateway at a different
app client and forget the interceptor and every request fails closed there; change
only the interceptor and the local check silently stops narrowing anything.

Compared as CloudFormation references, not strings
-------------------------------------------------
The pool and app client are MANAGED by this stack, so the client id
is a deploy-time token — ``{"Ref": "UserPoolAppClient..."}`` in the synthesized
template, not a literal. ``AuthResources`` says so explicitly ("they must never be
compared against a hardcoded id"), so these assertions compare the intrinsic
structures for equality. That is strictly stronger than a string match: two
different literals could coincide, but two references are equal only when they
resolve to the same resource.

All three declaration points are covered — the interceptor's environment, the
create-time gateway ``AuthorizerConfiguration``, and the ``AttachPolicyEngine``
``UpdateGateway`` payload, which re-declares the whole authorizer config and is
therefore a third place the client can drift.

Why a separate module (``code-modularity``)
-------------------------------------------
These assertions belong topically with ``tests/test_synth_config_wiring.py``, but
that file is already ~270 lines and adding them pushed it to ~395 — one edit from
the workspace 400-line HARD limit. It was itself split out of
``tests/test_synth_config.py`` for exactly that reason, so this follows the same
precedent and reuses the same ``tests/synth_helpers.py`` inspection helpers.
"""

from __future__ import annotations

from typing import Any

import pytest
from aws_cdk.assertions import Template

import synth_helpers as sh


@pytest.fixture(scope="module")
def full_stack() -> tuple[object, Template]:
    """Synthesize the real ``ScopedCredentialsStack`` once for this module.

    Returns:
        The ``(stack, Template)`` tuple from ``synth_helpers.build_full_stack``.
    """
    return sh.build_full_stack()


def _request_interceptor_env(template: Template) -> dict:
    """Return the container-image REQUEST interceptor's environment variables.

    The REQUEST interceptor deliberately declares no ``FunctionName`` (a fixed name
    would collide during the Zip->Image replacement), so it is located by
    ``PackageType: Image`` — the same discriminator
    ``test_request_interceptor_is_container_image`` uses, which also asserts it is
    unique.

    Args:
        template: The synthesized template.

    Returns:
        The ``Environment.Variables`` mapping.

    Raises:
        AssertionError: If there is not exactly one container-image Lambda.
    """
    image_fns = [
        res
        for res in sh.lambda_functions(template).values()
        if res["Properties"].get("PackageType") == "Image"
    ]
    assert len(image_fns) == 1, (
        f"expected exactly one container-image Lambda, got {len(image_fns)}"
    )
    return image_fns[0]["Properties"]["Environment"]["Variables"]


def _interceptor_client_ref(template: Template) -> Any:
    """Return the interceptor's configured client-id value.

    Args:
        template: The synthesized template.

    Returns:
        The ``COGNITO_ALLOWED_CLIENT_IDS`` value: a CloudFormation intrinsic dict
        under the managed pool, or a plain string if ever hardcoded.

    Raises:
        AssertionError: If the variable is absent or empty.
    """
    env = _request_interceptor_env(template)
    assert "COGNITO_ALLOWED_CLIENT_IDS" in env, (
        "the REQUEST interceptor needs COGNITO_ALLOWED_CLIENT_IDS — jwt_claims "
        "fails closed without it, so a missing var rejects every request"
    )
    value = env["COGNITO_ALLOWED_CLIENT_IDS"]
    assert value, "COGNITO_ALLOWED_CLIENT_IDS must not be empty (fails closed)"
    return value


def _join_intrinsics(value: Any) -> list[Any]:
    """Return the non-literal fragments of an ``Fn::Join`` value.

    ``AwsCustomResource`` serializes its SDK call to a JSON string with references
    spliced in as intrinsics, so the payload's client id survives only as one of
    these fragments (``synth_helpers.join_to_str`` deliberately discards them).

    Args:
        value: A template value, normally ``{"Fn::Join": [sep, [parts...]]}``.

    Returns:
        Every fragment that is not a literal string; empty if not an ``Fn::Join``.
    """
    if not isinstance(value, dict) or "Fn::Join" not in value:
        return []
    _, parts = value["Fn::Join"]
    return [part for part in parts if not isinstance(part, str)]


class TestInterceptorClientPinning:
    """The interceptor pins the SAME app client the gateway authorizer allows."""

    def test_allowed_client_ids_is_wired(
        self, full_stack: tuple[object, Template]
    ) -> None:
        # jwt_claims fails CLOSED on an unset value, so a missing env var here
        # would reject 100% of requests in the deployed stack.
        _, template = full_stack
        assert _interceptor_client_ref(template) is not None

    def test_matches_gateway_allowed_clients_at_creation(
        self, full_stack: tuple[object, Template]
    ) -> None:
        _, template = full_stack
        gateways = template.find_resources("AWS::BedrockAgentCore::Gateway")
        assert len(gateways) == 1, f"expected one gateway, got {len(gateways)}"
        allowed = next(iter(gateways.values()))["Properties"][
            "AuthorizerConfiguration"
        ]["CustomJWTAuthorizer"]["AllowedClients"]

        assert allowed == [_interceptor_client_ref(template)], (
            "the interceptor's COGNITO_ALLOWED_CLIENT_IDS must be the same "
            f"reference as the gateway authorizer's AllowedClients, got {allowed!r}"
        )

    def test_matches_gateway_allowed_clients_in_update_payload(
        self, full_stack: tuple[object, Template]
    ) -> None:
        # The AttachPolicyEngine UpdateGateway call re-declares the authorizer
        # config, so it is a third place the client list can drift. Its payload is
        # a JSON string with references spliced in, so the assertion is that the
        # interceptor's reference is among the spliced fragments.
        _, template = full_stack
        client_ref = _interceptor_client_ref(template)
        create = next(
            resource["Properties"]["Create"]
            for resource in template.find_resources("Custom::AWS").values()
            if "UpdateGateway" in sh.join_to_str(resource["Properties"]["Create"])
        )
        payload = sh.join_to_str(create)
        assert '"allowedClients"' in payload, (
            "the UpdateGateway payload must still declare allowedClients"
        )
        assert client_ref in _join_intrinsics(create), (
            "the UpdateGateway payload's allowedClients must reference the same "
            "app client the interceptor pins"
        )
