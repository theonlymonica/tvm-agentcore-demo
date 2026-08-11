"""CDK synth/config tests for interceptor wiring + frozen names.

Companion to ``tests/test_synth_config.py``. The two files exist because they
need different stack fixtures. That file's assertions (DocumentsTable schema +
published tool schemas) run against a MINIMAL stub stack, because neither the
table nor the tool schemas depend on the Docker-image Lambdas. The assertions
here need the REAL ``ScopedCredentialsStack`` instead — the RESPONSE interceptor's
execution role, the frozen ``scoped-credentials-*`` function names, and the per-Lambda
runtimes only exist on the production stack — so they share a single
module-scoped full-stack synth fixture.

What is asserted here:

* Exactly ONE RESPONSE interceptor is wired at BOTH declaration points: the
  create-time gateway ``InterceptorConfigurations`` AND the ``AttachPolicyEngine``
  ``UpdateGateway`` payload (which re-declares the full interceptor list and
  would otherwise silently drop it).
* ``pass_request_headers=True`` remains on the REQUEST interceptor at both
  declaration points.
* The RESPONSE interceptor's execution role holds no DynamoDB permission and no
  ``sts:AssumeRole``.
* The stack is ``ScopedCredentialsStack`` and every explicitly named Lambda follows the
  frozen ``scoped-credentials-*`` prefix.
* Every Python Lambda uses ``python3.14``; the container-image REQUEST
  interceptor carries no ``Runtime`` and ``PackageType: Image``.

The interceptor's ``COGNITO_ALLOWED_CLIENT_IDS`` / gateway ``allowedClients``
agreement is asserted separately, in
``tests/test_interceptor_client_pinning.py``.

Synthesis approach
------------------
Full-stack: ``synth_helpers.build_full_stack`` instantiates the real
``ScopedCredentialsStack`` (with a fixed account/region so no environment lookup is
needed) and returns ``(stack, Template.from_stack(stack))``. Synthesis is slower
than the stub stack because the stack stages the container-image REQUEST
interceptor and the agent Runtime image assets.
"""

from __future__ import annotations

import pytest
from aws_cdk.assertions import Template

import synth_helpers as sh


@pytest.fixture(scope="module")
def full_stack() -> tuple[object, Template]:
    """Synthesize the real ``ScopedCredentialsStack`` once; return ``(stack, template)``.

    Returns:
        The ``(stack, Template)`` tuple from ``synth_helpers.build_full_stack``.
        The ``stack`` is returned so the frozen stack-name assertion can read
        ``stack.stack_name`` (the stack name is cloud-assembly metadata and is
        not present in the template body).
    """
    return sh.build_full_stack()


# ---------------------------------------------------------------------------
# Exactly one RESPONSE interceptor wired, at BOTH declaration points
# ---------------------------------------------------------------------------


class TestResponseInterceptorWiring:
    """One RESPONSE interceptor at create time AND in the UpdateGateway payload."""

    def test_create_time_configs_have_one_request_one_response(
        self, full_stack: tuple[object, Template]
    ) -> None:
        _, template = full_stack
        configs = sh.gateway_interceptor_configs(template)
        request = [c for c in configs if c["InterceptionPoints"] == ["REQUEST"]]
        response = [c for c in configs if c["InterceptionPoints"] == ["RESPONSE"]]
        assert len(request) == 1, (
            f"expected one REQUEST interceptor at create time, got {len(request)}"
        )
        assert len(response) == 1, (
            f"expected exactly one RESPONSE interceptor at create time, "
            f"got {len(response)}"
        )

    def test_update_gateway_payload_redeclares_one_response(
        self, full_stack: tuple[object, Template]
    ) -> None:
        # The AttachPolicyEngine UpdateGateway call re-declares the full
        # interceptor list, so it MUST carry the RESPONSE entry alongside REQUEST
        # or it silently drops it (a split-brain vs. the template).
        _, template = full_stack
        call = sh.custom_resource_sdk_call(template, "UpdateGateway")
        configs = call["parameters"]["interceptorConfigurations"]
        request = [c for c in configs if c["interceptionPoints"] == ["REQUEST"]]
        response = [c for c in configs if c["interceptionPoints"] == ["RESPONSE"]]
        assert len(request) == 1, (
            f"UpdateGateway must re-declare one REQUEST interceptor, "
            f"got {len(request)}"
        )
        assert len(response) == 1, (
            f"UpdateGateway must re-declare exactly one RESPONSE interceptor, "
            f"got {len(response)}"
        )


# ---------------------------------------------------------------------------
# pass_request_headers=True remains on the REQUEST interceptor
# ---------------------------------------------------------------------------


class TestPassRequestHeaders:
    """``pass_request_headers=True`` is present at both declaration points."""

    def test_create_time_request_passes_headers(
        self, full_stack: tuple[object, Template]
    ) -> None:
        _, template = full_stack
        configs = sh.gateway_interceptor_configs(template)
        request = next(c for c in configs if c["InterceptionPoints"] == ["REQUEST"])
        assert request["InputConfiguration"]["PassRequestHeaders"] is True

    def test_update_gateway_request_passes_headers(
        self, full_stack: tuple[object, Template]
    ) -> None:
        _, template = full_stack
        call = sh.custom_resource_sdk_call(template, "UpdateGateway")
        configs = call["parameters"]["interceptorConfigurations"]
        request = next(c for c in configs if c["interceptionPoints"] == ["REQUEST"])
        assert request["inputConfiguration"]["passRequestHeaders"] is True


# ---------------------------------------------------------------------------
# RESPONSE interceptor role holds no DynamoDB and no sts:AssumeRole
# ---------------------------------------------------------------------------


class TestResponseInterceptorRole:
    """The RESPONSE interceptor's execution role is data-plane- and STS-free."""

    def test_role_has_no_dynamodb_and_no_assume_role(
        self, full_stack: tuple[object, Template]
    ) -> None:
        _, template = full_stack
        _, fn = sh.function_by_name(template, "scoped-credentials-response-interceptor")
        role_logical_id = sh.function_role_logical_id(fn)
        actions = sh.iam_actions_targeting_role(template, role_logical_id)

        dynamodb_actions = [a for a in actions if a.startswith("dynamodb:")]
        assert not dynamodb_actions, (
            f"RESPONSE interceptor role must hold no DynamoDB permission, "
            f"found {dynamodb_actions}"
        )
        assert "sts:AssumeRole" not in actions, (
            "RESPONSE interceptor role must hold no sts:AssumeRole permission"
        )


# ---------------------------------------------------------------------------
# Frozen stack + function names
# ---------------------------------------------------------------------------


class TestFrozenStackAndFunctionNames:
    """Stack is ``ScopedCredentialsStack``; Lambda names follow ``scoped-credentials-*``."""

    def test_stack_name_is_frozen(self, full_stack: tuple[object, Template]) -> None:
        stack, _ = full_stack
        assert stack.stack_name == sh.FROZEN_STACK_NAME

    def test_every_named_lambda_uses_frozen_prefix(
        self, full_stack: tuple[object, Template]
    ) -> None:
        _, template = full_stack
        named = [
            res["Properties"]["FunctionName"]
            for res in sh.lambda_functions(template).values()
            if "FunctionName" in res["Properties"]
        ]
        offenders = [n for n in named if not n.startswith(sh.FROZEN_FUNCTION_PREFIX)]
        assert not offenders, (
            f"every explicit Lambda function name must start with "
            f"{sh.FROZEN_FUNCTION_PREFIX!r}, offenders: {offenders}"
        )

    def test_required_function_names_present(
        self, full_stack: tuple[object, Template]
    ) -> None:
        _, template = full_stack
        named = {
            res["Properties"]["FunctionName"]
            for res in sh.lambda_functions(template).values()
            if "FunctionName" in res["Properties"]
        }
        missing = sh.REQUIRED_FUNCTION_NAMES - named
        assert not missing, f"missing frozen function name(s): {missing}"


# ---------------------------------------------------------------------------
# Every Python Lambda is PYTHON_3_14; the REQUEST interceptor is an image
# ---------------------------------------------------------------------------


class TestLambdaRuntimes:
    """Python Lambdas use ``python3.14``; the REQUEST interceptor is an image.

    Scope note: CDK auto-synthesizes Node.js helper Lambdas for its
    custom-resource providers (the framework ``onEvent`` handler and the AWS SDK
    call Lambda). Those are framework internals, NOT project Python components, so
    the "Python 3.14 everywhere" rule governs only Lambdas with a *Python*
    runtime. This asserts every Python-runtime Lambda is exactly ``python3.14``
    (guarding against 3.11/3.12/3.13) and that the container-image REQUEST
    interceptor carries no ``Runtime`` and ``PackageType: Image``.
    """

    def test_every_python_lambda_is_python_3_14(
        self, full_stack: tuple[object, Template]
    ) -> None:
        _, template = full_stack
        python_runtimes = {
            logical_id: res["Properties"]["Runtime"]
            for logical_id, res in sh.lambda_functions(template).items()
            if str(res["Properties"].get("Runtime", "")).startswith("python")
        }
        assert python_runtimes, "expected at least one Python Lambda in the stack"
        offenders = {
            lid: rt
            for lid, rt in python_runtimes.items()
            if rt != sh.FROZEN_PYTHON_RUNTIME
        }
        assert not offenders, (
            f"every Python Lambda must use {sh.FROZEN_PYTHON_RUNTIME}, "
            f"offenders: {offenders}"
        )

    def test_named_python_lambdas_are_python_3_14(
        self, full_stack: tuple[object, Template]
    ) -> None:
        # The three tools + seed + RESPONSE interceptor are all zip PYTHON_3_14.
        _, template = full_stack
        for name in sh.REQUIRED_FUNCTION_NAMES:
            _, fn = sh.function_by_name(template, name)
            assert fn["Properties"].get("Runtime") == sh.FROZEN_PYTHON_RUNTIME, (
                f"{name} must use {sh.FROZEN_PYTHON_RUNTIME}"
            )

    def test_request_interceptor_is_container_image(
        self, full_stack: tuple[object, Template]
    ) -> None:
        # The REQUEST interceptor is a DockerImageFunction: PackageType Image and
        # no Runtime property (its handler is the image CMD).
        _, template = full_stack
        image_fns = [
            res
            for res in sh.lambda_functions(template).values()
            if res["Properties"].get("PackageType") == "Image"
        ]
        assert len(image_fns) == 1, (
            f"expected exactly one container-image Lambda (the REQUEST "
            f"interceptor), got {len(image_fns)}"
        )
        assert "Runtime" not in image_fns[0]["Properties"], (
            "the container-image REQUEST interceptor must declare no Runtime"
        )
