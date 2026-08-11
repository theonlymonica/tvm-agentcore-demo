"""Auth-posture tests for the MANAGED Cognito identity layer.

The pool and app client used to be IMPORTED by literal id
(``UserPool.from_user_pool_id``), which put the entire security floor of the
identity layer outside CloudFormation and outside this test suite: MFA, password
policy, token validity, ``PreventUserExistenceErrors``, self-signup and the
allowed auth flows were all defined out-of-band and could drift silently. The
identity layer is now created as managed constructs rather than imported; these
tests are the drift detection that makes that posture enforceable.

What is asserted:

* the pool exists exactly once and pins MFA, password policy, self-signup,
  account recovery, feature plan and a destroy-on-teardown deletion policy;
* the app client allows **SRP only** — ``ALLOW_USER_PASSWORD_AUTH`` and
  ``ALLOW_ADMIN_USER_PASSWORD_AUTH`` must never appear — with
  ``PreventUserExistenceErrors=ENABLED``, short token validity, revocation on,
  no client secret and no OAuth/hosted-UI flows;
* one Cognito group per scope in ``SCOPE_GROUPS`` (group membership is the
  ``served_scope`` grant, so the groups are security-relevant resources);
* the deploy-time identifiers are published as stack outputs; and
* the gateway CUSTOM_JWT authorizer REFERENCES the managed pool/client rather
  than carrying a hardcoded id, and no retired import-by-id call survives
  anywhere in ``cdk/``.

The template is synthesized from the real ``create_auth_resources`` +
``create_gateway`` factories on a minimal stack (a zip stub Lambda stands in for
the container-image interceptor) so no Docker build is required.
"""

from __future__ import annotations

import ast
import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Import-path setup: the cdk/ modules import each other flat, so cdk/ must be on
# sys.path (same pattern as tests/synth_helpers.py and tests/test_seed.py).
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CDK_DIR = os.path.join(_REPO_ROOT, "cdk")
if _CDK_DIR not in sys.path:
    sys.path.insert(0, _CDK_DIR)

import aws_cdk as cdk  # noqa: E402
import aws_cdk.aws_lambda as lambda_  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402

from auth_resources import (  # noqa: E402
    ACCESS_TOKEN_VALIDITY_MINUTES,
    APP_CLIENT_NAME,
    ID_TOKEN_VALIDITY_MINUTES,
    PASSWORD_HISTORY_SIZE,
    PASSWORD_MIN_LENGTH,
    REFRESH_TOKEN_VALIDITY_HOURS,
    SCOPE_GROUPS,
    USER_POOL_NAME,
    create_auth_resources,
)
from gateway_resources import create_gateway  # noqa: E402
from policy_resources import create_policy_engine  # noqa: E402

# Auth flows that must never be enabled: both send a plaintext password to the
# Cognito API. ALLOW_REFRESH_TOKEN_AUTH is added by Cognito itself and is fine.
_FORBIDDEN_AUTH_FLOWS = frozenset(
    {"ALLOW_USER_PASSWORD_AUTH", "ALLOW_ADMIN_USER_PASSWORD_AUTH", "ALLOW_CUSTOM_AUTH"}
)


@pytest.fixture(scope="module")
def template() -> Template:
    """Synthesize a minimal stack carrying the managed auth layer + gateway."""
    app = cdk.App()
    stack = cdk.Stack(
        app,
        "TestAuthResourcesStack",
        env=cdk.Environment(account="123456789012", region="us-east-1"),
    )

    auth = create_auth_resources(stack)
    policy_engine = create_policy_engine(scope=stack)
    interceptor_fn = lambda_.Function(
        stack,
        "StubInterceptor",
        runtime=lambda_.Runtime.PYTHON_3_14,
        handler="index.handler",
        code=lambda_.Code.from_inline("def handler(event, context):\n    return {}\n"),
    )
    create_gateway(
        scope=stack,
        interceptor_fn=interceptor_fn,
        policy_engine=policy_engine,
        auth=auth,
    )
    return Template.from_stack(stack)


def _sole_resource(template: Template, cfn_type: str) -> dict:
    """Return the single resource of ``cfn_type``, asserting there is exactly one."""
    resources = template.find_resources(cfn_type)
    assert len(resources) == 1, f"expected exactly one {cfn_type}, got {len(resources)}"
    return next(iter(resources.values()))


# ---------------------------------------------------------------------------
# The pool is managed, not imported
# ---------------------------------------------------------------------------


class TestUserPoolIsManaged:
    """A managed pool must appear in the template (an imported one never does)."""

    def test_pool_present_and_named(self, template: Template) -> None:
        pool = _sole_resource(template, "AWS::Cognito::UserPool")
        assert pool["Properties"]["UserPoolName"] == USER_POOL_NAME

    def test_pool_is_destroyed_with_the_stack(self, template: Template) -> None:
        # The import approach left the pool behind after `cdk destroy`, which is
        # exactly the kind of drift this layer set out to remove. DESTROY makes
        # teardown complete.
        pool = _sole_resource(template, "AWS::Cognito::UserPool")
        assert pool.get("DeletionPolicy") == "Delete"
        assert pool.get("UpdateReplacePolicy") == "Delete"

    def test_deletion_protection_is_explicit(self, template: Template) -> None:
        pool = _sole_resource(template, "AWS::Cognito::UserPool")
        assert pool["Properties"].get("DeletionProtection") == "INACTIVE"


class TestUserPoolPosture:
    """Every setting the import approach hid is now pinned in the template."""

    def test_mfa_is_configured_with_totp(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPool")["Properties"]
        # OPTIONAL (not REQUIRED) is a documented demo trade-off — see the comment
        # on ``mfa`` in create_auth_resources. What matters for drift is that the
        # value is explicit and that the second factor is TOTP, never SMS.
        assert props["MfaConfiguration"] == "OPTIONAL"
        assert props["EnabledMfas"] == ["SOFTWARE_TOKEN_MFA"]

    def test_password_policy(self, template: Template) -> None:
        policy = _sole_resource(template, "AWS::Cognito::UserPool")["Properties"][
            "Policies"
        ]["PasswordPolicy"]
        assert policy["MinimumLength"] == PASSWORD_MIN_LENGTH
        assert PASSWORD_MIN_LENGTH >= 12, "password floor must stay above the default"
        assert policy["RequireLowercase"] is True
        assert policy["RequireUppercase"] is True
        assert policy["RequireNumbers"] is True
        assert policy["RequireSymbols"] is True
        assert policy["PasswordHistorySize"] == PASSWORD_HISTORY_SIZE

    def test_self_signup_disabled(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPool")["Properties"]
        assert props["AdminCreateUserConfig"]["AllowAdminCreateUserOnly"] is True

    def test_account_recovery_disabled(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPool")["Properties"]
        mechanisms = props.get("AccountRecoverySetting", {}).get(
            "RecoveryMechanisms", []
        )
        assert mechanisms == [{"Name": "admin_only", "Priority": 1}]

    def test_feature_plan_is_pinned(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPool")["Properties"]
        assert props["UserPoolTier"] == "ESSENTIALS"

    def test_username_only_signin(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPool")["Properties"]
        # No email/phone alias: nothing to alias-collide and no unverified
        # contact channel that could be used to take an account over.
        assert props.get("AliasAttributes") is None
        assert props.get("UsernameAttributes") is None


# ---------------------------------------------------------------------------
# The app client: SRP only
# ---------------------------------------------------------------------------


class TestAppClientAuthFlows:
    """The central finding: no plaintext-password auth flow is ever enabled."""

    def test_srp_only(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        flows = set(props["ExplicitAuthFlows"])
        assert "ALLOW_USER_SRP_AUTH" in flows
        assert not (flows & _FORBIDDEN_AUTH_FLOWS), (
            "app client must not enable a plaintext-password auth flow, got "
            f"{sorted(flows)}"
        )

    def test_client_name(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        assert props["ClientName"] == APP_CLIENT_NAME

    def test_no_client_secret(self, template: Template) -> None:
        # A public client cannot keep a secret; SRP is what makes that safe. The
        # CDK omits GenerateSecret entirely when generate_secret=False.
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        assert props.get("GenerateSecret") in (None, False)

    def test_client_references_the_managed_pool(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        assert "Ref" in props["UserPoolId"], "client must point at the managed pool"


class TestAppClientHardening:
    def test_prevent_user_existence_errors(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        assert props["PreventUserExistenceErrors"] == "ENABLED"

    def test_token_validity_is_short_and_explicit(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        assert props["AccessTokenValidity"] == ACCESS_TOKEN_VALIDITY_MINUTES
        assert props["IdTokenValidity"] == ID_TOKEN_VALIDITY_MINUTES
        assert props["RefreshTokenValidity"] == REFRESH_TOKEN_VALIDITY_HOURS * 60
        units = props["TokenValidityUnits"]
        assert units["AccessToken"] == "minutes"
        assert units["IdToken"] == "minutes"
        assert units["RefreshToken"] == "minutes"
        # Cognito's own default is 30 days; anything near that is a finding.
        assert REFRESH_TOKEN_VALIDITY_HOURS <= 24

    def test_token_revocation_enabled(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        assert props["EnableTokenRevocation"] is True

    def test_no_oauth_flows(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        # Leaving OAuth on would synthesize an implicit-grant client with a
        # placeholder callback URL.
        assert props.get("AllowedOAuthFlows") in (None, [])
        assert props.get("CallbackURLs") in (None, [])
        assert props.get("AllowedOAuthFlowsUserPoolClient") in (None, False)

    def test_cognito_is_the_only_idp(self, template: Template) -> None:
        props = _sole_resource(template, "AWS::Cognito::UserPoolClient")["Properties"]
        assert props["SupportedIdentityProviders"] == ["COGNITO"]


# ---------------------------------------------------------------------------
# Scope groups
# ---------------------------------------------------------------------------


class TestScopeGroups:
    """Group membership IS the served_scope grant, so the groups are in code."""

    def test_one_group_per_scope(self, template: Template) -> None:
        groups = template.find_resources("AWS::Cognito::UserPoolGroup")
        names = {g["Properties"]["GroupName"] for g in groups.values()}
        assert names == set(SCOPE_GROUPS)

    def test_groups_belong_to_the_managed_pool(self, template: Template) -> None:
        groups = template.find_resources("AWS::Cognito::UserPoolGroup")
        for group in groups.values():
            assert "Ref" in group["Properties"]["UserPoolId"]

    def test_no_foreign_partition_group(self, template: Template) -> None:
        # infra-secrets / hr-data are DynamoDB partitions only. A Cognito group
        # with either name would make an unreachable scope reachable.
        groups = template.find_resources("AWS::Cognito::UserPoolGroup")
        names = {g["Properties"]["GroupName"] for g in groups.values()}
        assert not (names & {"infra-secrets", "hr-data"})


# ---------------------------------------------------------------------------
# Deploy-time identifiers are published, not hardcoded
# ---------------------------------------------------------------------------


class TestOutputsAndReferences:
    """Consumers must resolve ids from the stack, never from a literal."""

    @pytest.mark.parametrize(
        "output_name",
        [
            "CognitoUserPoolId",
            "CognitoAppClientId",
            "CognitoDiscoveryUrl",
            "CognitoIssuer",
        ],
    )
    def test_output_present(self, template: Template, output_name: str) -> None:
        outputs = template.to_json().get("Outputs", {})
        assert output_name in outputs, f"missing stack output {output_name}"

    def test_gateway_authorizer_references_the_managed_client(
        self, template: Template
    ) -> None:
        gateways = template.find_resources("AWS::BedrockAgentCore::Gateway")
        assert len(gateways) == 1
        authorizer = next(iter(gateways.values()))["Properties"][
            "AuthorizerConfiguration"
        ]["CustomJWTAuthorizer"]
        # Both fields must be CloudFormation intrinsics over the managed
        # resources; a plain string here would mean a hardcoded id is back.
        assert not isinstance(authorizer["DiscoveryUrl"], str), (
            "discoveryUrl must be built from the managed pool id, not hardcoded"
        )
        assert authorizer["AllowedClients"] == [{"Ref": _client_logical_id(template)}]


def _client_logical_id(template: Template) -> str:
    """Return the logical id of the sole managed app client."""
    clients = template.find_resources("AWS::Cognito::UserPoolClient")
    assert len(clients) == 1
    return next(iter(clients))


class TestNoImportedPoolInSource:
    """Regression guard: the import approach must not creep back into cdk/."""

    def _cdk_sources(self) -> list[str]:
        return [
            os.path.join(_CDK_DIR, name)
            for name in sorted(os.listdir(_CDK_DIR))
            if name.endswith(".py")
        ]

    def test_no_from_user_pool_import_call(self) -> None:
        # Parsed with ast, not grepped: the module docstring legitimately
        # discusses the retired import approach by name, and only a real call
        # expression is a regression.
        forbidden = {
            "from_user_pool_id",
            "from_user_pool_client_id",
            "from_user_pool_arn",
        }
        offenders = []
        for path in self._cdk_sources():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in forbidden
                ):
                    offenders.append(
                        f"{os.path.basename(path)}:{node.lineno} "
                        f"({node.func.attr})"
                    )
        assert not offenders, (
            "the Cognito pool/client must be MANAGED, not imported by id — "
            f"found an import call at {offenders}"
        )
