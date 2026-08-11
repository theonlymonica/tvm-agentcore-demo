"""Managed Cognito identity resources for the scope claim.

This module is the CDK home for the Cognito identity layer the gateway
``CUSTOM_JWT`` authorizer and the interceptor's in-Lambda JWT verification
consume. Everything here is **MANAGED** by this stack: the user pool, the app
client, and the scope groups are created, updated and destroyed by
CloudFormation, so the security floor of the identity layer is expressed in code
and asserted by ``tests/test_auth_resources.py``.

Claim mechanism:
    ``served_scope`` is carried by ONE Cognito group per scope, emitted in the
    STANDARD ``cognito:groups`` claim of the user's access token. There is NO
    pre-token-generation trigger and NO custom attribute — neither is needed to
    carry the scope. The interceptor (``interceptor/jwt_claims.py``) maps a
    group to ``served_scope`` via the single-scope-group invariant (exactly one
    known scope group -> that scope; zero or more-than-one -> fail closed).

Managed rather than imported:
    The pool and the app client are managed constructs, not resources imported
    by literal id. That keeps the whole auth posture — MFA, password policy,
    token validity, threat-protection tier, ``PreventUserExistenceErrors``,
    self-signup — defined in this file, visible to CloudFormation and to the
    test suite, and removed again by ``cdk destroy``. Two consequences, accepted
    deliberately:

    - The pool id / app client id are DEPLOY-TIME values (CloudFormation
      tokens), NOT literals. Every consumer takes them from the
      :class:`AuthResources` returned by :func:`create_auth_resources` — see
      ``cdk/gateway_resources.py``, ``cdk/gateway_wiring.py`` and
      ``cdk/scoped_credentials_stack.py``. The ids are also published as stack outputs
      (``CognitoUserPoolId``, ``CognitoAppClientId``, ``CognitoDiscoveryUrl``,
      ``CognitoIssuer``) so operators and probe scripts can resolve them with
      ``aws cloudformation describe-stacks`` instead of hardcoding them.
    - Users are NOT managed here. The demo user is created post-deploy by an
      admin (see :data:`DEMO_USER_GROUP` and the commands recorded there),
      because CloudFormation cannot set a Cognito password.

Operational consequence of SRP-only (read before running any probe script):
    ``aws cognito-idp initiate-auth --auth-flow USER_PASSWORD_AUTH`` does not
    work against this client, and neither does ``admin-initiate-auth``:
    ``ALLOW_USER_PASSWORD_AUTH`` and ``ALLOW_ADMIN_USER_PASSWORD_AUTH`` are both
    off, and the AWS CLI cannot perform SRP (it does not compute ``SRP_A``). To
    mint a demo access token you therefore need an SRP-capable client — e.g. the
    ``pycognito`` library, or the AWS SDK's SRP helper — pointed at the
    ``CognitoUserPoolId`` / ``CognitoAppClientId`` stack outputs.

    If a scripted, CLI-only token path is required for the demo, the smallest
    safe relaxation is ``admin_user_password=True`` on the client below:
    ``ADMIN_USER_PASSWORD_AUTH`` still sends the password to Cognito, but only
    over an IAM-authenticated admin API call, so it is not reachable by an
    anonymous attacker the way the public ``USER_PASSWORD_AUTH`` flow is. That
    is a deliberate posture change: flip the flag, update the forbidden-flow set
    in ``tests/test_auth_resources.py``, and say so in the commit.

Auth posture expressed in code:
    - ``ALLOW_USER_SRP_AUTH`` only — ``USER_PASSWORD_AUTH`` and
      ``ADMIN_USER_PASSWORD_AUTH`` are OFF, so a plaintext password is never
      sent to the Cognito API.
    - OAuth/hosted-UI flows disabled (no implicit grant, no callback URLs).
    - ``PreventUserExistenceErrors=ENABLED`` (no username enumeration).
    - Short access/ID token validity, refresh-token validity bounded, token
      revocation enabled.
    - Password policy: length + all four character classes + password history.
    - Self-signup disabled, account recovery disabled (admin-created users only).
    - MFA OPTIONAL with TOTP — see the comment on ``mfa`` in
      :func:`create_auth_resources` for why this is not REQUIRED.

AWS documentation references:
    - ``aws_cdk.aws_cognito.UserPool`` (mfa, password_policy, feature_plan,
      self_sign_up_enabled, account_recovery, removal_policy):
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_cognito/UserPool.html
    - ``aws_cdk.aws_cognito.UserPoolClient`` (auth_flows,
      prevent_user_existence_errors, token validity, disable_o_auth):
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_cognito/UserPoolClient.html
    - Cognito authentication flows (SRP vs ``USER_PASSWORD_AUTH``):
      https://docs.aws.amazon.com/cognito/latest/developerguide/authentication-flows-selection-sdk.html
    - Cognito user pool password policy / threat protection (feature plans):
      https://docs.aws.amazon.com/cognito/latest/developerguide/managing-users.html
    - Cognito ``cognito:groups`` claim carried in access/ID tokens:
      https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-the-access-token.html
    - Verifying a JWT (issuer + JWKS URI shapes consumed by the interceptor):
      https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
    - OIDC discovery document URL shape
      (``.../.well-known/openid-configuration``) consumed by the gateway
      CUSTOM_JWT authorizer:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-create-api.html

Functions:
    create_auth_resources: Create the managed Cognito pool + app client +
        scope groups and return an AuthResources reference object.
"""

from __future__ import annotations

from dataclasses import dataclass

import aws_cdk as cdk
import aws_cdk.aws_cognito as cognito
from constructs import Construct

# ---------------------------------------------------------------------------
# Managed-resource naming and posture constants.
#
# These are the knobs the security posture of the identity layer is made of.
# They live at module level so ``tests/test_auth_resources.py`` can assert the
# synthesized template against them (drift becomes a test failure, not a
# surprise in the console).
# ---------------------------------------------------------------------------

#: Name of the managed user pool (the id is assigned by Cognito at deploy time).
USER_POOL_NAME = "scoped-credentials-pool"

#: Name of the managed app client (the id is assigned at deploy time).
APP_CLIENT_NAME = "scoped-credentials-client"

#: Scope groups created in the pool. ONE Cognito group per scope, carried in the
#: standard ``cognito:groups`` claim. This is the "known scope set" the
#: interceptor intersects ``cognito:groups`` against under the
#: single-scope-group invariant. ``tests/test_known_scopes.py`` enforces that
#: ``interceptor/jwt_claims._DEFAULT_KNOWN_SCOPES`` equals this tuple.
SCOPE_GROUPS = ("payments-core", "billing-internal")

#: The group a demo/operator user must belong to (exactly one scope group) for
#: the read path to resolve a ``served_scope``. Users are NOT CDK-managed:
#: CloudFormation cannot set a Cognito password, so create the demo user
#: post-deploy against the deployed pool id::
#:
#:     aws cognito-idp admin-create-user --user-pool-id "$POOL_ID" \
#:         --username demo-user --message-action SUPPRESS
#:     aws cognito-idp admin-set-user-password --user-pool-id "$POOL_ID" \
#:         --username demo-user --password "$PW" --permanent
#:     aws cognito-idp admin-add-user-to-group --user-pool-id "$POOL_ID" \
#:         --username demo-user --group-name payments-core
DEMO_USER_GROUP = "payments-core"

#: Minimum password length enforced by the pool. Above the Cognito default (8)
#: because the client is a public client and the password is the only factor
#: when MFA is not registered.
PASSWORD_MIN_LENGTH = 16

#: How many previous passwords the pool refuses to reuse.
PASSWORD_HISTORY_SIZE = 3

#: Access-token lifetime. Short because the access token is the bearer credential
#: the gateway accepts; 15 minutes is the practical floor for a CLI-driven demo
#: (Cognito's own minimum is 5 minutes).
ACCESS_TOKEN_VALIDITY_MINUTES = 15

#: ID-token lifetime (kept equal to the access token).
ID_TOKEN_VALIDITY_MINUTES = 15

#: Refresh-token lifetime. Bounded to a working day instead of the Cognito
#: default of 30 days.
REFRESH_TOKEN_VALIDITY_HOURS = 8


@dataclass
class AuthResources:
    """Container for the managed Cognito identity resource references.

    Mirrors the ``DataResources`` pattern in ``cdk/data_resources.py``. The
    string fields are CDK **tokens** resolved at deploy time, not literals — they
    are safe to interpolate into CloudFormation properties, Lambda environment
    variables and ``AwsCustomResource`` payloads, and they must never be
    compared against a hardcoded id.

    Attributes:
        user_pool: The managed user pool.
        app_client: The managed app client.
        discovery_url: OIDC discovery URL for the gateway CUSTOM_JWT authorizer.
        app_client_id: App client id for the authorizer's ``allowed_clients``.
        issuer: Token issuer (``iss``) the interceptor verifies against
            (``COGNITO_ISSUER``).
        jwks_url: JWKS URI the interceptor fetches RS256 signing keys from
            (``COGNITO_JWKS_URL``).
        scope_groups: The known scope-group names (candidate (a)); the
            interceptor's known-scope set for the single-scope-group invariant.
    """

    user_pool: cognito.IUserPool
    app_client: cognito.IUserPoolClient
    discovery_url: str
    app_client_id: str
    issuer: str
    jwks_url: str
    scope_groups: tuple[str, ...]


def create_auth_resources(scope: Construct) -> AuthResources:
    """Create the managed Cognito identity resources for this stack.

    Provisions a user pool with an explicit security posture, a public app
    client restricted to SRP, and one Cognito group per scope. Publishes the
    deploy-time identifiers as stack outputs so nothing downstream needs a
    hardcoded id.

    Args:
        scope: The CDK Stack or Construct to attach the resources to.

    Returns:
        AuthResources with the managed pool/client plus the discovery URL,
        issuer, JWKS URL and app client id its consumers need.
    """
    stack = cdk.Stack.of(scope)

    user_pool = cognito.UserPool(
        scope,
        "UserPool",
        user_pool_name=USER_POOL_NAME,
        # Admin-created users only. Self-signup on a pool whose group membership
        # IS the tenancy decision would let anyone mint an identity.
        self_sign_up_enabled=False,
        # Username-only sign-in: no email/phone alias, so there is no
        # alias-collision surface and no verified-contact channel to abuse.
        sign_in_aliases=cognito.SignInAliases(username=True),
        sign_in_case_sensitive=True,
        # No self-service recovery: an operator resets via admin APIs. Recovery
        # over an unverified channel would be a scope-granting bypass.
        account_recovery=cognito.AccountRecovery.NONE,
        # MFA is OPTIONAL, not REQUIRED, and this is a deliberate demo trade-off:
        # every probe/live-check path mints tokens NON-INTERACTIVELY
        # (`initiate-auth`), and MFA REQUIRED makes every such call return a
        # SOFTWARE_TOKEN_MFA challenge that no script here can answer. OPTIONAL
        # keeps TOTP available and — unlike the imported pool — makes the setting
        # explicit, code-reviewed and drift-tested. Production posture is
        # ``cognito.Mfa.REQUIRED``; flip this line and register TOTP for every
        # user before doing so.
        mfa=cognito.Mfa.OPTIONAL,
        # TOTP only. SMS as a second factor needs an SMS-publishing IAM role and
        # is phishable/SIM-swappable.
        mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
        password_policy=cognito.PasswordPolicy(
            min_length=PASSWORD_MIN_LENGTH,
            require_lowercase=True,
            require_uppercase=True,
            require_digits=True,
            require_symbols=True,
            password_history_size=PASSWORD_HISTORY_SIZE,
            temp_password_validity=cdk.Duration.days(1),
        ),
        # Feature plan pinned explicitly so the tier is auditable from code.
        # ESSENTIALS is the paid-tier floor that this demo needs; Cognito's
        # threat protection (adaptive auth / compromised-credential detection)
        # requires the PLUS plan — set ``feature_plan=cognito.FeaturePlan.PLUS``
        # together with ``standard_threat_protection_mode=FULL_FUNCTION`` if the
        # additional per-MAU cost is acceptable.
        feature_plan=cognito.FeaturePlan.ESSENTIALS,
        # Deleting the stack must delete the pool: an identity provider left
        # behind after the stack is torn down is unmanaged drift. Nothing of
        # record lives in this pool (documents are in DynamoDB); users are
        # admin-created demo users.
        deletion_protection=False,
        removal_policy=cdk.RemovalPolicy.DESTROY,
    )

    app_client = user_pool.add_client(
        "AppClient",
        user_pool_client_name=APP_CLIENT_NAME,
        # SRP ONLY. `user_password` (ALLOW_USER_PASSWORD_AUTH) and
        # `admin_user_password` are deliberately OFF: both send the plaintext
        # password to the Cognito API. `custom` is off (no challenge Lambdas).
        auth_flows=cognito.AuthFlow(
            user_srp=True,
            user_password=False,
            admin_user_password=False,
            custom=False,
        ),
        # Public client (no secret): the demo authenticates from a CLI, which
        # cannot keep a client secret. SRP is what makes that safe.
        generate_secret=False,
        # No hosted UI / OAuth: leaving OAuth enabled would synthesize an
        # implicit-grant client with a placeholder callback URL.
        disable_o_auth=True,
        supported_identity_providers=[
            cognito.UserPoolClientIdentityProvider.COGNITO,
        ],
        # No username enumeration from failed auth attempts.
        prevent_user_existence_errors=True,
        # Allow refresh tokens to be revoked (logout / incident response).
        enable_token_revocation=True,
        access_token_validity=cdk.Duration.minutes(ACCESS_TOKEN_VALIDITY_MINUTES),
        id_token_validity=cdk.Duration.minutes(ID_TOKEN_VALIDITY_MINUTES),
        refresh_token_validity=cdk.Duration.hours(REFRESH_TOKEN_VALIDITY_HOURS),
    )

    # One Cognito group per scope. Group membership is admin-controlled and is
    # the ONLY way a user obtains a served_scope, so these are
    # security-relevant resources and belong in the template.
    for index, group_name in enumerate(SCOPE_GROUPS):
        cognito.CfnUserPoolGroup(
            scope,
            f"ScopeGroup{_construct_suffix(group_name)}",
            user_pool_id=user_pool.user_pool_id,
            group_name=group_name,
            description=f"served_scope={group_name} (single-scope-group invariant)",
            precedence=index,
        )

    issuer = (
        f"https://cognito-idp.{stack.region}.amazonaws.com/{user_pool.user_pool_id}"
    )
    discovery_url = f"{issuer}/.well-known/openid-configuration"
    jwks_url = f"{issuer}/.well-known/jwks.json"

    # Publish the deploy-time identifiers so operators and scripts resolve them
    # from the stack instead of hardcoding them: the pool id changes whenever the
    # pool is replaced, and a hardcoded copy silently points at a dead pool.
    cdk.CfnOutput(
        scope,
        "CognitoUserPoolId",
        value=user_pool.user_pool_id,
        description="Managed Cognito user pool id (gateway CUSTOM_JWT issuer)",
    )
    cdk.CfnOutput(
        scope,
        "CognitoAppClientId",
        value=app_client.user_pool_client_id,
        description="Managed Cognito app client id (SRP-only, public client)",
    )
    cdk.CfnOutput(
        scope,
        "CognitoDiscoveryUrl",
        value=discovery_url,
        description="OIDC discovery URL consumed by the gateway CUSTOM_JWT authorizer",
    )
    cdk.CfnOutput(
        scope,
        "CognitoIssuer",
        value=issuer,
        description="Token issuer the interceptor verifies iss against",
    )

    return AuthResources(
        user_pool=user_pool,
        app_client=app_client,
        discovery_url=discovery_url,
        app_client_id=app_client.user_pool_client_id,
        issuer=issuer,
        jwks_url=jwks_url,
        scope_groups=SCOPE_GROUPS,
    )


def _construct_suffix(group_name: str) -> str:
    """Turn a scope-group name into a CDK-safe construct-id suffix.

    Construct ids must be alphanumeric-ish and stable: the logical id derived
    from them is what keeps a group from being replaced on every deploy.

    Args:
        group_name: A scope-group name such as ``payments-core``.

    Returns:
        The name with separators removed and each word capitalized, e.g.
        ``PaymentsCore``.
    """
    return "".join(part.capitalize() for part in group_name.replace("_", "-").split("-"))
