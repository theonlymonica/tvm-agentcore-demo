#!/usr/bin/env python3
"""Mint a demo access token for the gateway, over SRP.

Why this script exists
----------------------
The managed Cognito app client allows ``ALLOW_USER_SRP_AUTH`` only:
``USER_PASSWORD_AUTH`` and ``ADMIN_USER_PASSWORD_AUTH`` are both off, so a
plaintext password never reaches the Cognito API (see ``cdk/auth_resources.py``).
The AWS CLI cannot perform SRP — it does not compute ``SRP_A`` — so
``aws cognito-idp initiate-auth`` has no way to authenticate against this client
and there is no CLI path to a token. This script is that path.

Design notes
------------
**Ids come from the stack, never from a literal.** The pool and app client are
CloudFormation-managed, so their ids are deploy-time values that change whenever
the pool is replaced. Hardcoding them is exactly the mistake that made the
identifiers in this repository's git history a problem in the first place, so the
ids are read from the stack outputs (``CognitoUserPoolId`` /
``CognitoAppClientId``) on every run.

**The password is never printed and never passed on a command line.** It is read
from an SSM ``SecureString`` and handed straight to the SRP exchange, then
dropped. The token itself IS printed — on stdout, alone, so the script composes:

    TOKEN="$(python3 scripts/mint_demo_token.py)"

Everything else (progress, claims, warnings) goes to stderr so it cannot
contaminate that capture.

**What this does NOT do.** It does not create the demo user; CloudFormation
cannot set a Cognito password, so the user is created post-deploy by an admin
(see ``DEMO_USER_GROUP`` in ``cdk/auth_resources.py``). It also does not verify
the token's signature — the gateway's ``CUSTOM_JWT`` authorizer and the
interceptor do that. ``--check`` only inspects claims for the misconfigurations
that would otherwise surface as an opaque gateway error.

Usage
-----
    python3 scripts/mint_demo_token.py                      # token on stdout
    python3 scripts/mint_demo_token.py --claims             # + claims on stderr
    python3 scripts/mint_demo_token.py --check              # assert invariants
    python3 scripts/mint_demo_token.py --username other-user

Requires ``pycognito`` (pinned in ``requirements-dev.txt``) and credentials for
the account the stack is deployed in.

Citation:
  - Cognito auth-flow selection (why the CLI cannot do SRP):
    https://docs.aws.amazon.com/cognito/latest/developerguide/authentication-flows-selection-sdk.html
  - Access-token claims (``token_use``, ``client_id``, ``cognito:groups``):
    https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-the-access-token.html
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
from typing import Any

#: Stack whose outputs carry the managed pool / client ids.
DEFAULT_STACK = "ToxicFlowStack"

#: SSM SecureString holding the demo user's password.
DEFAULT_PASSWORD_PARAM = "/toxic-flow/manual/DEMO_PASSWORD"

#: The demo user created post-deploy (see cdk/auth_resources.py DEMO_USER_GROUP).
DEFAULT_USERNAME = "demo-user"

#: Stack output keys produced by cdk/auth_resources.py.
POOL_OUTPUT = "CognitoUserPoolId"
CLIENT_OUTPUT = "CognitoAppClientId"

#: The scope groups the interceptor recognises. MUST stay in sync with
#: ``cdk/auth_resources.py`` SCOPE_GROUPS and
#: ``interceptor/jwt_claims._DEFAULT_KNOWN_SCOPES``;
#: ``tests/test_known_scopes.py`` pins those two to each other.
KNOWN_SCOPE_GROUPS = frozenset({"payments-core", "billing-internal"})


def decode_claims(token: str) -> dict[str, Any]:
    """Base64url-decode a JWT's payload segment.

    NOT a verification: the signature is checked by the gateway authorizer and
    again by the interceptor. This exists so ``--check`` can report a
    misconfiguration locally instead of leaving the caller to interpret an
    opaque gateway error.

    Args:
        token: The compact-serialization JWT.

    Returns:
        The decoded payload claims.

    Raises:
        ValueError: If the token is not three dot-separated segments, or the
            payload is not valid base64url-encoded JSON.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"not a compact JWT: expected 3 dot-separated segments, got {len(parts)}"
        )
    payload = parts[1]
    try:
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"payload is not valid base64url: {exc}") from exc
    try:
        claims = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(claims, dict):
        raise ValueError(f"payload is not a JSON object, got {type(claims).__name__}")
    return claims


def check_claims(
    claims: dict[str, Any],
    *,
    expected_client_id: str,
    known_scopes: frozenset[str] = KNOWN_SCOPE_GROUPS,
) -> list[str]:
    """Report the claim problems that would make the gateway reject this token.

    Mirrors the invariants enforced downstream, so a bad token is diagnosed here
    rather than as a generic failure later:

    * ``token_use`` must be ``access`` — the interceptor rejects an id token;
    * ``client_id`` must be the deployed app client — the gateway's
      ``allowedClients`` and the interceptor's pinning both check it;
    * exactly ONE known scope group must be present — the single-scope-group
      invariant. Zero or more than one fails closed with no ``served_scope``.

    Args:
        claims: Decoded access-token claims.
        expected_client_id: The app client id resolved from the stack outputs.
        known_scopes: The recognised scope-group names.

    Returns:
        Human-readable problems, empty when the token satisfies every invariant.
    """
    problems: list[str] = []

    token_use = claims.get("token_use")
    if token_use != "access":
        problems.append(
            f"token_use is {token_use!r}, expected 'access' — the interceptor "
            "rejects anything else"
        )

    client_id = claims.get("client_id")
    if client_id != expected_client_id:
        problems.append(
            f"client_id is {client_id!r}, expected {expected_client_id!r} — the "
            "gateway's allowedClients would reject this token"
        )

    groups = claims.get("cognito:groups")
    if not isinstance(groups, list):
        problems.append(
            f"cognito:groups is {groups!r}, expected a list — no served_scope "
            "can be derived, so the request fails closed"
        )
    else:
        matched = sorted({g for g in groups if g in known_scopes})
        if len(matched) != 1:
            problems.append(
                f"expected exactly ONE known scope group, found {matched!r} "
                f"(groups={groups!r}, known={sorted(known_scopes)!r}) — the "
                "single-scope-group invariant fails closed"
            )

    return problems


def _resolve_ids(stack_name: str, region: str | None, profile: str | None) -> tuple[str, str]:
    """Read the managed pool / client ids from the stack outputs.

    Args:
        stack_name: The CloudFormation stack to read.
        region: AWS region, or None for the session default.
        profile: AWS profile name, or None for the session default.

    Returns:
        ``(user_pool_id, app_client_id)``.

    Raises:
        SystemExit: If the stack or either output is missing.
    """
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region)
    cfn = session.client("cloudformation")
    try:
        stacks = cfn.describe_stacks(StackName=stack_name)["Stacks"]
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        raise SystemExit(f"cannot read stack {stack_name!r}: {exc}") from exc

    outputs = {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}
    missing = [k for k in (POOL_OUTPUT, CLIENT_OUTPUT) if k not in outputs]
    if missing:
        raise SystemExit(
            f"stack {stack_name!r} has no {missing!r} output(s) — is the managed "
            "Cognito change deployed?"
        )
    return outputs[POOL_OUTPUT], outputs[CLIENT_OUTPUT]


def _read_password(param: str, region: str | None, profile: str | None) -> str:
    """Fetch the demo password from SSM. The value is never logged.

    Args:
        param: SSM parameter name (a SecureString).
        region: AWS region, or None for the session default.
        profile: AWS profile name, or None for the session default.

    Returns:
        The decrypted password.

    Raises:
        SystemExit: If the parameter cannot be read.
    """
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region)
    try:
        got = session.client("ssm").get_parameter(Name=param, WithDecryption=True)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        raise SystemExit(f"cannot read SSM parameter {param!r}: {exc}") from exc
    return got["Parameter"]["Value"]


def main(argv: list[str] | None = None) -> int:
    """Mint a token and write it to stdout.

    Args:
        argv: Argument vector, or None to use ``sys.argv``.

    Returns:
        Process exit status (0 on success, 1 when ``--check`` finds problems).
    """
    parser = argparse.ArgumentParser(
        description="Mint a demo access token for the gateway over SRP.",
        epilog='usage: TOKEN="$(python3 scripts/mint_demo_token.py)"',
    )
    parser.add_argument("--stack", default=DEFAULT_STACK)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password-param", default=DEFAULT_PASSWORD_PARAM)
    parser.add_argument("--profile", default=None, help="AWS profile name")
    parser.add_argument("--region", default=None, help="AWS region")
    parser.add_argument(
        "--claims", action="store_true", help="print decoded claims to stderr"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the claims the gateway and interceptor require; exit 1 on any problem",
    )
    args = parser.parse_args(argv)

    try:
        from pycognito import Cognito
    except ModuleNotFoundError:
        raise SystemExit(
            "pycognito is not installed. The app client is SRP-only and the AWS "
            "CLI cannot compute SRP_A, so this library is required:\n"
            "    pip install -r requirements-dev.txt"
        ) from None

    pool_id, client_id = _resolve_ids(args.stack, args.region, args.profile)
    print(
        f"resolved from {args.stack}: pool={pool_id} client={client_id}",
        file=sys.stderr,
    )

    password = _read_password(args.password_param, args.region, args.profile)
    user = Cognito(pool_id, client_id, username=args.username)
    try:
        user.authenticate(password=password)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        raise SystemExit(
            f"SRP authentication failed for {args.username!r}: {exc}\n"
            "Does the user exist in this pool, with a permanent password set?"
        ) from exc
    finally:
        del password

    token = user.access_token

    if args.claims or args.check:
        claims = decode_claims(token)
        if args.claims:
            interesting = {
                k: claims.get(k)
                for k in ("token_use", "client_id", "cognito:groups", "iss", "exp")
                if k in claims
            }
            print(json.dumps(interesting, indent=2, sort_keys=True), file=sys.stderr)
        if args.check:
            problems = check_claims(claims, expected_client_id=client_id)
            if problems:
                print("token FAILED the claim checks:", file=sys.stderr)
                for p in problems:
                    print(f"  - {p}", file=sys.stderr)
                return 1
            print("claim checks passed", file=sys.stderr)

    # stdout carries ONLY the token, so the caller can capture it.
    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
