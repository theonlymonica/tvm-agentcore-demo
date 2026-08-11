"""
JWT claim extraction for the scope-injecting REQUEST interceptor.

This module derives the authoritative ``served_scope`` from the inbound user's
Bearer token using the claim mechanism: the
standard Cognito ``cognito:groups`` claim, with the single-scope-group
invariant.

Trust root — signature VERIFIED in the interceptor (defense in depth):
    The token is validated by the AgentCore Gateway ``CUSTOM_JWT`` inbound
    authorizer (signature, allowed clients) before the interceptor runs, but the
    interceptor no longer trusts that upstream check alone. It verifies the token
    ITSELF against the Cognito user pool JWKS — RS256 signature, issuer (``iss``),
    and expiry (``exp``) — using ``PyJWT`` (``jwt.PyJWKClient``) BEFORE reading any
    claim. Any verification failure fails closed (returns ``None``), exactly as a
    missing claim does.

Self-sufficient trust boundary — token TYPE and CLIENT pinned here too:
    Signature + issuer + expiry do not establish that a token is the ONE kind of
    token this interceptor is contracted to accept, so ``_claims_acceptable``
    asserts two more claims on the verified claim set: ``token_use == "access"``
    (the ID token shares this pool's JWKS, issuer and ``cognito:groups``, so
    nothing else distinguishes the two types) and ``client_id`` ∈
    ``COGNITO_ALLOWED_CLIENT_IDS`` (the claim the gateway validates via
    ``allowedClients``, re-checked here so this boundary does not depend on
    upstream config it cannot see). Both fail closed, as does an unset allow-list.

JWT verification dependency (container-image Lambda):
    Signature verification pulls in ``cryptography`` (via ``PyJWT[crypto]``), which
    has native binaries. The interceptor is therefore packaged as a container-image
    Lambda built on the AWS Lambda Python base image (see ``interceptor/Dockerfile``
    and ``cdk/scoped_credentials_stack.py``), so the native dependency matches the runtime +
    architecture. The JWKS URL and issuer are supplied via the ``COGNITO_JWKS_URL``
    / ``COGNITO_ISSUER`` environment variables; the JWKS is fetched once per
    execution environment and cached by ``PyJWKClient`` (not fetched on every call).

Known-scope set (group -> scope mapping) — documented choice:
    The single-scope-group invariant intersects ``cognito:groups`` with a
    "known scope set" and returns the scope IFF EXACTLY ONE matches. That set
    is the set of Cognito GROUPS that exist in the pool — i.e. the scopes a
    user can actually be granted via group membership — which is exactly
    ``cdk/auth_resources.SCOPE_GROUPS`` = ``("payments-core",
    "billing-internal")``. Rationale: a user can only be a member of groups
    that EXIST, so only those group names can ever appear in ``cognito:groups``.
    The seed data has four scope PARTITIONS (``payments-core``,
    ``billing-internal``, ``infra-secrets``, ``hr-data``), but ``infra-secrets``
    and ``hr-data`` exist only as DynamoDB partitions, NOT as Cognito groups —
    no token can carry them in ``cognito:groups`` — so they are intentionally
    excluded from the group->scope known set. The interceptor Lambda is bundled
    from ``interceptor/`` and does NOT import ``cdk/``, so the set is redefined
    here as a module constant (``KNOWN_SCOPES``) that MUST stay in sync with
    ``SCOPE_GROUPS``. An optional ``KNOWN_SCOPE_GROUPS`` environment
    variable (comma-separated) overrides the constant without a code change;
    it defaults to the constant so the current no-env interceptor deployment
    works unchanged.

Security:
    This module NEVER logs the Authorization header or the JWT.

Functions:
    served_scope_from_authorization: Verify the Bearer token and derive
        served_scope from the validated ``cognito:groups`` claim, or None (fail
        closed).

AWS documentation references:
    - Cognito JWT verification — JWKS URI
      ``https://cognito-idp.<Region>.amazonaws.com/<userPoolId>/.well-known/jwks.json``,
      issuer ``https://cognito-idp.<Region>.amazonaws.com/<userPoolId>``, RS256:
      https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
    - Cognito access-token ``cognito:groups`` claim ("An array of the names of
      user pool groups that have your user as a member"):
      https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-the-access-token.html
    - Cognito user pool groups & the ``cognito:groups`` claim (precedence
      resolves to ``cognito:preferred_role``, NOT to claim array order — so
      selection MUST NOT be by array position):
      https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-user-groups.html
    - Cognito ACCESS token — ``token_use``: "In an access token, its value is
      ``access``" (the ID token's counterpart value is documented on the
      ID-token page):
      https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-the-access-token.html
    - AgentCore inbound JWT authorizer — allowed clients are validated against the
      ``client_id`` claim, so the local check mirrors the gateway's:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/inbound-jwt-authorizer.html
"""

from __future__ import annotations

import os
from typing import Any, Optional

import jwt
from jwt import PyJWKClient

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: The Cognito claim that carries the user's group memberships.
_COGNITO_GROUPS_CLAIM = "cognito:groups"

#: The ONLY JWS algorithm accepted. Cognito signs with RS256; pinning the list
#: means the token header's ``alg`` never chooses the primitive, which is what
#: blocks HS256 confusion (HMAC using the published JWKS public key as the
#: "secret") and ``alg: none``. Kept as a named constant because a behavioural
#: test cannot cover this decision on its own — see
#: ``tests/test_interceptor_jwt_algorithm_pinning.py``.
_ALGORITHMS: list[str] = ["RS256"]

#: Cognito's token-type claim and the ONE value this interceptor accepts. The ID
#: token carries ``token_use: id`` and is signed by the same JWKS with the same
#: issuer, so this is the only claim that distinguishes the two token types.
_TOKEN_USE_CLAIM = "token_use"
_REQUIRED_TOKEN_USE = "access"

#: The access token's client-identity claim (``aud`` is null on a Cognito access
#: token, so ``client_id`` — not ``aud`` — carries the app client). This is the
#: same claim the gateway's ``allowedClients`` validates.
_CLIENT_ID_CLAIM = "client_id"

#: Env var (comma-separated) listing the app client ids whose access tokens this
#: interceptor accepts. Wired in ``cdk/scoped_credentials_stack.py`` from the managed app
#: client. Unset/empty fails CLOSED — see ``_allowed_client_ids``.
_ALLOWED_CLIENT_IDS_ENV = "COGNITO_ALLOWED_CLIENT_IDS"

#: Cognito JWKS URL + issuer for signature verification, from the environment
#: (set by ``cdk/scoped_credentials_stack.py`` from ``cdk/auth_resources.py``). Empty
#: when unset; verification then fails closed (returns None).
_JWKS_URL = os.environ.get("COGNITO_JWKS_URL", "").strip()
_ISSUER = os.environ.get("COGNITO_ISSUER", "").strip()

#: PyJWKClient caches the pool's signing keys in the execution environment. It
#: is constructed lazily on first use and fetches the JWKS on first key lookup
#: (then reuses cached keys — not fetched on every call).
_jwk_client: Optional[PyJWKClient] = None

#: Known scope-group set for the single-scope-group invariant. MUST stay in
#: sync with ``cdk/auth_resources.SCOPE_GROUPS`` (a drift test enforces
#: this — see ``tests/test_known_scopes.py``). These are the Cognito GROUPS the
#: stack provisions in the managed pool (the scopes a user can be granted via
#: group membership).
#:
#: SECURITY INVARIANT (explicit boundary, not an accident): a scope becomes
#: reachable as ``served_scope`` ONLY if it is BOTH (a) a Cognito group in this
#: set AND (b) a DynamoDB table partition. The seed has four partitions
#: (``payments-core``, ``billing-internal``, ``infra-secrets``, ``hr-data``) but
#: only the first two are Cognito groups, so ``infra-secrets`` / ``hr-data`` can
#: never appear in a token's ``cognito:groups`` and can never become
#: ``served_scope``. That is a deliberate boundary. WARNING: creating a Cognito
#: group named after a foreign partition (e.g. an ``infra-secrets`` group) would
#: SILENTLY make that partition reachable — do not add foreign-partition names
#: here or as Cognito groups. ``tests/test_known_scopes.py`` asserts
#: ``infra-secrets`` / ``hr-data`` fail closed so the boundary is enforced by a
#: test, not by the current group configuration.
_DEFAULT_KNOWN_SCOPES: frozenset[str] = frozenset(
    {"payments-core", "billing-internal"}
)

#: Optional env var (comma-separated) to override the known-scope set without a
#: code change. Defaults to ``_DEFAULT_KNOWN_SCOPES`` when unset/empty so the
#: current no-data-plane-env interceptor deployment works unchanged.
_KNOWN_SCOPES_ENV = "KNOWN_SCOPE_GROUPS"


def _known_scopes() -> frozenset[str]:
    """Return the known scope-group set (env override or module default).

    Reads the optional ``KNOWN_SCOPE_GROUPS`` environment variable as a
    comma-separated list; falls back to ``_DEFAULT_KNOWN_SCOPES`` when the
    variable is unset or contains no non-empty entries.

    Returns:
        The frozenset of known scope-group names to intersect ``cognito:groups``
        against under the single-scope-group invariant.
    """
    raw = os.environ.get(_KNOWN_SCOPES_ENV, "")
    parsed = {part.strip() for part in raw.split(",") if part.strip()}
    return frozenset(parsed) if parsed else _DEFAULT_KNOWN_SCOPES


def _allowed_client_ids() -> frozenset[str]:
    """Return the app client ids whose access tokens are accepted.

    Reads ``COGNITO_ALLOWED_CLIENT_IDS`` as a comma-separated list. Unlike
    ``_known_scopes`` there is deliberately NO code default: an empty result means
    the caller fails closed rather than accepting any client, matching the
    unset-``COGNITO_JWKS_URL`` / unset-``COGNITO_ISSUER`` posture.

    Returns:
        The frozenset of allowed ``client_id`` values; empty when the variable is
        unset or holds no non-empty entries.
    """
    raw = os.environ.get(_ALLOWED_CLIENT_IDS_ENV, "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def served_scope_from_authorization(authorization: Optional[str]) -> Optional[str]:
    """Verify the Bearer token and derive ``served_scope`` from its claims.

    Extracts the Bearer token, VERIFIES it against the Cognito user pool JWKS
    (RS256 signature, issuer, expiry), reads the standard ``cognito:groups``
    claim, and applies the single-scope-group invariant to select exactly one
    served scope.

    Args:
        authorization: The raw ``Authorization`` header value (expected form
            ``"Bearer <jwt>"``). May be None or malformed.

    Returns:
        The single matching served-scope string, or None to fail closed when the
        header is missing/malformed, the token fails signature/issuer/expiry
        verification, the claim is absent, or the single-scope-group invariant is
        not satisfied.

    Security:
        Never logs the header or the token.
    """
    if not authorization or not isinstance(authorization, str):
        return None
    if not authorization.strip().lower().startswith("bearer "):
        return None

    token = authorization.strip().split(" ", 1)[1].strip()
    if not token:
        return None

    claims = _verify_token(token)
    if claims is None:
        return None

    return _scope_from_cognito_groups(claims.get(_COGNITO_GROUPS_CLAIM))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_jwk_client() -> Optional[PyJWKClient]:
    """Return the cached PyJWKClient, constructing it lazily on first use.

    Returns None when ``COGNITO_JWKS_URL`` is unset (misconfiguration), so
    verification fails closed rather than trusting an unverified token.

    Returns:
        The module-cached ``PyJWKClient``, or None if no JWKS URL is configured.
    """
    global _jwk_client
    if _jwk_client is not None:
        return _jwk_client
    if not _JWKS_URL:
        return None
    # PyJWKClient caches signing keys per instance; constructed once per
    # execution environment. Construction does not fetch (fetch is lazy on
    # first get_signing_key_from_jwt).
    _jwk_client = PyJWKClient(_JWKS_URL, cache_keys=True)
    return _jwk_client


def _verify_token(token: str) -> Optional[dict[str, Any]]:
    """Verify a Cognito JWT and return its claims, or None (fail closed).

    Verifies the RS256 signature against the pool JWKS (key matched by the
    token's ``kid``), the issuer (``iss`` == ``COGNITO_ISSUER``), and the expiry
    (``exp``). The audience is NOT verified (Cognito access tokens carry
    ``aud=null``); instead the token TYPE and the CLIENT are pinned from the
    decoded claims by ``_claims_acceptable`` — ``token_use == "access"`` and
    ``client_id`` in ``COGNITO_ALLOWED_CLIENT_IDS`` — so this trust boundary does
    not depend on the gateway's ``allowedClients`` configuration.

    Any failure — missing config, unknown ``kid``, bad signature, wrong issuer,
    expired token, malformed token, wrong token type, unpinned client — returns
    None so the caller fails closed. Never logs the token.

    Args:
        token: The compact-serialization JWT string.

    Returns:
        The verified claims mapping, or None on any verification failure.
    """
    client = _get_jwk_client()
    if client is None or not _ISSUER:
        return None
    try:
        signing_key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=_ALGORITHMS,
            issuer=_ISSUER,
            options={"verify_aud": False, "require": ["exp", "iss"]},
        )
    except Exception:
        # Any verification/decoding error (InvalidSignature, ExpiredSignature,
        # InvalidIssuer, PyJWKClientError, DecodeError, ...) -> fail closed.
        return None

    if not isinstance(claims, dict):
        return None
    if not _claims_acceptable(claims):
        return None
    return claims


def _claims_acceptable(claims: dict[str, Any]) -> bool:
    """Return True if verified claims satisfy the accepted-token contract.

    Applies the two post-decode assertions the cryptographic checks cannot make:
    ``token_use`` MUST be exactly ``"access"``, and ``client_id`` MUST appear in
    ``COGNITO_ALLOWED_CLIENT_IDS`` (an unset/empty allow-list rejects everything).
    See the module docstring for why.

    Side-effect free, but NOT pure: the verdict depends on the env allow-list,
    which is re-read on every call (like ``_known_scopes``) so the value is never
    frozen at import time. Testable without minting a signed token.

    Args:
        claims: The claims mapping returned by a SUCCESSFUL ``jwt.decode`` —
            signature, issuer and expiry are already verified by the caller.

    Returns:
        True if the token is an access token from an allowed client, else False.

    Security:
        Never logs the claims (they identify the user) or the rejection reason.
    """
    if claims.get(_TOKEN_USE_CLAIM) != _REQUIRED_TOKEN_USE:
        return False

    allowed = _allowed_client_ids()
    if not allowed:
        return False

    client_id = claims.get(_CLIENT_ID_CLAIM)
    if not isinstance(client_id, str) or client_id not in allowed:
        return False

    return True


def _scope_from_cognito_groups(groups: Any) -> Optional[str]:
    """Map the ``cognito:groups`` claim to a single ``served_scope``.

    Applies the single-scope-group invariant: intersect the
    user's group memberships with the known scope set and return the scope IFF
    EXACTLY ONE known scope group matches. Zero matches or more than one match
    fail closed (return None).

    Selection is by SET MEMBERSHIP, never by array position: ``cognito:groups``
    array order does NOT reflect group precedence (precedence resolves to
    ``cognito:preferred_role``, not to claim order).

    Args:
        groups: The value of the ``cognito:groups`` claim. Expected to be a
            list of group-name strings; any other type fails closed.

    Returns:
        The single matching served-scope string, or None (fail closed).
    """
    if not isinstance(groups, list) or not groups:
        return None

    known = _known_scopes()
    # Deduplicate via a set so a repeated group name cannot be miscounted as
    # more than one match.
    matches = {g for g in groups if isinstance(g, str) and g in known}
    if len(matches) != 1:
        return None
    return next(iter(matches))
