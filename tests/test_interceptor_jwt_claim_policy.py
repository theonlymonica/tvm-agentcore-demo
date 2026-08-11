"""Tests for the interceptor's accepted-token contract.

``interceptor/jwt_claims.py`` verifies the RS256 signature, the issuer and the
expiry, but signature validity alone does NOT establish that a token is the one
KIND of token the interceptor is contracted to accept. Two post-decode assertions
close that gap, and these tests pin both:

1. ``token_use == "access"``. Cognito signs the ID token and the access token with
   the SAME JWKS and the SAME issuer, and BOTH carry ``cognito:groups`` — so
   before this check an ID token satisfied every verification the interceptor
   performed and derived a ``served_scope``.
2. ``client_id`` in ``COGNITO_ALLOWED_CLIENT_IDS``. ``verify_aud`` is False (the
   Cognito access token has ``aud=null``), so client identity lives in the
   ``client_id`` claim. Pinning it here makes the trust boundary self-sufficient
   instead of dependent on the gateway's ``allowedClients`` staying correct.

Test-design notes
-----------------
* **Real signatures, not a stubbed ``jwt.decode``.** The end-to-end class mints
  tokens with a locally generated RSA key and feeds the matching public key
  through a stub JWKS client, so ``jwt.decode`` runs for real and these tests
  exercise the actual verification path rather than a mock of it. Only the
  network-bound JWKS fetch is replaced.
* **Independently authored fixtures.** The client ids and group names here are
  hardcoded literals, NOT imports of the production constants. A test that
  re-declares the value under test can only prove the code is self-consistent;
  these assert against the threat model instead.
* **One claim differs per pair.** The ID-token cases are byte-identical to the
  accepted access-token case except for the claim under test, so a rejection
  cannot be explained by anything but that claim.
"""

from __future__ import annotations

import datetime
import types
from typing import Any, Optional

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from interceptor import jwt_claims

# ---------------------------------------------------------------------------
# Independently authored fixtures (deliberately NOT imported from cdk/)
# ---------------------------------------------------------------------------

#: The app client the interceptor is configured to trust.
_TRUSTED_CLIENT = "1example0client0id0trusted"

#: A second, untrusted client in the SAME user pool: same JWKS, same issuer, so
#: its tokens are cryptographically valid and differ only in `client_id`.
_ROGUE_CLIENT = "9rogue0client0id0same0pool9"

#: An issuer string of the documented Cognito shape.
_ISSUER = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TESTPOOL1"

#: A group name that IS in the interceptor's known-scope set, so a rejection can
#: never be attributed to the group mapping failing closed for another reason.
_RESOLVABLE_GROUP = "payments-core"


def _access_claims(**overrides: Any) -> dict[str, Any]:
    """Return a valid Cognito ACCESS-token claim set, with optional overrides.

    Args:
        **overrides: Claims to add or replace. Passing a value of ``None`` for a
            key REMOVES that claim (to model an absent claim, which is distinct
            from a claim present with a null value).

    Returns:
        The claim mapping.
    """
    claims: dict[str, Any] = {
        "iss": _ISSUER,
        "exp": datetime.datetime.now(tz=datetime.timezone.utc)
        + datetime.timedelta(hours=1),
        "token_use": "access",
        "client_id": _TRUSTED_CLIENT,
        "cognito:groups": [_RESOLVABLE_GROUP],
    }
    for key, value in overrides.items():
        if value is None:
            claims.pop(key, None)
        else:
            claims[key] = value
    return claims


def _id_token_claims(**overrides: Any) -> dict[str, Any]:
    """Return a Cognito ID-token claim set carrying the SAME groups.

    Mirrors what Cognito actually issues: ``token_use: id``, the client in ``aud``
    rather than ``client_id``, and the identical ``cognito:groups`` array — which
    is precisely why the ID token used to resolve a ``served_scope``.

    Args:
        **overrides: Claims to add or replace (``None`` removes the claim).

    Returns:
        The claim mapping.
    """
    return _access_claims(
        token_use="id", client_id=None, aud=_TRUSTED_CLIENT, **overrides
    )


@pytest.fixture
def allow_trusted_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the interceptor to accept only ``_TRUSTED_CLIENT``."""
    monkeypatch.setenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, _TRUSTED_CLIENT)


# ---------------------------------------------------------------------------
# The allow-list reader
# ---------------------------------------------------------------------------


class TestAllowedClientIds:
    """``_allowed_client_ids`` parses the env var and has NO code default."""

    def test_unset_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No code default: an unset var must yield an empty set so callers fail
        # closed rather than accepting every client in the pool.
        monkeypatch.delenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, raising=False)
        assert jwt_claims._allowed_client_ids() == frozenset()

    def test_whitespace_only_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, "  , ,  ")
        assert jwt_claims._allowed_client_ids() == frozenset()

    def test_single_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, _TRUSTED_CLIENT)
        assert jwt_claims._allowed_client_ids() == frozenset({_TRUSTED_CLIENT})

    def test_multiple_values_are_trimmed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            jwt_claims._ALLOWED_CLIENT_IDS_ENV,
            f" {_TRUSTED_CLIENT} , {_ROGUE_CLIENT} ",
        )
        assert jwt_claims._allowed_client_ids() == frozenset(
            {_TRUSTED_CLIENT, _ROGUE_CLIENT}
        )


# ---------------------------------------------------------------------------
# The claim policy (env-dependent, side-effect free)
# ---------------------------------------------------------------------------


class TestClaimsAcceptable:
    """``_claims_acceptable`` enforces token type AND client pinning."""

    def test_access_token_from_trusted_client_is_accepted(
        self, allow_trusted_client: None
    ) -> None:
        # Positive control: without this passing, every rejection below is vacuous.
        assert jwt_claims._claims_acceptable(_access_claims()) is True

    def test_id_token_is_rejected(self, allow_trusted_client: None) -> None:
        assert jwt_claims._claims_acceptable(_id_token_claims()) is False

    def test_token_use_id_alone_flips_the_verdict(
        self, allow_trusted_client: None
    ) -> None:
        # Same client_id, same groups — ONLY token_use differs. This isolates the
        # rejection to the token type and nothing else.
        accepted = _access_claims()
        rejected = _access_claims(token_use="id")
        assert jwt_claims._claims_acceptable(accepted) is True
        assert jwt_claims._claims_acceptable(rejected) is False

    def test_missing_token_use_is_rejected(
        self, allow_trusted_client: None
    ) -> None:
        assert (
            jwt_claims._claims_acceptable(_access_claims(token_use=None)) is False
        )

    @pytest.mark.parametrize(
        "value", ["Access", "ACCESS", " access", "access ", "refresh", ""]
    )
    def test_token_use_must_match_exactly(
        self, allow_trusted_client: None, value: str
    ) -> None:
        # Cognito emits the lowercase literal; anything else is not a token type
        # this interceptor recognises, so it must not be normalised into one.
        assert (
            jwt_claims._claims_acceptable(_access_claims(token_use=value)) is False
        )

    def test_rogue_client_same_pool_is_rejected(
        self, allow_trusted_client: None
    ) -> None:
        # The whole point of local pinning: a cryptographically valid access token
        # from another app client in the same pool must not be accepted.
        assert (
            jwt_claims._claims_acceptable(_access_claims(client_id=_ROGUE_CLIENT))
            is False
        )

    def test_missing_client_id_is_rejected(
        self, allow_trusted_client: None
    ) -> None:
        assert (
            jwt_claims._claims_acceptable(_access_claims(client_id=None)) is False
        )

    @pytest.mark.parametrize(
        "value",
        [
            [_TRUSTED_CLIENT],
            {"id": _TRUSTED_CLIENT},
            123,
            True,
        ],
    )
    def test_non_string_client_id_is_rejected(
        self, allow_trusted_client: None, value: Any
    ) -> None:
        # A list containing the trusted id must not satisfy the check via a
        # membership coincidence; the claim has to BE the string.
        assert (
            jwt_claims._claims_acceptable(_access_claims(client_id=value)) is False
        )

    def test_unset_allow_list_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A missing env var must NOT degrade to "accept any client" — that would
        # silently restore the dependence on gateway config this check removes.
        monkeypatch.delenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, raising=False)
        assert jwt_claims._claims_acceptable(_access_claims()) is False

    def test_empty_allow_list_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, "   ")
        assert jwt_claims._claims_acceptable(_access_claims()) is False

    def test_multi_client_allow_list_accepts_each_member(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            jwt_claims._ALLOWED_CLIENT_IDS_ENV,
            f"{_TRUSTED_CLIENT},{_ROGUE_CLIENT}",
        )
        assert jwt_claims._claims_acceptable(_access_claims()) is True
        assert (
            jwt_claims._claims_acceptable(_access_claims(client_id=_ROGUE_CLIENT))
            is True
        )


# ---------------------------------------------------------------------------
# End-to-end: real RS256 verification through the public entry point
# ---------------------------------------------------------------------------


_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _sign(claims: dict[str, Any]) -> str:
    """Sign ``claims`` as an RS256 JWT with the module's test key.

    Args:
        claims: The claim set to encode.

    Returns:
        The compact-serialization JWT.
    """
    return jwt.encode(
        claims, _PRIVATE_KEY, algorithm="RS256", headers={"kid": "test-kid"}
    )


class _StubJWKClient:
    """Stands in for ``PyJWKClient``, returning the local test public key.

    Replaces ONLY the network-bound JWKS fetch; signature verification itself
    still runs for real inside ``jwt.decode``.
    """

    def get_signing_key_from_jwt(self, token: str) -> types.SimpleNamespace:
        """Return an object exposing the public key as ``.key``.

        Args:
            token: The JWT whose signing key is requested (ignored — this stub
                holds a single key).

        Returns:
            A ``SimpleNamespace`` with a ``key`` attribute, matching the shape
            ``PyJWKClient`` returns.
        """
        return types.SimpleNamespace(key=_PRIVATE_KEY.public_key())


@pytest.fixture
def verifying_interceptor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the interceptor at the local test key and the trusted client."""
    monkeypatch.setattr(jwt_claims, "_ISSUER", _ISSUER)
    monkeypatch.setattr(jwt_claims, "_get_jwk_client", lambda: _StubJWKClient())
    monkeypatch.setenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, _TRUSTED_CLIENT)


def _resolve(claims: dict[str, Any]) -> Optional[str]:
    """Run the full public path for a freshly signed token.

    Args:
        claims: The claim set to sign and submit.

    Returns:
        The derived ``served_scope``, or None if the token was rejected.
    """
    return jwt_claims.served_scope_from_authorization(f"Bearer {_sign(claims)}")


class TestEndToEndTokenAcceptance:
    """The contract holds through ``served_scope_from_authorization``."""

    def test_valid_access_token_still_resolves_scope(
        self, verifying_interceptor: None
    ) -> None:
        # Regression guard: the new assertions must not break the supported flow.
        assert _resolve(_access_claims()) == _RESOLVABLE_GROUP

    def test_id_token_no_longer_resolves_scope(
        self, verifying_interceptor: None
    ) -> None:
        # The pre-fix behaviour: this token is correctly signed, correctly issued,
        # unexpired, and carries a resolvable group — it used to yield a scope.
        assert _resolve(_id_token_claims()) is None

    def test_access_token_from_rogue_client_does_not_resolve_scope(
        self, verifying_interceptor: None
    ) -> None:
        assert _resolve(_access_claims(client_id=_ROGUE_CLIENT)) is None

    def test_expired_access_token_still_fails_closed(
        self, verifying_interceptor: None
    ) -> None:
        # The pre-existing cryptographic checks are untouched by this change.
        stale = _access_claims(
            exp=datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(seconds=1)
        )
        assert _resolve(stale) is None

    def test_wrong_issuer_still_fails_closed(
        self, verifying_interceptor: None
    ) -> None:
        other = _access_claims(
            iss="https://cognito-idp.us-east-1.amazonaws.com/us-east-1_OTHERPOOL"
        )
        assert _resolve(other) is None

    def test_unset_allow_list_rejects_a_valid_access_token(
        self, verifying_interceptor: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Misconfiguration must fail closed, not open — same posture the module
        # already takes for an unset COGNITO_JWKS_URL / COGNITO_ISSUER.
        monkeypatch.delenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, raising=False)
        assert _resolve(_access_claims()) is None
