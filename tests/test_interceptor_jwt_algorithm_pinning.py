"""Tests for the interceptor's JWS algorithm handling.

``interceptor/jwt_claims._verify_token`` passes ``algorithms=_ALGORITHMS``
(``["RS256"]``) to ``jwt.decode``. Two classic JWS substitution attacks are in
scope:

1. **HS256 confusion.** The attacker takes the pool's PUBLIC key — published at
   the JWKS endpoint and therefore not a secret — re-signs a payload of their
   choosing with HMAC-SHA256 using that public key as the shared secret, and sets
   ``alg: HS256``. A verifier that honours the header's ``alg`` HMACs with "the
   key" and the forgery verifies.
2. **``alg: none``.** An unsigned token with an empty signature.

What these tests can and cannot attribute
-----------------------------------------
Both forgeries are rejected — but **not solely because of the pinned list**, and
saying otherwise would be false. ``_verify_token`` hands ``jwt.decode`` the
``cryptography`` ``RSAPublicKey`` OBJECT returned by ``PyJWKClient``, and PyJWT
independently refuses to use an asymmetric key as an HMAC secret
(``InvalidKeyError``) or to accept ``alg: none`` alongside a key. Verified by
mutation: widening the list to ``["RS256", "HS256"]`` leaves every behavioural
test below green, because the second barrier still fires.

So there are two independent barriers, and the behavioural tests cover the
OUTCOME an attacker cares about (a forged token derives no scope) without being
able to prove which barrier produced it. Covering the pinning DECISION therefore
needs a direct assertion on ``_ALGORITHMS`` — that is what
``test_only_rs256_is_pinned`` is for, and it is the one test here that fails when
the list is widened.

Neither attack had any coverage before (the pinning predates this change); the gap
surfaced during a security review.

Test-design notes
-----------------
* **Tokens are forged by hand**, not with ``jwt.encode``. PyJWT refuses to use an
  asymmetric PEM as an HMAC secret, so the attack cannot be expressed through its
  encoder — which is the point: an attacker does not use our library's guard
  rails. ``_forge`` assembles the JWS directly from base64url segments.
* **The forgery is proved well-formed first.** A rejection is only evidence of
  anything if the token would otherwise have been accepted, so
  ``test_forged_token_is_well_formed_when_its_alg_is_allowed`` decodes a
  hand-built HS256 token with a plain secret and a permissive algorithm list and
  asserts it verifies. Without that control these tests would pass just as
  happily against a malformed string.
* Fixtures are hardcoded literals, deliberately not imports of the production
  constants, and this module is self-contained (its own key, stub and claim
  builder) so it stays well inside the ``code-modularity`` line limit.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import types
from typing import Any, Optional

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from interceptor import jwt_claims

_TRUSTED_CLIENT = "1example0client0id0trusted"
_ISSUER = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TESTPOOL1"
_RESOLVABLE_GROUP = "payments-core"
_KID = "test-kid"

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)

#: The pool's public key in the form an attacker would fetch it from the JWKS
#: endpoint — the "secret" the HS256 confusion attack HMACs with.
_PUBLIC_KEY_PEM = (
    _PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
)


def _claims() -> dict[str, Any]:
    """Return a JSON-serialisable, otherwise-valid access-token claim set.

    ``exp`` is an integer NumericDate because these claims are serialised by hand
    rather than by ``jwt.encode``.

    Returns:
        The claim mapping.
    """
    expires = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(
        hours=1
    )
    return {
        "iss": _ISSUER,
        "exp": int(expires.timestamp()),
        "token_use": "access",
        "client_id": _TRUSTED_CLIENT,
        "cognito:groups": [_RESOLVABLE_GROUP],
    }


def _b64(raw: bytes) -> str:
    """Return unpadded base64url text for one JWS segment.

    Args:
        raw: The bytes to encode.

    Returns:
        The base64url string with ``=`` padding stripped, per RFC 7515 §2.
    """
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _forge(alg: str, secret: Optional[bytes], claims: dict[str, Any]) -> str:
    """Assemble a JWS by hand with an attacker-chosen ``alg`` header.

    Args:
        alg: The value to put in the header's ``alg`` field.
        secret: HMAC key to sign the signing input with, or None for an empty
            signature (``alg: none``).
        claims: The payload.

    Returns:
        The compact-serialization token.
    """
    header = {"alg": alg, "typ": "JWT", "kid": _KID}
    signing_input = (
        f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(claims).encode())}"
    )
    signature = (
        b""
        if secret is None
        else hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{signing_input}.{_b64(signature)}"


class _StubJWKClient:
    """Returns the local test public key in place of a JWKS fetch."""

    def get_signing_key_from_jwt(self, token: str) -> types.SimpleNamespace:
        """Return an object exposing the public key as ``.key``.

        Args:
            token: Ignored — this stub holds a single key.

        Returns:
            A ``SimpleNamespace`` matching ``PyJWKClient``'s return shape.
        """
        return types.SimpleNamespace(key=_PRIVATE_KEY.public_key())


@pytest.fixture
def verifying_interceptor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the interceptor at the local test key and the trusted client."""
    monkeypatch.setattr(jwt_claims, "_ISSUER", _ISSUER)
    monkeypatch.setattr(jwt_claims, "_get_jwk_client", lambda: _StubJWKClient())
    monkeypatch.setenv(jwt_claims._ALLOWED_CLIENT_IDS_ENV, _TRUSTED_CLIENT)


def _resolve(token: str) -> Optional[str]:
    """Submit a token through the public entry point.

    Args:
        token: The compact-serialization JWT.

    Returns:
        The derived ``served_scope``, or None if the token was rejected.
    """
    return jwt_claims.served_scope_from_authorization(f"Bearer {token}")


class TestAlgorithmPinning:
    """A forged HS256 / unsigned token derives no scope, and RS256 stays pinned."""

    def test_only_rs256_is_pinned(self) -> None:
        # The ONLY test here that fails when the pinned list is widened. The
        # behavioural tests below cannot cover this decision, because PyJWT's
        # asymmetric-key-as-HMAC-secret guard rejects the forgery independently
        # (see the module docstring) — so the decision is asserted directly.
        assert jwt_claims._ALGORITHMS == ["RS256"], (
            "the interceptor must accept RS256 only: widening this list lets the "
            "token header's alg choose the primitive"
        )

    def test_forged_token_is_well_formed_when_its_alg_is_allowed(self) -> None:
        # Control for the rejection tests below: prove `_forge` produces a token
        # PyJWT genuinely accepts when the algorithm is permitted, so a rejection
        # downstream is not merely a malformed string being thrown out.
        # 32 bytes: RFC 7518 §3.2 sets that as the SHA-256 HMAC minimum, and a
        # shorter secret makes PyJWT emit InsecureKeyLengthWarning.
        secret = b"a-plain-symmetric-secret-32bytes"
        token = _forge("HS256", secret, _claims())

        decoded = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            issuer=_ISSUER,
            options={"verify_aud": False, "require": ["exp", "iss"]},
        )

        assert decoded["client_id"] == _TRUSTED_CLIENT
        assert decoded["cognito:groups"] == [_RESOLVABLE_GROUP]

    def test_hs256_signed_with_the_public_key_is_rejected(
        self, verifying_interceptor: None
    ) -> None:
        # The classic confusion attack. Every claim is exactly what the accepted
        # access token carries — only the signing algorithm differs. Asserts the
        # OUTCOME (no scope); attribution between the two barriers is not claimed.
        token = _forge("HS256", _PUBLIC_KEY_PEM, _claims())

        assert _resolve(token) is None

    def test_alg_none_unsigned_token_is_rejected(
        self, verifying_interceptor: None
    ) -> None:
        token = _forge("none", None, _claims())

        assert _resolve(token) is None

    def test_alg_none_unsigned_token_is_rejected(
        self, verifying_interceptor: None
    ) -> None:
        token = _forge("none", None, _claims())

        assert _resolve(token) is None

    def test_genuine_rs256_token_still_resolves(
        self, verifying_interceptor: None
    ) -> None:
        # Positive control: the rejections above are not the module failing closed
        # on these claims for some unrelated reason.
        token = jwt.encode(
            _claims(), _PRIVATE_KEY, algorithm="RS256", headers={"kid": _KID}
        )

        assert _resolve(token) == _RESOLVABLE_GROUP
