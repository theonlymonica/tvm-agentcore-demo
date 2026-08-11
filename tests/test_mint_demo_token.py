"""The demo-token minter's claim checks actually reject bad tokens.

``scripts/mint_demo_token.py`` exists because the managed app client is SRP-only
and the AWS CLI cannot compute ``SRP_A``, so there is no CLI path to a token. Its
``--check`` mode mirrors the invariants enforced downstream (gateway
``allowedClients``, the interceptor's ``token_use`` / client pinning, and the
single-scope-group rule) so a misconfigured token is diagnosed locally instead of
surfacing as an opaque gateway error.

These tests cover the two pure functions only — no AWS, no network. The SRP
exchange and the stack-output lookup are I/O and are exercised by running the
script against a deployed stack.

The claim fixtures below are written independently rather than derived from the
module's own constants, so a change to those constants cannot silently make these
assertions agree with it.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mint_demo_token import check_claims, decode_claims  # noqa: E402

#: Written out by hand, NOT imported from the module under test.
_CLIENT_ID = "exampleclientid00000000000"


def _jwt(payload: dict) -> str:
    """Build a compact JWT with the given payload and a dummy header/signature.

    Args:
        payload: The claims to encode.

    Returns:
        A three-segment token. The signature is not real — nothing under test
        verifies it.
    """

    def seg(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.c2ln"


def _good(**overrides) -> dict:
    """A claim set that satisfies every invariant, with optional overrides."""
    claims = {
        "token_use": "access",
        "client_id": _CLIENT_ID,
        "cognito:groups": ["payments-core"],
        "iss": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_EXAMPLE01",
    }
    claims.update(overrides)
    return claims


class TestDecodeClaims:
    """Padding-free base64url decodes, and malformed input raises."""

    def test_round_trips_a_payload(self) -> None:
        assert decode_claims(_jwt(_good()))["client_id"] == _CLIENT_ID

    @pytest.mark.parametrize(
        "payload",
        [
            {"a": "x"},
            {"a": "xy"},
            {"a": "xyz"},
            {"a": "xyzw"},
        ],
    )
    def test_handles_every_padding_remainder(self, payload: dict) -> None:
        # base64url in a JWT is unpadded; the decoder must re-add 0-3 '='.
        assert decode_claims(_jwt(payload)) == payload

    @pytest.mark.parametrize(
        "token",
        [
            "not-a-jwt",
            "only.two",
            "a.b.c.d",
        ],
    )
    def test_rejects_wrong_segment_count(self, token: str) -> None:
        with pytest.raises(ValueError, match="3 dot-separated segments"):
            decode_claims(token)

    def test_rejects_a_payload_that_is_not_json(self) -> None:
        bad = base64.urlsafe_b64encode(b"not json").decode().rstrip("=")
        with pytest.raises(ValueError, match="not valid JSON"):
            decode_claims(f"aGVhZGVy.{bad}.c2ln")

    def test_rejects_a_payload_that_is_not_an_object(self) -> None:
        arr = base64.urlsafe_b64encode(b"[1,2]").decode().rstrip("=")
        with pytest.raises(ValueError, match="not a JSON object"):
            decode_claims(f"aGVhZGVy.{arr}.c2ln")


class TestCheckClaims:
    """Each downstream invariant is reported, and a good token is silent."""

    def test_a_good_token_has_no_problems(self) -> None:
        assert check_claims(_good(), expected_client_id=_CLIENT_ID) == []

    def test_rejects_an_id_token(self) -> None:
        problems = check_claims(
            _good(token_use="id"), expected_client_id=_CLIENT_ID
        )
        assert any("token_use" in p for p in problems), problems

    def test_rejects_a_foreign_client_id(self) -> None:
        problems = check_claims(
            _good(client_id="someoneelsesclient0000000"),
            expected_client_id=_CLIENT_ID,
        )
        assert any("client_id" in p for p in problems), problems

    def test_rejects_zero_known_scope_groups(self) -> None:
        # A group the interceptor does not know is NOT a scope; fails closed.
        problems = check_claims(
            _good(**{"cognito:groups": ["some-other-group"]}),
            expected_client_id=_CLIENT_ID,
        )
        assert any("exactly ONE" in p for p in problems), problems

    def test_rejects_two_known_scope_groups(self) -> None:
        # The ambiguous case: array order is not a precedence signal, so two
        # known scopes must fail rather than pick one.
        problems = check_claims(
            _good(**{"cognito:groups": ["payments-core", "billing-internal"]}),
            expected_client_id=_CLIENT_ID,
        )
        assert any("exactly ONE" in p for p in problems), problems

    def test_ignores_unknown_groups_alongside_one_known_scope(self) -> None:
        # Extra non-scope groups are fine; exactly one KNOWN scope is present.
        assert (
            check_claims(
                _good(**{"cognito:groups": ["payments-core", "some-team"]}),
                expected_client_id=_CLIENT_ID,
            )
            == []
        )

    @pytest.mark.parametrize("groups", [None, "payments-core", 42, {}])
    def test_rejects_groups_that_are_not_a_list(self, groups: object) -> None:
        problems = check_claims(
            _good(**{"cognito:groups": groups}), expected_client_id=_CLIENT_ID
        )
        assert any("expected a list" in p for p in problems), problems

    def test_reports_every_problem_at_once(self) -> None:
        # An operator should see all of it in one run, not one per attempt.
        problems = check_claims(
            {"token_use": "id", "client_id": "wrong", "cognito:groups": []},
            expected_client_id=_CLIENT_ID,
        )
        assert len(problems) == 3, problems
