"""Vend-path timing/config example tests.

These are example/unit tests (not property-based) covering the wall-clock arm of
the session-policy temporal guard on the *integrated* ``vend_scoped_credentials``
path in ``interceptor/scoped_credentials.py``. The pure ``build_session_policy``
builder is covered separately by the property-based test in
``tests/test_session_policy.py``; here the concern is the behavior that the pure
builder deliberately excludes:

* the ``aws:CurrentTime`` value the vend path injects tracks the real wall clock,
  landing within ``[_SESSION_POLICY_TTL_SECONDS - 2, _SESSION_POLICY_TTL_SECONDS
  + 2]`` seconds of a UTC timestamp captured immediately before the call;
* ``sts:AssumeRole`` is called EXACTLY ONCE with ``DurationSeconds=900`` — the STS
  floor; and
* ``_SESSION_POLICY_TTL_SECONDS`` is the documented one-minute constant.

STS is stubbed at the ``interceptor.scoped_credentials.boto3.client`` boundary so
the REAL ``vend_scoped_credentials`` runs (it computes ``expires_at`` from
``datetime.now(timezone.utc)`` internally and formats ``%Y-%m-%dT%H:%M:%SZ``). The
fake STS records the kwargs it was called with — in particular the ``Policy``
session-policy JSON string and ``DurationSeconds`` — following the ``_FakeSts``
pattern in ``tests/test_context_injection.py``.

AWS documentation references:

* STS ``AssumeRole`` — the inline session policy is passed as the ``Policy``
  request parameter (a JSON string) and ``DurationSeconds`` has a minimum value of
  900 seconds:
  https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
* ``DateLessThan`` is a Date condition operator used with the ``aws:CurrentTime``
  global condition key; the value is an ISO 8601 date/time string (the doc example
  is ``"2020-06-30T23:59:59Z"``, i.e. ``%Y-%m-%dT%H:%M:%SZ``):
  https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_condition_operators.html
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

import interceptor.scoped_credentials as scoped_credentials
from interceptor.scoped_credentials import (
    READ_ACTIONS,
    vend_scoped_credentials,
)

# ---------------------------------------------------------------------------
# Fixed inputs for the vend path. The values are arbitrary — the assertions are
# about timing/config, not about the scope/table/role strings themselves.
# ---------------------------------------------------------------------------
_ROLE_ARN = "arn:aws:iam::123456789012:role/DocumentsAccessRole"
_SERVED_SCOPE = "payments-core"
_TABLE_ARN = "arn:aws:dynamodb:us-east-1:123456789012:table/DocumentsTable"

# The `%Y-%m-%dT%H:%M:%SZ` format the vend path writes and the tests parse back.
_ISO_Z = "%Y-%m-%dT%H:%M:%SZ"

# Canonical STS example credential values returned by the fake `assume_role`.
_FAKE_STS_CREDENTIALS: dict[str, Any] = {
    "AccessKeyId": "ASIAIOSFODNN7EXAMPLE",
    "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "SessionToken": "IQoJb3JpZ2luX2VjEXAMPLETOKEN",
    # STS documents Expiration as a Timestamp; irrelevant to these tests but
    # included so the fake response matches the real STS `Credentials` shape.
    "Expiration": datetime(2099, 1, 1, tzinfo=timezone.utc),
}


class _RecordingFakeSts:
    """Stand-in for a boto3 STS client that records every ``assume_role`` call.

    Follows the ``_FakeSts`` pattern in ``tests/test_context_injection.py`` but
    additionally records the kwargs of each call (so the ``Policy`` session-policy
    string and ``DurationSeconds`` can be inspected) and the call count (so the
    "exactly once" assertion can be checked).
    """

    def __init__(self, credentials: dict[str, Any]) -> None:
        self._credentials = credentials
        self.calls: list[dict[str, Any]] = []

    def assume_role(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D401
        """Record the kwargs and return the fixed temporary credentials."""
        self.calls.append(kwargs)
        return {"Credentials": self._credentials}


@pytest.fixture
def fake_sts(monkeypatch: pytest.MonkeyPatch) -> _RecordingFakeSts:
    """Patch ``interceptor.scoped_credentials.boto3.client`` to a recording fake.

    Returns the fake so a test can inspect the recorded ``assume_role`` calls
    after driving ``vend_scoped_credentials``.
    """
    sts = _RecordingFakeSts(_FAKE_STS_CREDENTIALS)
    monkeypatch.setattr(
        scoped_credentials.boto3,
        "client",
        lambda service_name, *a, **k: sts,
    )
    return sts


def _current_time_from_policy(policy_json: str) -> str:
    """Extract the ``aws:CurrentTime`` value from a session-policy JSON string.

    Args:
        policy_json: The ``Policy`` kwarg passed to ``assume_role`` (a JSON string).

    Returns:
        The ``DateLessThan`` / ``aws:CurrentTime`` value on the single ``Allow``
        statement.
    """
    policy = json.loads(policy_json)
    condition = policy["Statement"][0]["Condition"]
    return condition["DateLessThan"]["aws:CurrentTime"]


def test_vend_current_time_tracks_wall_clock(fake_sts: _RecordingFakeSts) -> None:
    """`aws:CurrentTime` lands within TTL ± 2 s of a pre-call UTC timestamp.

    Captures a UTC timestamp immediately BEFORE the vend, then parses the
    ``aws:CurrentTime`` written into the session policy and asserts it falls
    between ``_SESSION_POLICY_TTL_SECONDS - 2`` and ``_SESSION_POLICY_TTL_SECONDS
    + 2`` seconds after the captured timestamp.
    """
    before = datetime.now(timezone.utc)
    vend_scoped_credentials(_ROLE_ARN, _SERVED_SCOPE, _TABLE_ARN, READ_ACTIONS)

    assert len(fake_sts.calls) == 1
    current_time_str = _current_time_from_policy(fake_sts.calls[0]["Policy"])

    # Parses under the exact ISO 8601 whole-second UTC format the vend path uses.
    parsed = datetime.strptime(current_time_str, _ISO_Z).replace(tzinfo=timezone.utc)

    ttl = scoped_credentials._SESSION_POLICY_TTL_SECONDS
    delta_seconds = (parsed - before).total_seconds()
    assert ttl - 2 <= delta_seconds <= ttl + 2, (
        f"aws:CurrentTime {current_time_str!r} is {delta_seconds}s after the "
        f"pre-call timestamp; expected within [{ttl - 2}, {ttl + 2}]"
    )


def test_vend_calls_assume_role_once_with_duration_900(
    fake_sts: _RecordingFakeSts,
) -> None:
    """`assume_role` is called exactly once with ``DurationSeconds=900``."""
    vend_scoped_credentials(_ROLE_ARN, _SERVED_SCOPE, _TABLE_ARN, READ_ACTIONS)

    assert len(fake_sts.calls) == 1
    assert fake_sts.calls[0]["DurationSeconds"] == 900


def test_session_policy_ttl_seconds_is_sixty() -> None:
    """The temporal-window constant is the documented one minute."""
    assert scoped_credentials._SESSION_POLICY_TTL_SECONDS == 60
