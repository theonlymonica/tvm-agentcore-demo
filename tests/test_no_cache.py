"""Cache-absence unit test.

The vend path holds **no credential cache**: every call to
``interceptor.scoped_credentials.vend_scoped_credentials`` issues its own
``sts:AssumeRole``, and nothing is reused between calls.

Why that is a security property rather than a performance choice: the session
policy the interceptor attaches to the assume is time-bounded to about a minute,
so a cached credential set could be handed to a tool after its window had already
closed. The failure would surface inside the tool as an opaque authorization
error on a request the caller was entitled to make — or, worse, invite someone to
"fix" it by widening the window. Not caching keeps the credential lifetime and the
request lifetime the same thing.

What is asserted: with STS faked at the
``interceptor.scoped_credentials.boto3.client`` boundary (the ``_FakeSts`` /
``_RecordingFakeSts`` pattern also used by ``tests/test_context_injection.py`` and
``tests/test_vend_timing.py``), two consecutive vends made with the SAME
``(role_arn, served_scope, table_arn, actions)`` each issue their own
``assume_role`` call, and each returns a single credentials dict carrying exactly
``access_key_id`` / ``secret_access_key`` / ``session_token`` — not a tuple, and
with no cache-state flag riding along.

AWS grounding: STS ``AssumeRole`` returns temporary credentials under the
``Credentials`` key (``AccessKeyId`` / ``SecretAccessKey`` / ``SessionToken`` /
``Expiration``); the fake ``assume_role`` mirrors that response shape, including
the ``Expiration`` the vend path must drop:
https://docs.aws.amazon.com/STS/latest/APIReference/API_Credentials.html
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

import interceptor.scoped_credentials as scoped_credentials
from interceptor.scoped_credentials import READ_ACTIONS, vend_scoped_credentials

# ---------------------------------------------------------------------------
# Fixed vend-path inputs. The exact strings are irrelevant to these assertions
# (which are about caching, not scope/table/role content); they only need to be
# well-formed enough for the two calls to be truly identical.
# ---------------------------------------------------------------------------
_ROLE_ARN = "arn:aws:iam::123456789012:role/DocumentsAccessRole"
_SERVED_SCOPE = "payments-core"
_TABLE_ARN = "arn:aws:dynamodb:us-east-1:123456789012:table/DocumentsTable"

#: The exact set of keys a vended credentials dict must carry — the three
#: snake_case fields and nothing else (no cache-state flag, no ``Expiration``).
_CREDENTIAL_KEYS = {"access_key_id", "secret_access_key", "session_token"}

#: Canonical STS example credential values returned by the fake ``assume_role``.
#: ``Expiration`` is included so the fake matches the real STS ``Credentials``
#: shape; ``vend_scoped_credentials`` must drop it from its return.
_FAKE_STS_CREDENTIALS: dict[str, Any] = {
    "AccessKeyId": "ASIAIOSFODNN7EXAMPLE",
    "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "SessionToken": "IQoJb3JpZ2luX2VjEXAMPLETOKEN",
    "Expiration": datetime(2099, 1, 1, tzinfo=timezone.utc),
}


class _RecordingFakeSts:
    """Stand-in for a boto3 STS client that counts every ``assume_role`` call.

    Keeps only the call count, which is all the no-caching assertion needs.
    """

    def __init__(self, credentials: dict[str, Any]) -> None:
        self._credentials = credentials
        self.assume_role_calls = 0

    def assume_role(self, **_kwargs: Any) -> dict[str, Any]:
        """Count the call and return the fixed temporary credentials."""
        self.assume_role_calls += 1
        # Return a fresh copy so a caller mutating the dict cannot bleed into the
        # next call's response.
        return {"Credentials": dict(self._credentials)}


def test_two_identical_vends_issue_two_assume_role_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two identical vends hit STS twice and each returns a single creds dict.

    With STS mocked to count calls, ``vend_scoped_credentials`` is called twice
    with the SAME ``(role_arn, served_scope, table_arn, actions)``. Because there
    is no cache, ``assume_role`` is invoked exactly twice — a count of one would
    mean the second tool call ran on credentials minted for the first — and each
    call returns a single credentials dict with exactly the three snake_case
    fields.
    """
    sts = _RecordingFakeSts(_FAKE_STS_CREDENTIALS)
    monkeypatch.setattr(
        scoped_credentials.boto3,
        "client",
        lambda service_name, *a, **k: sts,
    )

    first = vend_scoped_credentials(_ROLE_ARN, _SERVED_SCOPE, _TABLE_ARN, READ_ACTIONS)
    second = vend_scoped_credentials(_ROLE_ARN, _SERVED_SCOPE, _TABLE_ARN, READ_ACTIONS)

    # No caching: two identical calls still issue two AssumeRole calls.
    assert sts.assume_role_calls == 2

    expected = {
        "access_key_id": _FAKE_STS_CREDENTIALS["AccessKeyId"],
        "secret_access_key": _FAKE_STS_CREDENTIALS["SecretAccessKey"],
        "session_token": _FAKE_STS_CREDENTIALS["SessionToken"],
    }
    for creds in (first, second):
        # A single credentials dict — not a (credentials, cache-state) pair.
        assert isinstance(creds, dict)
        assert not isinstance(creds, tuple)
        # Exactly the three fields: no cache flag, no `Expiration`, no junk.
        assert set(creds.keys()) == _CREDENTIAL_KEYS
        assert creds == expected
