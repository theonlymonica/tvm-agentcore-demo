"""
Safety tests for the shared, per-container STS client in the interceptor.

``interceptor/scoped_credentials.py`` builds the STS client once per container
instead of once per request. These tests pin the properties that make that safe,
including the one that answers "what happens when the execution role's own
credentials rotate inside a long-lived container".

The credential-rotation answer, in short: **this module never touches
credentials.** It passes no credential arguments to ``boto3.client``, so
resolution and refresh remain entirely botocore's business, exactly as before the
change. Concretely (verified against the installed SDK):

* ``boto3.client()`` delegates to ``_get_default_session()``, which returns a
  MODULE-LEVEL ``DEFAULT_SESSION`` created once per process.
* ``botocore.session.Session.get_credentials()`` caches: "If they have already
  been loaded, this will return the cached credentials."

So even when a client was constructed per request, every one of those clients was
handed the SAME credential object, resolved once per container. Client lifetime and
credential lifetime were already decoupled; sharing the client changes only the
lifetime of the client object and its HTTP connection, neither of which holds
credentials. Signing reads the credential object per request.

:func:`test_no_credentials_are_passed_to_boto3_client` is the regression guard for
that property: if someone ever "helpfully" freezes credentials into the shared
client, this test fails.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import pytest

from interceptor import scoped_credentials
from interceptor.scoped_credentials import READ_ACTIONS, vend_scoped_credentials

_ROLE_ARN = "arn:aws:iam::123456789012:role/DocumentsAccessRole"
_SERVED_SCOPE = "payments-core"
_TABLE_ARN = "arn:aws:dynamodb:us-east-1:123456789012:table/DocumentsTable"

_FAKE_CREDENTIALS: dict[str, Any] = {
    "AccessKeyId": "ASIAIOSFODNN7EXAMPLE",
    "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "SessionToken": "IQoJb3JpZ2luX2VjEXAMPLETOKEN",
    "Expiration": datetime(2099, 1, 1, tzinfo=timezone.utc),
}


class _CountingFakeSts:
    """Stand-in STS client counting its own ``assume_role`` calls."""

    def __init__(self) -> None:
        self.assume_role_calls = 0

    def assume_role(self, **_kwargs: Any) -> dict[str, Any]:
        self.assume_role_calls += 1
        return {"Credentials": _FAKE_CREDENTIALS}


class _ClientFactory:
    """Records every ``boto3.client`` call and returns one fake per call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.clients: list[_CountingFakeSts] = []
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> _CountingFakeSts:
        client = _CountingFakeSts()
        with self._lock:
            self.calls.append({"args": args, "kwargs": kwargs})
            self.clients.append(client)
        return client


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> _ClientFactory:
    """Patch ``boto3.client`` in the interceptor module and return the recorder.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        The :class:`_ClientFactory` recording constructions.
    """
    recorder = _ClientFactory()
    monkeypatch.setattr(scoped_credentials.boto3, "client", recorder)
    scoped_credentials.reset_sts_client()
    return recorder


def _vend() -> dict[str, str]:
    """Perform one vend with fixed arguments.

    Returns:
        The vended credentials dict.
    """
    return vend_scoped_credentials(
        _ROLE_ARN, _SERVED_SCOPE, _TABLE_ARN, READ_ACTIONS
    )


# ---------------------------------------------------------------------------
# Reuse semantics
# ---------------------------------------------------------------------------


def test_client_is_built_lazily_not_at_import(factory: _ClientFactory) -> None:
    """No STS client exists until the first vend needs one.

    Construction is deferred so importing the module never requires resolvable
    credentials, and so the cost lands on the container's first request rather
    than on module import.
    """
    assert factory.calls == []
    _vend()
    assert len(factory.calls) == 1


def test_client_is_reused_across_vends(factory: _ClientFactory) -> None:
    """Many vends share ONE client, while each still calls ``AssumeRole`` once.

    This is the whole optimization in one assertion: the tenant-agnostic client is
    reused, and the tenant-bound credentials are still vended fresh every time
    (no credential cache).
    """
    for _ in range(5):
        _vend()

    # One construction for five requests.
    assert len(factory.calls) == 1
    # But five separate AssumeRole calls on that one client.
    assert factory.clients[0].assume_role_calls == 5


def test_sts_client_accessor_returns_a_stable_object(
    factory: _ClientFactory,
) -> None:
    """The accessor hands back the same object on every call."""
    first = scoped_credentials._sts_client()
    second = scoped_credentials._sts_client()

    assert first is second
    assert len(factory.calls) == 1


def test_concurrent_first_calls_build_exactly_one_client(
    factory: _ClientFactory,
) -> None:
    """Racing threads through the cold path still construct only one client.

    Guards the double-checked lock. boto3 warns that calling ``boto3.client()``
    from inside a concurrent context can cause response-ordering or SSL-module
    failures (https://docs.aws.amazon.com/boto3/latest/guide/clients.html), which
    is why construction is serialized and happens once.
    """
    thread_count = 16
    start = threading.Barrier(thread_count)
    seen: list[Any] = []
    seen_lock = threading.Lock()

    def worker() -> None:
        start.wait()
        client = scoped_credentials._sts_client()
        with seen_lock:
            seen.append(client)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(factory.calls) == 1
    assert len(seen) == thread_count
    assert all(client is seen[0] for client in seen)


def test_reset_sts_client_forces_reconstruction(factory: _ClientFactory) -> None:
    """The test seam drops the singleton so the next call rebuilds it.

    Production never calls this; the suite relies on it (see
    ``conftest.reset_boto3_caches``) so a cached fake cannot outlive the
    ``monkeypatch`` that produced it.
    """
    first = scoped_credentials._sts_client()
    scoped_credentials.reset_sts_client()
    second = scoped_credentials._sts_client()

    assert first is not second
    assert len(factory.calls) == 2


# ---------------------------------------------------------------------------
# The credential-rotation invariant
# ---------------------------------------------------------------------------


def test_no_credentials_are_passed_to_boto3_client(
    factory: _ClientFactory,
) -> None:
    """The shared client is built with the service name and nothing else.

    This is the regression guard for credential rotation in a long-lived
    container. Because no credential is passed here, the client resolves through
    botocore's provider chain and signs each request from the credential object
    the session holds — so refresh behaviour is whatever botocore provides, and is
    identical to the previous per-request-client code. If someone were to freeze
    credentials into this shared client, rotation WOULD break, and this assertion
    is what fails.
    """
    _vend()

    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call["args"] == ("sts",)

    forbidden = {
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "config",
    }
    assert not (forbidden & set(call["kwargs"])), (
        "the shared STS client must not capture credentials; credential "
        f"resolution belongs to botocore. Got kwargs: {call['kwargs']!r}"
    )
    assert call["kwargs"] == {}


def test_vended_credentials_are_not_retained_by_the_module(
    factory: _ClientFactory,
) -> None:
    """Successive vends return independent dicts; nothing is memoized.

    The client is shared, but the credentials that pass through it are not
    retained anywhere in the module.
    """
    first = _vend()
    second = _vend()

    assert first == second  # same fake STS response
    assert first is not second  # but distinct objects, freshly built each time
    assert factory.clients[0].assume_role_calls == 2
