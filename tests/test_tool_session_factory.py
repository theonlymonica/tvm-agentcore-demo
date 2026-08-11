"""
Safety tests for the tool-side credential-free session factory.

The tool no longer builds a ``boto3.Session`` per request. It reuses a per-thread,
CREDENTIAL-FREE session as a model-cache factory and passes the vended credentials
to ``Session.resource()`` instead. That removed a measured 97.4 ms p50 of
per-request service/resource model parsing.

Reusing an object across requests is only safe if the object carries no tenant
identity. These tests pin the three properties that make the factory session
qualify, and the one failure mode the change could have introduced:

1. The reused session is constructed with NO credentials, so it holds no tenant
   state (:func:`test_factory_session_is_built_without_credentials`).
2. The vended credentials arrive per request, at ``resource()``, and nothing else
   reaches it (:func:`test_credentials_are_passed_per_request_to_resource`).
3. The session object is REUSED across requests while a FRESH resource is built
   for each one (:func:`test_session_is_reused_but_resource_is_per_request`).
4. FAIL-CLOSED ORDERING: a missing or malformed ``context`` raises
   :class:`ScopedCredentialsError` and never reaches ``resource()``
   (:func:`test_malformed_context_never_reaches_resource`). This is the failure
   mode the optimization could have introduced: with a module-level session, a bug
   that dropped the credential kwargs would silently fall back to the DEFAULT
   CREDENTIAL CHAIN instead of erroring. The tool execution role holds no DynamoDB
   grant so the read would still be denied, but it would surface as an opaque
   ``AccessDenied`` rather than a clear contract violation.
5. THREAD SCOPE: concurrent callers each get their OWN session, never a shared
   one, and each still receives its own credentials
   (:func:`test_each_thread_gets_its_own_session`,
   :func:`test_concurrent_requests_do_not_cross_credentials`). boto3 documents
   ``Session`` and resource objects as NOT thread safe and recommends one per
   thread (https://docs.aws.amazon.com/boto3/latest/guide/session.html,
   https://docs.aws.amazon.com/boto3/latest/guide/resources.html), so the session
   is held in thread-local storage rather than shared per container.

Everything is stubbed at the ``boto3.Session`` boundary; no AWS call is made.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from common import scoped_credentials as tool_scoped_credentials
from common.scoped_credentials import (
    ScopedCredentialsError,
    documents_table_from_event,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeTable:
    """Stand-in for a boto3 DynamoDB ``Table`` resource."""

    def __init__(self, name: str, credentials: dict[str, Any]) -> None:
        self.name = name
        #: The credentials the owning resource was created with, so a test can
        #: prove which tenant's credentials this table would call with.
        self.credentials = credentials


class _FakeDynamoResource:
    """Stand-in for a boto3 ``dynamodb`` service resource."""

    def __init__(self, credentials: dict[str, Any]) -> None:
        self._credentials = credentials

    def Table(self, name: str) -> _FakeTable:  # noqa: N802 (boto3 API name)
        return _FakeTable(name, self._credentials)


class _RecordingSession:
    """Stand-in ``boto3.Session`` recording its construction and resource calls."""

    def __init__(self, construction_kwargs: dict[str, Any], log: _Log) -> None:
        self.construction_kwargs = construction_kwargs
        self._log = log

    def resource(self, service_name: str, **kwargs: Any) -> _FakeDynamoResource:
        self._log.resource_calls.append(
            {"service_name": service_name, "kwargs": kwargs, "session": self}
        )
        return _FakeDynamoResource(kwargs)


class _Log:
    """Thread-safe record of every session construction and resource creation."""

    def __init__(self) -> None:
        self.sessions: list[_RecordingSession] = []
        self.resource_calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def session_factory(self, **kwargs: Any) -> _RecordingSession:
        """Stand in for ``boto3.Session``, recording each construction.

        Args:
            **kwargs: The kwargs the production code passed to ``Session(...)``.

        Returns:
            A recording session.
        """
        session = _RecordingSession(kwargs, self)
        with self._lock:
            self.sessions.append(session)
        return session


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch, scoped_env: dict[str, str]) -> _Log:
    """Patch ``boto3.Session`` in the tool module and return the recorder.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        scoped_env: ensures ``DOCUMENTS_TABLE_NAME`` is set.

    Returns:
        The :class:`_Log` recording constructions and resource calls.
    """
    recorder = _Log()
    monkeypatch.setattr(
        tool_scoped_credentials.boto3, "Session", recorder.session_factory
    )
    tool_scoped_credentials.reset_factory_session()
    return recorder


def _event(index: int = 0) -> dict[str, Any]:
    """Build a valid tool event carrying a complete injected ``context``.

    Args:
        index: Distinguishes one caller's credentials from another's.

    Returns:
        A Lambda event with ``doc_id`` and a well-formed ``context``.
    """
    return {
        "doc_id": f"PAY-{index:03d}",
        "context": {
            "served_scope": f"scope-{index}",
            "tenant_credentials": {
                "access_key_id": f"AKIA{index}",
                "secret_access_key": f"secret{index}",
                "session_token": f"token{index}",
            },
        },
    }


# ---------------------------------------------------------------------------
# 1-3: the reused object carries no tenant identity
# ---------------------------------------------------------------------------


def test_factory_session_is_built_without_credentials(log: _Log) -> None:
    """The reused session is constructed with NO credentials at all.

    This is the property that makes reuse safe: a session holding credentials
    would be shared tenant state.
    """
    documents_table_from_event(_event())

    assert len(log.sessions) == 1
    assert log.sessions[0].construction_kwargs == {}


def test_credentials_are_passed_per_request_to_resource(
    log: _Log, scoped_env: dict[str, str]
) -> None:
    """The three vended fields map onto the boto3 credential kwargs at ``resource()``.

    Exact equality, so no extra field (``Expiration``, scope, junk) rides along.
    """
    table = documents_table_from_event(_event(7))

    assert len(log.resource_calls) == 1
    call = log.resource_calls[0]
    assert call["service_name"] == "dynamodb"
    assert call["kwargs"] == {
        "aws_access_key_id": "AKIA7",
        "aws_secret_access_key": "secret7",
        "aws_session_token": "token7",
    }
    assert table.name == scoped_env["DOCUMENTS_TABLE_NAME"]


def test_session_is_reused_but_resource_is_per_request(log: _Log) -> None:
    """Across requests on one thread: ONE session, but a FRESH resource each time.

    The session is the cacheable, tenant-agnostic half; the resource is bound to
    one tenant's credentials and must never be reused.
    """
    first = documents_table_from_event(_event(1))
    second = documents_table_from_event(_event(2))

    # One session construction for three requests-worth of work.
    assert len(log.sessions) == 1
    # But one resource creation per request, each with its own credentials.
    assert len(log.resource_calls) == 2
    assert log.resource_calls[0]["kwargs"]["aws_access_key_id"] == "AKIA1"
    assert log.resource_calls[1]["kwargs"]["aws_access_key_id"] == "AKIA2"
    # And the tables are distinct objects bound to distinct credentials.
    assert first is not second
    assert first.credentials != second.credentials


# ---------------------------------------------------------------------------
# 4: fail-closed ordering — the failure mode the optimization could introduce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param({"doc_id": "PAY-001"}, id="context-missing"),
        pytest.param({"doc_id": "PAY-001", "context": None}, id="context-none"),
        pytest.param({"doc_id": "PAY-001", "context": "nope"}, id="context-not-object"),
        pytest.param({"doc_id": "PAY-001", "context": {}}, id="context-empty"),
        pytest.param(
            {"doc_id": "PAY-001", "context": {"served_scope": "s"}},
            id="credentials-missing",
        ),
        pytest.param(
            {
                "doc_id": "PAY-001",
                "context": {
                    "served_scope": "s",
                    "tenant_credentials": {"access_key_id": "AKIA"},
                },
            },
            id="credentials-incomplete",
        ),
        pytest.param(
            {
                "doc_id": "PAY-001",
                "context": {
                    "served_scope": "",
                    "tenant_credentials": {
                        "access_key_id": "AKIA",
                        "secret_access_key": "s",
                        "session_token": "t",
                    },
                },
            },
            id="scope-empty",
        ),
    ],
)
def test_malformed_context_never_reaches_resource(
    log: _Log, event: dict[str, Any]
) -> None:
    """A malformed ``context`` raises BEFORE any resource is built.

    Validation must run first. If it did not, the credential kwargs would be
    absent from the ``resource()`` call and the factory session would silently
    fall back to the DEFAULT CREDENTIAL CHAIN — the tool's execution role —
    turning a clear contract violation into an opaque ``AccessDenied``.
    """
    with pytest.raises(ScopedCredentialsError):
        documents_table_from_event(event)

    # The decisive assertion: no resource was created, so no call could have been
    # made under any credentials, vended or ambient.
    assert log.resource_calls == []


# ---------------------------------------------------------------------------
# 5: thread scope
# ---------------------------------------------------------------------------


def test_each_thread_gets_its_own_session(log: _Log) -> None:
    """Concurrent callers never share a session; each thread builds its own.

    boto3 documents ``Session`` as not thread safe, so the factory is thread-local
    rather than shared per container. Two requests per thread prove the session is
    still reused WITHIN a thread: 4 requests across 4 threads must produce exactly
    4 sessions, not 8 and not 1.
    """
    thread_count = 4
    start = threading.Barrier(thread_count)
    seen: dict[int, list[Any]] = {}
    seen_lock = threading.Lock()

    def worker(index: int) -> None:
        start.wait()
        sessions = [
            tool_scoped_credentials._factory_session(),
            tool_scoped_credentials._factory_session(),
        ]
        documents_table_from_event(_event(index))
        documents_table_from_event(_event(index))
        with seen_lock:
            seen[index] = sessions

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # One session per thread, reused within the thread.
    assert len(log.sessions) == thread_count
    # Each thread saw the same session object on both lookups...
    for index in range(thread_count):
        assert seen[index][0] is seen[index][1]
    # ...and no two threads shared one.
    distinct = {id(sessions[0]) for sessions in seen.values()}
    assert len(distinct) == thread_count
    # Two resource creations per thread, one per request.
    assert len(log.resource_calls) == thread_count * 2


def test_concurrent_requests_do_not_cross_credentials(log: _Log) -> None:
    """Under concurrency, every table is bound to its OWN caller's credentials.

    The point of the whole design: a warm process must never serve one tenant
    through another tenant's credentials. Each thread reads with a distinct
    credential set and asserts it got exactly its own back.
    """
    thread_count = 8
    start = threading.Barrier(thread_count)
    results: dict[int, dict[str, Any]] = {}
    results_lock = threading.Lock()

    def worker(index: int) -> None:
        start.wait()
        table = documents_table_from_event(_event(index))
        with results_lock:
            results[index] = table.credentials

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == thread_count
    for index, credentials in results.items():
        assert credentials == {
            "aws_access_key_id": f"AKIA{index}",
            "aws_secret_access_key": f"secret{index}",
            "aws_session_token": f"token{index}",
        }
