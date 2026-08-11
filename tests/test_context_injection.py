"""Property test — the ``context`` injection wire contract round-trips exactly.

The property under test: "``context`` injection contract round-trips exactly and
adds only ``context``".

The test drives the REAL end-to-end contract with only the two external
boundaries stubbed (STS and the JWT claim), following the pattern the tool-side
wire-contract tests use:

* The interceptor-side STS ``AssumeRole`` is faked at the ``boto3.client`` level
  in ``interceptor.scoped_credentials`` so the *real* ``vend_scoped_credentials``
  mapping runs — this is what proves the STS ``Credentials`` fields ``Expiration``
  and any junk keys are excluded from ``tenant_credentials``.
* ``served_scope_from_authorization`` is stubbed to return an arbitrary generated
  scope, so no real JWT is needed.
* The tool-side ``boto3.Session`` is faked in ``common.scoped_credentials`` so the
  round-trip can assert, by exact dict equality, that the three
  ``tenant_credentials`` fields map onto the boto3 session kwargs *by name* and
  that nothing else (no ``Expiration``, no junk) reaches the session.

The forwarded ``params.arguments`` the interceptor emits is fed straight back in
as the tool's Lambda ``event`` (a Lambda target receives exactly the
``params.arguments`` map), so the read-back through ``served_scope_from_event`` /
``documents_table_from_event`` is a true round-trip of what the interceptor wrote.

AWS grounding (verified against the AWS documentation):

* STS ``Credentials`` response shape — ``AccessKeyId`` (String), ``SecretAccessKey``
  (String), ``SessionToken`` (String), ``Expiration`` (Timestamp):
  https://docs.aws.amazon.com/STS/latest/APIReference/API_Credentials.html
* boto3 ``Session`` credential keyword arguments the three snake_case fields map
  onto — ``aws_access_key_id`` / ``aws_secret_access_key`` / ``aws_session_token``:
  https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_assumerole.html
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import interceptor.handler as interceptor_handler
import interceptor.scoped_credentials as interceptor_scoped_credentials

import common.scoped_credentials as tool_scoped_credentials
from common.scoped_credentials import (
    documents_table_from_event,
    served_scope_from_event,
)

# ---------------------------------------------------------------------------
# Tool names (frozen composite names). The interceptor injects
# `context` for the three scoped tools and passes everything else through.
# ---------------------------------------------------------------------------
SCOPED_COMPOSITE_NAMES = (
    "ReadDocument___read_document",
    "SearchDocuments___search_documents",
    "Reply___reply",
)
OUT_OF_SCOPE_NAME = "OtherTarget___not_a_scoped_tool"

# Model-supplied argument keys the interceptor MUST leave unmodified.
# Deliberately excludes `served_scope` and the retired flat credential fields so
# the "no top-level served_scope / no flat fields" assertions are meaningful.
MODEL_ARG_KEYS = ("doc_id", "document_id", "query", "body")

# STS Credentials keys the mapping consumes; junk keys must avoid these.
_STS_CORE_KEYS = frozenset(
    {"AccessKeyId", "SecretAccessKey", "SessionToken", "Expiration"}
)

# Printable, whitespace-free ASCII so strip() is a no-op and scopes/creds stay
# non-empty (round-trip is exact rather than needing to model whitespace).
_NO_WS = st.characters(min_codepoint=33, max_codepoint=126)


# ---------------------------------------------------------------------------
# Test doubles for the two external boundaries (STS) and the tool-side session
# ---------------------------------------------------------------------------


class _FakeTable:
    """Stand-in for a boto3 DynamoDB ``Table`` resource (records its name)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeDynamoResource:
    """Stand-in for a boto3 ``dynamodb`` service resource."""

    def Table(self, name: str) -> _FakeTable:  # noqa: N802 (boto3 API name)
        return _FakeTable(name)


class _FakeSession:
    """Stand-in for ``boto3.Session`` that only yields a dynamodb resource.

    The tool now reuses a per-thread, credential-free session as a model-cache
    factory and passes the vended credentials to ``resource()`` instead of to
    ``Session(...)``, so this fake records the kwargs it receives at the
    ``resource()`` call — that is where the credential mapping now lands.
    """

    def __init__(self, captured: dict[str, Any] | None = None) -> None:
        self._captured = captured

    def resource(self, service_name: str, **kwargs: Any) -> _FakeDynamoResource:
        assert service_name == "dynamodb"
        if self._captured is not None:
            self._captured["kwargs"] = kwargs
        return _FakeDynamoResource()


class _FakeSts:
    """Stand-in for a boto3 STS client whose ``assume_role`` returns fixed creds."""

    def __init__(self, credentials: dict[str, Any]) -> None:
        self._credentials = credentials

    def assume_role(self, **_kwargs: Any) -> dict[str, Any]:
        # STS returns the temporary credentials under the `Credentials` key.
        return {"Credentials": self._credentials}


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


def _scopes() -> st.SearchStrategy[str]:
    """Arbitrary non-empty, whitespace-free, WILDCARD-FREE served-scope strings.

    ``vend_scoped_credentials`` refuses to vend for a scope containing an IAM
    wildcard (``*`` / ``?``), because the scoped roles' identity policies compare
    ``dynamodb:LeadingKeys`` against ``${aws:PrincipalTag/scope}`` with
    ``StringLike`` — a wildcard in the tag value would widen that comparison to
    every partition. Such a scope therefore fails CLOSED and produces a
    short-circuit error instead of a transformed request, which is a different
    contract from the one under test here.

    The rejection itself is covered by
    ``tests/test_scope_tag_abac.py::TestUntaggableScopeFailsClosed`` (including
    that no credential is minted), so excluding the characters here removes an
    overlap, not coverage.
    """
    return st.text(
        alphabet=st.characters(
            min_codepoint=33, max_codepoint=126, exclude_characters="*?"
        ),
        min_size=1,
        max_size=64,
    )


def _cred_value() -> st.SearchStrategy[str]:
    """Arbitrary non-empty, whitespace-free credential value strings."""
    return st.text(alphabet=_NO_WS, min_size=1, max_size=48)


@st.composite
def _sts_credentials(draw: st.DrawFn) -> dict[str, Any]:
    """Build an arbitrary STS ``Credentials`` dict with Expiration + junk keys.

    The three mapped fields plus a Timestamp-typed ``Expiration`` and arbitrary
    extra keys, so the test proves ``Expiration`` and junk are excluded from the
    vended ``tenant_credentials``. The core keys are applied last so a junk key
    can never shadow them.
    """
    junk = draw(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=12).filter(
                lambda k: k not in _STS_CORE_KEYS
            ),
            values=st.one_of(
                st.text(max_size=16),
                st.integers(),
                st.booleans(),
                st.none(),
                st.lists(st.text(max_size=8), max_size=3),
            ),
            max_size=5,
        )
    )
    return {
        **junk,
        "AccessKeyId": draw(_cred_value()),
        "SecretAccessKey": draw(_cred_value()),
        "SessionToken": draw(_cred_value()),
        # STS documents Expiration as a Timestamp; it must never reach the wire.
        "Expiration": draw(st.datetimes()),
    }


def _model_arguments() -> st.SearchStrategy[dict[str, str]]:
    """Arbitrary model-supplied argument maps (subset of MODEL_ARG_KEYS)."""
    return st.dictionaries(
        keys=st.sampled_from(MODEL_ARG_KEYS),
        values=st.text(max_size=40),
        max_size=len(MODEL_ARG_KEYS),
    )


def _preexisting_context() -> st.SearchStrategy[tuple[bool, Any]]:
    """(include?, value) for a model-supplied ``arguments["context"]`` to overwrite."""
    return st.tuples(
        st.booleans(),
        st.one_of(
            st.none(),
            st.text(max_size=16),
            st.integers(),
            st.dictionaries(st.text(max_size=6), st.text(max_size=6), max_size=3),
            st.lists(st.text(max_size=6), max_size=3),
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tools_call_event(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a REQUEST-interceptor ``tools/call`` input event."""
    return {
        "mcp": {
            "gatewayRequest": {
                "headers": {"Authorization": "Bearer token"},
                "body": {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
            }
        }
    }


def _forwarded_arguments(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the forwarded ``params.arguments`` from an allow envelope."""
    return result["mcp"]["transformedGatewayRequest"]["body"]["params"]["arguments"]


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    scope=_scopes(),
    sts_credentials=_sts_credentials(),
    model_args=_model_arguments(),
    preexisting=_preexisting_context(),
    tool_name=st.sampled_from(SCOPED_COMPOSITE_NAMES + (OUT_OF_SCOPE_NAME,)),
)
def test_context_injection_contract(
    scope: str,
    sts_credentials: dict[str, Any],
    model_args: dict[str, str],
    preexisting: tuple[bool, Any],
    tool_name: str,
    monkeypatch: pytest.MonkeyPatch,
    scoped_env: dict[str, str],
) -> None:
    """`context` injection round-trips exactly and adds only `context`.

    For all served scopes and all STS ``AssumeRole`` responses (including
    responses carrying ``Expiration`` and arbitrary junk keys), the REQUEST
    interceptor writes at ``arguments["context"]`` an object with exactly
    ``served_scope`` (the input scope) and ``tenant_credentials`` holding exactly
    the three snake_case fields mapped from the STS ``AccessKeyId`` /
    ``SecretAccessKey`` / ``SessionToken`` (no ``Expiration``, no junk); reading it
    back through the tool-side helpers recovers the scope and maps the three
    values onto ``aws_access_key_id`` / ``aws_secret_access_key`` /
    ``aws_session_token`` by name; any pre-existing ``arguments["context"]`` is
    overwritten wholesale; the forwarded arguments equal the model-supplied
    arguments plus exactly one added key ``context`` (no top-level ``served_scope``
    and no flat credential fields, model args untouched); and on the
    out-of-scope pass-through path no key is added at all.
    """
    # --- Stub the two external boundaries (STS + JWT) and the tool-side session.
    # Fake STS at boto3.client level so the REAL vend mapping runs.
    monkeypatch.setattr(
        interceptor_scoped_credentials.boto3,
        "client",
        lambda service_name, *a, **k: _FakeSts(sts_credentials),
    )
    # Derive a fixed scope without a real JWT.
    monkeypatch.setattr(
        interceptor_handler,
        "served_scope_from_authorization",
        lambda _auth: scope,
    )
    # Capture the kwargs the tool-side session is built with.
    captured: dict[str, Any] = {}

    def _fake_session(**kwargs: Any) -> _FakeSession:
        # The reused factory session MUST be constructed with no credentials —
        # it is shared across requests, so any credential on it would be shared
        # tenant state. The vended credentials must arrive at resource() instead.
        captured["session_kwargs"] = kwargs
        return _FakeSession(captured)

    monkeypatch.setattr(tool_scoped_credentials.boto3, "Session", _fake_session)

    # Hypothesis runs MANY examples inside a single test function, while fixtures
    # (including conftest's autouse reset) run once per function. The STS client
    # and the factory session are now built once and reused, so without an
    # explicit per-example reset the FIRST example's fakes would be reused by
    # every later example and the credential assertions below would compare this
    # example's expected values against the previous example's fake.
    interceptor_scoped_credentials.reset_sts_client()
    tool_scoped_credentials.reset_factory_session()

    # There is still no CREDENTIAL cache: every call vends fresh from the current
    # example's fake STS — no cross-example credential reuse is possible. Only the
    # tenant-agnostic client/session objects are reused.

    # --- Build the model-supplied arguments (optionally with a stale `context`).
    arguments_input: dict[str, Any] = dict(model_args)
    include_ctx, preexisting_value = preexisting
    if include_ctx:
        arguments_input["context"] = preexisting_value
    snapshot = copy.deepcopy(arguments_input)

    result = interceptor_handler.handler(
        _tools_call_event(tool_name, arguments_input), None
    )
    out_args = _forwarded_arguments(result)

    # --- Out-of-scope pass-through: zero keys added, arguments unchanged. -------
    if tool_name not in SCOPED_COMPOSITE_NAMES:
        assert out_args == snapshot
        return

    # --- Scoped path: exactly one added key `context`; model args untouched. ----
    # Exactly the model keys plus `context` — this exact-key-set check already
    # proves no flat credential fields (and no top-level served_scope) are added.
    assert set(out_args.keys()) == set(snapshot.keys()) | {"context"}
    # No top-level served_scope argument is written on any path.
    assert "served_scope" not in out_args
    # Every model-supplied argument is left exactly as the model wrote it.
    for key, value in snapshot.items():
        if key == "context":
            continue  # a stale context is overwritten, asserted below
        assert out_args[key] == value

    # --- `context` shape is exactly served_scope + tenant_credentials. ----------
    ctx = out_args["context"]
    assert set(ctx.keys()) == {"served_scope", "tenant_credentials"}
    assert ctx["served_scope"] == scope  # overwritten wholesale

    tenant_credentials = ctx["tenant_credentials"]
    assert set(tenant_credentials.keys()) == {
        "access_key_id",
        "secret_access_key",
        "session_token",
    }
    # Mapped from the STS response fields...
    assert tenant_credentials["access_key_id"] == sts_credentials["AccessKeyId"]
    assert tenant_credentials["secret_access_key"] == sts_credentials["SecretAccessKey"]
    assert tenant_credentials["session_token"] == sts_credentials["SessionToken"]
    # ...with Expiration and every other STS key excluded.
    for sts_key in sts_credentials:
        if sts_key not in ("AccessKeyId", "SecretAccessKey", "SessionToken"):
            assert sts_key not in tenant_credentials

    # --- Tool side reads it back and round-trips exactly. -----------------------
    assert served_scope_from_event(out_args) == scope
    table = documents_table_from_event(out_args)
    assert table.name == scoped_env["DOCUMENTS_TABLE_NAME"]
    # The reused factory session carries NO credentials (no shared tenant state).
    assert captured["session_kwargs"] == {}
    # The three fields map onto the boto3 credential kwargs BY NAME and nothing
    # else (no Expiration, no junk) reaches the resource — proven by exact
    # equality. Unchanged assertion; only the seam moved from Session() to
    # resource(), because that is where the credentials are now supplied.
    assert captured["kwargs"] == {
        "aws_access_key_id": sts_credentials["AccessKeyId"],
        "aws_secret_access_key": sts_credentials["SecretAccessKey"],
        "aws_session_token": sts_credentials["SessionToken"],
    }
