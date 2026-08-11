"""Property test — tools and the vend path fail closed and echo nothing.

The property under test: "tools and the vend path fail closed and echo no
credential or argument material".

The property is universally quantified over two independent fail-closed
surfaces, both exercised in every generated example:

* **Surface A — the tool side.** For all *malformed* injected
  ``context`` objects (absent, non-object, missing/empty ``served_scope``, or an
  incomplete/empty ``tenant_credentials``), each of the three tool handlers
  (``read_document``, ``search_documents``, ``reply``) returns a GENERIC error
  dict, builds NO DynamoDB client from any source other than
  ``context.tenant_credentials`` (in particular it never falls back to its
  execution role or the default credential chain), and echoes no credential key,
  no credential value, no served-scope value, and no model-supplied argument value
  anywhere in the response.

* **Surface B — the vend path.** For all STS ``AssumeRole`` failures
  (``ClientError`` and ``BotoCoreError``), ``vend_scoped_credentials`` FAILS
  CLOSED: it propagates the error (returning no credential), and the propagated
  exception discloses no scope, role, table, or credential detail beyond the
  botocore default (the vend code adds no leaking detail).

Test doubles follow the fake-STS / fake-session pattern established in
``tests/test_context_injection.py`` and ``tests/test_vend_timing.py``:

* the tool-side ``boto3.Session`` factory
  (``common.scoped_credentials.boto3.Session``) is monkeypatched to RECORD every
  construction, so Surface A can assert it was NEVER called on a malformed
  ``context`` (the fallback-detection probe);
* the interceptor-side STS client (``interceptor.scoped_credentials.boto3.client``)
  is monkeypatched to a fake whose ``assume_role`` raises the chosen botocore
  error, so Surface B drives the REAL ``vend_scoped_credentials`` failure path.

All generated credential / scope / argument values carry distinctive, prefixed
markers (e.g. ``CREDVAL_...``, ``SCOPEVAL_...``, ``ARGDOC_...``) that can never be
a substring of a generic error message, so the "no echo" assertions are both
sound (no coincidental substring match against a generic message) and meaningful
(a genuine echo would surface the marker).

AWS grounding (verified against the AWS documentation):

* STS ``AssumeRole`` — the inline session policy rides in ``Policy`` and the call
  raises a botocore ``ClientError`` / ``BotoCoreError`` on failure, which the vend
  path propagates unchanged (fail closed):
  https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
* boto3 ``Session`` credential keyword arguments the tool side would build a
  client from — never reached on a malformed ``context``:
  https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_assumerole.html
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from botocore.exceptions import BotoCoreError, ClientError

# Tool handlers (tools/ is on sys.path via the root conftest.py).
from read_document.handler import handler as read_document_handler
from search_documents.handler import handler as search_documents_handler
from reply.handler import handler as reply_handler

# Modules monkeypatched to detect a credential-source fallback (tool side) and to
# raise from a fake STS (vend path).
import common.scoped_credentials as tool_scoped_credentials
import interceptor.scoped_credentials as interceptor_scoped_credentials
from interceptor.scoped_credentials import (
    READ_ACTIONS,
    WRITE_ACTIONS,
    vend_scoped_credentials,
)

# Credential-shaped keys that MUST NOT appear anywhere in a tool response.
# The three snake_case fields plus the two container keys.
_FORBIDDEN_KEYS = frozenset(
    {
        "context",
        "tenant_credentials",
        "access_key_id",
        "secret_access_key",
        "session_token",
    }
)

# The three completeness fields inside a well-formed ``tenant_credentials`` object.
_CRED_FIELDS = ("access_key_id", "secret_access_key", "session_token")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeTable:
    """Stand-in for a boto3 DynamoDB ``Table`` (only reached on a regression)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeDynamoResource:
    """Stand-in for a boto3 ``dynamodb`` service resource."""

    def Table(self, name: str) -> _FakeTable:  # noqa: N802 (boto3 API name)
        return _FakeTable(name)


class _FakeSession:
    """Stand-in for ``boto3.Session`` yielding a dynamodb resource.

    Accepts (and ignores) the credential kwargs that the tool now passes to
    ``resource()`` rather than to ``Session(...)``.
    """

    def resource(self, service_name: str, **kwargs: Any) -> _FakeDynamoResource:
        return _FakeDynamoResource()


class _RaisingSts:
    """Stand-in STS client whose ``assume_role`` always raises ``error``."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def assume_role(self, **_kwargs: Any) -> dict[str, Any]:
        raise self._error


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# A distinctive token alphabet: uppercase + digits only, so a prefixed marker can
# never be a substring of any lowercase generic error message.
_TOKEN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _marker(prefix: str) -> st.SearchStrategy[str]:
    """Return a strategy for distinctive, non-empty ``PREFIX_XXXX`` markers."""
    return st.text(alphabet=_TOKEN_ALPHABET, min_size=6, max_size=14).map(
        lambda body: f"{prefix}_{body}"
    )


def _full_credentials(draw: st.DrawFn) -> dict[str, str]:
    """Build a complete, valid three-field ``tenant_credentials`` dict."""
    return {field: draw(_marker("CREDVAL")) for field in _CRED_FIELDS}


@st.composite
def _malformed_contexts(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a genuinely MALFORMED ``context`` and the secrets embedded in it.

    Every returned shape is guaranteed to fail the tool-side fail-closed
    validation (``validated_context``), so the tool never reaches the
    credential-source build.

    Returns:
        A dict with keys ``present`` (whether ``event`` carries a ``context``
        key), ``value`` (the malformed context value when present), and
        ``secrets`` (the credential / served-scope string values embedded in the
        malformed context, which the tool response must never echo).
    """
    kind = draw(st.sampled_from(["missing", "non_object", "bad_scope", "bad_creds"]))

    if kind == "missing":
        # No ``context`` key on the event at all.
        return {"present": False, "value": None, "secrets": []}

    if kind == "non_object":
        # ``context`` present but not a dict.
        value = draw(
            st.one_of(
                st.none(),
                st.booleans(),
                st.integers(),
                _marker("CTXSTR"),
                st.lists(_marker("CTXITEM"), max_size=3),
            )
        )
        return {"present": True, "value": value, "secrets": []}

    scope_value = draw(_marker("SCOPEVAL"))
    full_creds = _full_credentials(draw)

    if kind == "bad_scope":
        # Valid, complete credentials but an invalid served_scope: validation
        # fails at the scope check before the credentials are ever consumed.
        bad_scope = draw(
            st.one_of(st.none(), st.just(""), st.just("   "), st.integers(), st.booleans())
        )
        context = {"served_scope": bad_scope, "tenant_credentials": full_creds}
        return {"present": True, "value": context, "secrets": list(full_creds.values())}

    # kind == "bad_creds": valid served_scope, malformed tenant_credentials.
    variant = draw(
        st.sampled_from(
            ["missing", "not_dict", "missing_field", "empty_value", "non_string_value"]
        )
    )

    if variant == "missing":
        context = {"served_scope": scope_value}
        return {"present": True, "value": context, "secrets": [scope_value]}

    if variant == "not_dict":
        bad = draw(
            st.one_of(
                st.none(),
                st.integers(),
                _marker("CREDSSTR"),
                st.lists(_marker("CREDSITEM"), max_size=2),
            )
        )
        context = {"served_scope": scope_value, "tenant_credentials": bad}
        return {"present": True, "value": context, "secrets": [scope_value]}

    if variant == "missing_field":
        dropped = draw(st.sampled_from(_CRED_FIELDS))
        partial = {k: v for k, v in full_creds.items() if k != dropped}
        context = {"served_scope": scope_value, "tenant_credentials": partial}
        return {
            "present": True,
            "value": context,
            "secrets": [scope_value, *partial.values()],
        }

    if variant == "empty_value":
        field = draw(st.sampled_from(_CRED_FIELDS))
        creds = dict(full_creds)
        creds[field] = ""
        context = {"served_scope": scope_value, "tenant_credentials": creds}
        secrets = [scope_value, *(v for v in creds.values() if v)]
        return {"present": True, "value": context, "secrets": secrets}

    # variant == "non_string_value"
    field = draw(st.sampled_from(_CRED_FIELDS))
    creds = dict(full_creds)
    creds[field] = draw(st.one_of(st.none(), st.integers(), st.booleans()))
    context = {"served_scope": scope_value, "tenant_credentials": creds}
    secrets = [scope_value, *(v for v in creds.values() if isinstance(v, str) and v)]
    return {"present": True, "value": context, "secrets": secrets}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_keys_and_strings(
    obj: Any, keys: set[str], strings: list[str]
) -> None:
    """Recursively collect every dict key and every leaf string in ``obj``."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(key)
            _collect_keys_and_strings(value, keys, strings)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_keys_and_strings(item, keys, strings)
    elif isinstance(obj, str):
        strings.append(obj)


def _assert_generic_error_no_echo(
    response: dict[str, Any], secret_values: list[str]
) -> None:
    """Assert ``response`` is a generic error that echoes no forbidden material.

    Args:
        response: A tool handler response.
        secret_values: Credential / served-scope / argument marker values that
            must appear nowhere in the response.
    """
    # A generic error dict carrying only the ``error`` key: no raw event,
    # arguments, or context object can ride alongside it.
    assert isinstance(response, dict)
    assert set(response.keys()) == {"error"}

    keys: set[str] = set()
    strings: list[str] = []
    _collect_keys_and_strings(response, keys, strings)

    # Recursively, no credential-shaped key is present.
    assert _FORBIDDEN_KEYS.isdisjoint(keys)

    # None of the generated credential / scope / argument values is echoed.
    blob = "\n".join(strings)
    for secret in secret_values:
        assert secret not in blob


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    malformed=_malformed_contexts(),
    doc_arg=_marker("ARGDOC"),
    query_arg=_marker("ARGQUERY"),
    body_arg=_marker("ARGBODY"),
    role_arn=_marker("ROLE"),
    vend_scope=_marker("VENDSCOPE"),
    table_arn=_marker("TABLE"),
    actions=st.sampled_from((READ_ACTIONS, WRITE_ACTIONS)),
    error_kind=st.sampled_from(("client_error", "botocore_error")),
)
def test_fail_closed_no_echo(
    malformed: dict[str, Any],
    doc_arg: str,
    query_arg: str,
    body_arg: str,
    role_arn: str,
    vend_scope: str,
    table_arn: str,
    actions: list[str],
    error_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    scoped_env: dict[str, str],
) -> None:
    """Tools and the vend path fail closed and echo no credential/argument material.

    Surface A: every tool handler, given a malformed ``context``, returns a
    generic error, builds no DynamoDB client (the recorded ``boto3.Session``
    factory is never called — the fallback probe), and echoes no
    credential/scope/argument value or forbidden key. Surface B:
    ``vend_scoped_credentials`` propagates any STS ``ClientError`` /
    ``BotoCoreError`` (returning no credential) and adds no leaking scope/role/
    table detail to the raised exception.
    """
    # --- Surface A: tool side, malformed context -----------------------------
    # Record every tool-side boto3.Session construction. On a malformed context
    # the tool MUST fail closed before building any client, so this stays empty
    # (no execution-role / default-chain fallback).
    session_calls: list[dict[str, Any]] = []

    def _recording_session(**kwargs: Any) -> _FakeSession:
        session_calls.append(kwargs)
        return _FakeSession()

    monkeypatch.setattr(tool_scoped_credentials.boto3, "Session", _recording_session)
    # Hypothesis runs many examples per test function; drop the reused factory
    # session so THIS example's recorder is the one the tool would reach. Without
    # this, a session cached by an earlier example would keep `session_calls`
    # empty for the wrong reason and silently weaken the probe below.
    tool_scoped_credentials.reset_factory_session()

    context_part: dict[str, Any] = (
        {"context": malformed["value"]} if malformed["present"] else {}
    )
    # Distinctive, valid model args so search/reply proceed past their own
    # argument validation to the context fail-closed path, and so any echo of an
    # argument value would surface a marker.
    tool_events = (
        (read_document_handler, {"doc_id": doc_arg, **context_part}),
        (search_documents_handler, {"query": query_arg, **context_part}),
        (reply_handler, {"doc_id": doc_arg, "body": body_arg, **context_part}),
    )
    tool_secrets = [*malformed["secrets"], doc_arg, query_arg, body_arg]

    for handler_fn, event in tool_events:
        response = handler_fn(event, None)
        _assert_generic_error_no_echo(response, tool_secrets)

    # No DynamoDB client was built from ANY source: the tool never fell
    # back to its execution role or the default credential chain.
    assert session_calls == []

    # --- Surface B: vend path, STS failure -----------------------------------
    if error_kind == "client_error":
        sts_error: BaseException = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "not authorized"}},
            "AssumeRole",
        )
    else:
        sts_error = BotoCoreError()

    monkeypatch.setattr(
        interceptor_scoped_credentials.boto3,
        "client",
        lambda service_name, *a, **k: _RaisingSts(sts_error),
    )
    # Per-example reset for the same reason: the STS client is reused per
    # container, so this example's raising stub must replace any cached one.
    interceptor_scoped_credentials.reset_sts_client()

    # Fails closed: the error PROPAGATES (no credential is returned).
    with pytest.raises((ClientError, BotoCoreError)) as excinfo:
        vend_scoped_credentials(role_arn, vend_scope, table_arn, actions)

    # The propagated exception discloses no scope/role/table detail — the vend
    # code adds nothing beyond the botocore default.
    disclosed = f"{excinfo.value}\n{excinfo.value.args!r}"
    for marker in (role_arn, vend_scope, table_arn):
        assert marker not in disclosed
