"""RESPONSE interceptor properties and unit tests.

Everything that exercises the RESPONSE interceptor's *general* behaviour lives
here: two Hypothesis properties and three example-based unit tests. The
regression-specific suites stay separate —
``tests/test_response_interceptor_fail_closed.py`` (the exception path withholds
rather than passes through) and ``tests/test_response_interceptor_scrub_gaps.py``
(credential shapes the scrubber must catch).

Five concerns are covered, in this order:

* **Deep-equal pass-through of protocol messages.** Representative ``initialize``
  and ``tools/list`` reply bodies carry no credential-shaped key and no
  credential-shaped value, so the credential removal is a structural no-op and the
  handler must return them byte-for-byte, without mutating the input.

* **Count-only finding log.** When the handler scrubs a body carrying ``N``
  credential-shaped removals, it logs ONLY the integer count via
  ``response_scrub removed=<N>`` — never a removed key name, a removed value, or
  any body content. The scrubber's counting semantics are one removal per
  credential-key-name key and one per access-key-id-shaped string value/element,
  with no double counting.

* **No body/key/value logged on the EXCEPTION path either.** A malformed ``mcp``
  structure whose read raises before the scrub runs (``mcp.gatewayResponse`` is a
  list, so the defensive ``.get("body", ...)`` raises ``AttributeError``) must
  still yield a valid ``interceptorOutputVersion: "1.0"`` envelope and log only the
  detail-free ``response_scrub error=1 stage=envelope withheld=0 removed=0``.

  Note the split this covers only half of: the ENVELOPE-read failure tested here is
  the fail-OPEN half of the ``except`` branch, and it is safe only because the read
  failed before any body was held, so the empty default is what gets echoed. The
  fail-CLOSED half — the scrub itself raising, where echoing a body it could not
  vouch for would be a credential bypass — lives in
  ``tests/test_response_interceptor_fail_closed.py``.

* **Property: all credential material is stripped.** For ANY JSON-shaped body with
  credential-shaped material injected at arbitrary depth, the pure scrubber removes
  every credential key name and every access-key-id-shaped string value/element,
  preserves everything else, and never mutates its input.

* **Property: no-op on credential-free bodies, and never errors on any shape.** For
  any credential-free body and any envelope shape — including missing ``body``,
  missing ``gatewayResponse``, missing ``mcp``, and an ``mcp`` /
  ``gatewayResponse`` that is not a dict — the handler returns a well-formed
  envelope, never raises, returns a present body deep-equal, and behaves
  identically with or without ``mcp.gatewayRequest``.

Import resolution (disambiguation note): ``from handler import handler`` resolves
to ``response_interceptor/handler.py``. The root ``conftest.py`` prepends
``response_interceptor/`` to ``sys.path`` (the zip Lambda's archive root, handler
``handler.handler``), and there is NO top-level ``handler.py`` at the repository
root or under ``tools/`` (the tool handlers live at ``tools/<tool>/handler.py`` and
import as ``<tool>.handler``, e.g. ``read_document.handler``). The module-level
guard below asserts the imported module really is the RESPONSE interceptor, so a
future top-level ``handler.py`` can never silently shadow this import. The scrubber
is imported flat (``from credential_scrubber import scrub``) exactly as the zip
Lambda imports it at its archive root.

RESPONSE interceptor I/O (input at ``mcp.gatewayResponse.body``, output at
``mcp.transformedGatewayResponse``) per the AWS documentation:
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-types.html

How the count-only logging is captured/asserted: the handler logs via the root
logger (``logging.getLogger()``) at ``INFO``. The tests use pytest's ``caplog``
fixture under ``caplog.at_level(logging.INFO)`` to capture the emitted
``LogRecord``s, then assert on both ``caplog.records`` (exactly one
``response_scrub`` record whose message is a bare count) and ``caplog.text``
(contains none of the forbidden name/value/body substrings).
"""

from __future__ import annotations

import copy
import logging
import re
import string
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

# Both resolve to response_interceptor/ — see the module docstring's
# import-resolution note.
import handler as response_interceptor_module
from credential_scrubber import scrub
from handler import handler

# ---------------------------------------------------------------------------
# Disambiguation guard: fail loudly if `import handler` ever resolves to anything
# other than the RESPONSE interceptor (response_interceptor/handler.py).
# ---------------------------------------------------------------------------
assert response_interceptor_module.__file__ is not None
assert response_interceptor_module.__file__.replace("\\", "/").endswith(
    "response_interceptor/handler.py"
), (
    "expected the RESPONSE interceptor handler at response_interceptor/handler.py, "
    f"got {response_interceptor_module.__file__!r}"
)


# ---------------------------------------------------------------------------
# The credential surface
#
# Deliberately SPELLED OUT here rather than imported from
# ``credential_scrubber``: these tests must assert against a stated contract, not
# against the module's own constants — importing them would let a wrong edit to the
# scrubber's surface pass every test it breaks. This is the single declaration in
# the file; every concern below reads it.
# ---------------------------------------------------------------------------

#: Rule 1 — the five authoritative credential key names (exact, case-sensitive).
CREDENTIAL_KEY_NAMES: frozenset[str] = frozenset(
    {
        "context",
        "tenant_credentials",
        "access_key_id",
        "secret_access_key",
        "session_token",
    }
)
_CREDENTIAL_KEY_LIST = sorted(CREDENTIAL_KEY_NAMES)

#: Rule 2 — the anchored access-key-id value shape (defense-in-depth heuristic).
ACCESS_KEY_ID_RE = re.compile(r"^(ASIA|AKIA)[A-Z0-9]{16}$")


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: The interceptor output envelope version the handler must always emit.
_OUTPUT_VERSION = "1.0"

#: Status code echoed back when the input carries none (handler default).
_DEFAULT_STATUS_CODE = 200

#: The exact count-only log message the handler emits (format string
#: ``"response_scrub removed=%d"``); no key name or value ever appears in it.
_LOG_PREFIX = "response_scrub"
_LOG_PATTERN = re.compile(r"^response_scrub removed=(\d+)$")

#: The exception-path log line (format string
#: ``"response_scrub error=1 stage=<stage> withheld=%d removed=%d"``). ``stage``
#: names which exception path ran — ``envelope`` (read failed, nothing held),
#: ``scrub`` (the scrub raised, body withheld) or ``post_scrub`` (the scrub
#: succeeded, a later step failed). ``withheld=1`` means the body was suppressed.
#: Still carries no key and no value.
_ERROR_LOG_PATTERN = re.compile(
    r"^response_scrub error=1 stage=(envelope|scrub|post_scrub) withheld=([01]) removed=(\d+)$"
)

#: A canonical, correctly shaped access-key-id string (``ASIA`` + 16 upper-alnum,
#: 20 chars total) that :data:`ACCESS_KEY_ID_RE` matches exactly. Placed under a
#: NON-credential key so the value-shape rule is exercised independently of the
#: key-name rule.
_ACCESS_KEY_ID_SHAPED = "ASIAIOSFODNN7EXAMPLE"

#: Plain (non-access-key-shaped) credential values carried under credential KEY
#: names so those removals are attributable to the key-name rule and the values are
#: distinctive enough to detect if they ever leaked into a log.
_LEAKED_KEY_MATERIAL = "leaked-key-material-xyz"
_LEAKED_TOKEN_MATERIAL = "leaked-token-material-xyz"
_LEAKED_SECRET_MATERIAL = "leaked-secret-material-xyz"


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------


def _response_event(body: Any, *, status_code: int = _DEFAULT_STATUS_CODE) -> dict[str, Any]:
    """Build the RESPONSE interceptor input envelope carrying ``body``.

    Mirrors the gateway's RESPONSE interceptor input contract: the tool reply sits
    at ``mcp.gatewayResponse.body`` alongside ``mcp.gatewayResponse.statusCode``.

    Args:
        body: The reply body the interceptor scrubs.
        status_code: The gateway response status code to carry.

    Returns:
        A well-formed interceptor input event.
    """
    return {"mcp": {"gatewayResponse": {"statusCode": status_code, "body": body}}}


def _transformed_body(result: dict[str, Any]) -> Any:
    """Extract ``mcp.transformedGatewayResponse.body`` from a handler result.

    Args:
        result: The value returned by :func:`handler`.

    Returns:
        The scrubbed/echoed body the handler placed in its output envelope.
    """
    return result["mcp"]["transformedGatewayResponse"]["body"]


def _assert_valid_envelope(result: Any) -> None:
    """Assert ``result`` is a well-formed ``interceptorOutputVersion: "1.0"`` envelope.

    Checks the types too, not just the values, because the no-op property drives
    degenerate and malformed inputs through the handler and a non-dict result would
    otherwise surface as a confusing ``TypeError`` inside the assertion.

    Args:
        result: The value returned by :func:`handler`.
    """
    assert isinstance(result, dict)
    assert result.get("interceptorOutputVersion") == _OUTPUT_VERSION
    mcp = result.get("mcp")
    assert isinstance(mcp, dict)
    transformed = mcp.get("transformedGatewayResponse")
    assert isinstance(transformed, dict)
    assert set(transformed.keys()) == {"statusCode", "body"}


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------


def _scrub_log_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the ``response_scrub`` log records captured so far.

    Args:
        caplog: pytest log-capture fixture.

    Returns:
        Every captured record whose message begins with the count-only prefix.
    """
    return [
        record
        for record in caplog.records
        if record.getMessage().startswith(_LOG_PREFIX)
    ]


def _assert_count_only_log(caplog: pytest.LogCaptureFixture, expected_count: int) -> None:
    """Assert exactly one bare-integer ``response_scrub removed=<N>`` line was logged.

    Confirms the finding log is count-only: exactly one ``response_scrub`` record,
    its message matches ``^response_scrub removed=\\d+$`` (a bare integer, nothing
    else), and that integer equals ``expected_count``.

    Args:
        caplog: pytest log-capture fixture.
        expected_count: The number of removals the handler should have reported.
    """
    records = _scrub_log_records(caplog)
    assert len(records) == 1, f"expected one finding log, got {len(records)}"
    message = records[0].getMessage()
    match = _LOG_PATTERN.match(message)
    assert match is not None, f"log line is not a bare count: {message!r}"
    assert int(match.group(1)) == expected_count


def _assert_error_log(
    caplog: pytest.LogCaptureFixture, *, stage: str, withheld: int, removed: int
) -> None:
    """Assert exactly one detail-free exception-path ``response_scrub`` line was logged.

    The exception path must stay as leak-free as the success path while still being
    TELLABLE APART from it: a scrub that failed must not look like a body that was
    genuinely clean. So the line carries ``error=1``, the stage that failed, and a
    ``withheld`` flag — and nothing else.

    Args:
        caplog: pytest log-capture fixture.
        stage: Which exception path ran — ``envelope``, ``scrub`` or ``post_scrub``.
        withheld: ``1`` when the handler suppressed the body (the scrub raised),
            ``0`` when there was no unvouched-for body to suppress.
        removed: The removal count the line must report (always ``0`` here — an
            aborted scrub removed nothing).
    """
    records = _scrub_log_records(caplog)
    assert len(records) == 1, f"expected one finding log, got {len(records)}"
    message = records[0].getMessage()
    match = _ERROR_LOG_PATTERN.match(message)
    assert match is not None, f"log line is not a detail-free error line: {message!r}"
    assert match.group(1) == stage
    assert int(match.group(2)) == withheld
    assert int(match.group(3)) == removed


def _assert_no_leak(caplog: pytest.LogCaptureFixture, forbidden: list[str]) -> None:
    """Assert none of ``forbidden`` (names/values/body content) appears in the logs.

    Checks both the aggregate ``caplog.text`` and each individual record message so
    that no credential key name, credential value, or body content is echoed on any
    path.

    Args:
        caplog: pytest log-capture fixture.
        forbidden: Substrings that MUST NOT appear anywhere in the captured logs.
    """
    text = caplog.text
    for needle in forbidden:
        assert needle not in text, f"log leaked forbidden content: {needle!r}"
    for record in caplog.records:
        message = record.getMessage()
        for needle in forbidden:
            assert needle not in message, f"record leaked forbidden content: {needle!r}"


# ---------------------------------------------------------------------------
# Representative protocol messages (credential-free by inspection). These are the
# exact non-tools/call bodies the interceptor must pass through untouched, shared
# by the deep-equal unit test and the no-op property.
# ---------------------------------------------------------------------------


def _initialize_body() -> dict[str, Any]:
    """A representative JSON-RPC ``initialize`` reply body (no credential material)."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": True}, "logging": {}},
            "serverInfo": {"name": "support-tools-gateway", "version": "1.0.0"},
        },
    }


def _tools_list_body() -> dict[str, Any]:
    """A representative JSON-RPC ``tools/list`` reply body (no credential material)."""
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "tools": [
                {
                    "name": "ReadDocument___read_document",
                    "description": "Read a single document by id.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"doc_id": {"type": "string"}},
                        "required": ["doc_id"],
                    },
                },
                {
                    "name": "SearchDocuments___search_documents",
                    "description": "Search documents in the served scope.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ]
        },
    }


#: The protocol messages the no-op property samples from: the two above plus a
#: notification (no ``id``) and a ping result (empty ``result``).
_MCP_PROTOCOL_MESSAGES: list[dict[str, Any]] = [
    _initialize_body(),
    _tools_list_body(),
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 3, "result": {}},
]


# ---------------------------------------------------------------------------
# Key / value strategies
#
# Both properties need lowercase-only dict keys that can never spell a credential
# name, so they share one factory. The FILTER is the guarantee — not the length
# bound, which only shapes the generated space (``context`` is the one credential
# name spellable from a-z alone).
# ---------------------------------------------------------------------------


def _noncredential_keys(max_size: int) -> st.SearchStrategy[str]:
    """Dict keys that are provably never a credential key name.

    Args:
        max_size: Maximum key length to generate.

    Returns:
        A strategy for lowercase ASCII keys, filtered against
        :data:`CREDENTIAL_KEY_NAMES`.
    """
    return st.text(
        alphabet=string.ascii_lowercase, min_size=1, max_size=max_size
    ).filter(lambda key: key not in CREDENTIAL_KEY_NAMES)


# --- strip property: keys/values, plus injected credential material ---

#: Non-credential dict keys for the strip property (length 1-6).
_NONCRED_KEY = _noncredential_keys(max_size=6)

#: Non-credential string values: lowercase/digit/space only, so they can NEVER
#: match the uppercase-anchored access-key-id regex.
_NONCRED_STR = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", max_size=12)

#: Non-credential leaf scalars (all provably non-credential-shaped).
_NONCRED_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    _NONCRED_STR,
)

#: A VALID 20-char access-key-id-shaped string: ASIA (STS temp) / AKIA (long-term)
#: prefix + 16 uppercase-alnum chars. Every value fully matches ACCESS_KEY_ID_RE.
_ACCESS_KEY_STR = st.builds(
    lambda prefix, tail: prefix + tail,
    st.sampled_from(["ASIA", "AKIA"]),
    st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=16, max_size=16),
)

#: Arbitrary value parked UNDER a credential key name. Its content is irrelevant to
#: the output: the scrubber deletes a credential key without recursing into it, so
#: the whole subtree disappears regardless of what it holds.
_CRED_KEY_VALUE = st.one_of(
    _ACCESS_KEY_STR,
    _NONCRED_SCALAR,
    st.dictionaries(_NONCRED_KEY, _NONCRED_SCALAR, max_size=2),
    st.lists(_ACCESS_KEY_STR, max_size=2),
)


# --- no-op property: credential-free keys/values and envelope shapes ---

#: Dict keys for the no-op property (length 1-10).
_NOOP_KEY = _noncredential_keys(max_size=10)

#: String values: lowercase letters + digits + a few safe punctuation, NO uppercase
#: A-Z — so a value can never match the access-key-id shape.
_VALUE_ALPHABET = string.ascii_lowercase + string.digits + " ._-/:"

#: Envelope shapes exercised — a full body plus five degenerate/malformed shapes.
_ENVELOPE_KINDS = (
    "full",
    "missing_body",
    "missing_gateway_response",
    "missing_mcp",
    "mcp_not_dict",
    "gateway_response_not_dict",
)


def _value_text() -> st.SearchStrategy[str]:
    """Credential-free string values (no uppercase, so no access-key-id shape)."""
    return st.text(alphabet=_VALUE_ALPHABET, max_size=40)


def _json_bodies() -> st.SearchStrategy[Any]:
    """Arbitrary credential-free JSON values, or a representative MCP message.

    Leaves cover the degenerate scalar shapes the property calls out (a body that is
    a string, number, bool, or ``None``); the recursive step builds nested
    objects/arrays; and the union folds in the fixed MCP protocol messages.
    """
    leaves = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        _value_text(),
    )
    nested = st.recursive(
        leaves,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(keys=_NOOP_KEY, values=children, max_size=4),
        ),
        max_leaves=15,
    )
    return st.one_of(nested, st.sampled_from(_MCP_PROTOCOL_MESSAGES))


def _gateway_requests() -> st.SearchStrategy[dict[str, Any]]:
    """Varied ``mcp.gatewayRequest`` objects to exercise the optional method gate.

    Covers ``tools/call`` (scan runs), non-``tools/call`` (scan skipped), absent /
    non-string method, a raw-string request body (the ``AttributeError`` gate
    branch), and an empty request — none may change the result for a credential-free
    body.
    """
    return st.sampled_from(
        [
            {"body": {"method": "tools/call"}},
            {"body": {"method": "initialize"}},
            {"body": {"method": "tools/list"}},
            {"body": {"method": "ping"}},
            {"body": {}},  # method absent
            {"body": {"method": 123}},  # non-string method
            {"body": "raw-string-request-body"},  # body not a dict
            {},  # empty gatewayRequest
        ]
    )


# ---------------------------------------------------------------------------
# (dirty, clean) generators for the strip property — clean is the exact expected
# scrub output, built constructively (no deletion logic) so the equality assertion
# is not circular.
# ---------------------------------------------------------------------------


def _gen_node(draw: st.DrawFn, depth: int) -> tuple[Any, Any]:
    """Return a ``(dirty, clean)`` pair for one node at the given depth budget."""
    if depth <= 0:
        value = draw(_NONCRED_SCALAR)
        return value, value
    # Bias toward scalars so structures stay bounded but still nest.
    kind = draw(st.sampled_from(("scalar", "scalar", "dict", "list")))
    if kind == "scalar":
        value = draw(_NONCRED_SCALAR)
        return value, value
    if kind == "dict":
        return _gen_dict(draw, depth)
    return _gen_list(draw, depth)


def _gen_dict(draw: st.DrawFn, depth: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a dict with preserved base entries plus injected credential material.

    Base (non-credential) entries go into BOTH dirty and clean. Injected credential
    material goes into dirty only:

    * credential-named keys — the scrubber deletes the key and its whole subtree, so
      they are absent from clean;
    * access-key-id-shaped strings under FRESH non-credential keys — the scrubber
      deletes the key/value pair, so they are absent from clean. Placing them under
      non-credential keys makes the value-shape branch fire independently of the
      key-name rule.
    """
    dirty: dict[str, Any] = {}
    clean: dict[str, Any] = {}

    for key in draw(st.lists(_NONCRED_KEY, unique=True, max_size=4)):
        dirty_child, clean_child = _gen_node(draw, depth - 1)
        dirty[key] = dirty_child
        clean[key] = clean_child

    # Rule 1: credential-named keys — removed with their whole subtree.
    for cred_key in draw(
        st.lists(st.sampled_from(_CREDENTIAL_KEY_LIST), unique=True, max_size=3)
    ):
        dirty[cred_key] = draw(_CRED_KEY_VALUE)  # absent from clean

    # Rule 2: access-key-id-shaped VALUES under NON-credential keys.
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        inj_key = draw(_NONCRED_KEY)
        while inj_key in dirty:  # keep it distinct from every existing key
            inj_key += "x"
        dirty[inj_key] = draw(_ACCESS_KEY_STR)  # absent from clean

    return dirty, clean


def _gen_list(draw: st.DrawFn, depth: int) -> tuple[list[Any], list[Any]]:
    """Build a list of preserved nodes interleaved with injected access-key strings.

    A preserved element is appended to BOTH dirty and clean at the same relative
    position; an injected access-key-id-shaped string element is appended to dirty
    only. The scrubber drops those string elements while keeping the order of
    survivors, which the clean list mirrors exactly.
    """
    dirty: list[Any] = []
    clean: list[Any] = []

    for _ in range(draw(st.integers(min_value=0, max_value=4))):
        if draw(st.integers(min_value=0, max_value=3)) == 0:
            # Injected access-key-id-shaped element — removed, so dirty only.
            dirty.append(draw(_ACCESS_KEY_STR))
        else:
            dirty_child, clean_child = _gen_node(draw, depth - 1)
            dirty.append(dirty_child)
            clean.append(clean_child)

    return dirty, clean


@st.composite
def _credential_bearing_bodies(draw: st.DrawFn) -> tuple[dict[str, Any], dict[str, Any]]:
    """A JSON-object response body plus its exact expected scrubbed form."""
    return _gen_dict(draw, depth=3)


# ---------------------------------------------------------------------------
# Recursive walkers — collect every dict key and every string VALUE/element at any
# depth, so absence can be asserted directly.
# ---------------------------------------------------------------------------


def _iter_keys(value: Any) -> Iterator[str]:
    """Yield every dict key reachable at any depth."""
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str):
                yield key
            yield from _iter_keys(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_keys(item)


def _iter_string_values(value: Any) -> Iterator[str]:
    """Yield every string reachable as a dict VALUE or list element at any depth.

    Dict keys are intentionally NOT traversed here: the value-shape rule applies to
    values and list elements, so this walker mirrors exactly what the absence check
    must cover.
    """
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_string_values(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_string_values(item)
    elif isinstance(value, str):
        yield value


def _is_credential_free(value: Any) -> bool:
    """Return whether ``value`` has no credential-shaped key or value at any depth.

    Mirrors the two removal rules so the no-op property's deep-equal assertion is
    meaningful: no dict key is an exact credential name, and no string matches the
    anchored access-key-id pattern.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            if key in CREDENTIAL_KEY_NAMES:
                return False
            if not _is_credential_free(child):
                return False
        return True
    if isinstance(value, list):
        return all(_is_credential_free(item) for item in value)
    if isinstance(value, str):
        return ACCESS_KEY_ID_RE.match(value) is None
    return True


def _build_input(
    kind: str,
    body: Any,
    status_code: int,
    gateway_request: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a (possibly degenerate/malformed) interceptor input envelope of ``kind``.

    ``gateway_request``, when not ``None`` and the envelope's ``mcp`` is a dict, is
    attached at ``mcp.gatewayRequest`` to prove the handler ignores it. ``body`` is
    used only by the ``full`` kind.
    """
    if kind == "missing_mcp":
        env: dict[str, Any] = {}
    elif kind == "mcp_not_dict":
        env = {"mcp": "mcp-is-not-a-dict"}
    elif kind == "missing_gateway_response":
        env = {"mcp": {}}
    elif kind == "gateway_response_not_dict":
        env = {"mcp": {"gatewayResponse": "gateway-response-is-not-a-dict"}}
    elif kind == "missing_body":
        env = {"mcp": {"gatewayResponse": {"statusCode": status_code}}}
    else:  # "full"
        env = _response_event(body, status_code=status_code)

    # The optional method gate reads mcp.gatewayRequest only if present; attach it
    # only when mcp is a dict (a non-dict/absent mcp cannot carry the sub-object).
    if gateway_request is not None and isinstance(env.get("mcp"), dict):
        env["mcp"]["gatewayRequest"] = gateway_request
    return env


# ---------------------------------------------------------------------------
# Representative protocol messages round-trip deep-equal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body_factory",
    [_initialize_body, _tools_list_body],
    ids=["initialize", "tools/list"],
)
def test_credential_free_body_returned_deep_equal(body_factory: Any) -> None:
    """``initialize`` / ``tools/list`` bodies are returned deep-equal (no-op scrub).

    Neither body carries a credential-shaped key or value, so the removal is a
    structural no-op and the returned ``mcp.transformedGatewayResponse.body`` must
    equal the input ``mcp.gatewayResponse.body`` exactly, with the original status
    echoed and the input body left unmutated.
    """
    body = body_factory()
    snapshot = copy.deepcopy(body)
    event = _response_event(body, status_code=_DEFAULT_STATUS_CODE)

    result = handler(event, None)

    _assert_valid_envelope(result)
    # Deep-equal pass-through of the body.
    assert _transformed_body(result) == snapshot
    # Original status code echoed unchanged.
    assert result["mcp"]["transformedGatewayResponse"]["statusCode"] == _DEFAULT_STATUS_CODE
    # The input body object was not mutated.
    assert body == snapshot


# ---------------------------------------------------------------------------
# Success path — count-only finding log, no leaked content
# ---------------------------------------------------------------------------


def test_finding_log_is_count_only_on_success_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A body with N credential removals logs ONLY the integer count N.

    The body carries three removals: two credential-KEY-NAME keys —
    ``session_token`` and a nested ``access_key_id`` — plus one
    access-key-id-SHAPED string under the non-credential key ``note``. That is
    ``N = 3`` (no double counting). The handler must:

    * remove all three from the returned body, and
    * log exactly ``response_scrub removed=3`` — a bare integer, with no removed
      key name, no removed value, and no surrounding body content.
    """
    body = {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {
            "content": [
                {"type": "text", "text": "Document body text"},
                {"access_key_id": _LEAKED_KEY_MATERIAL},  # key-name removal (1)
                {"note": _ACCESS_KEY_ID_SHAPED},  # value-shape removal (1)
            ],
            "session_token": _LEAKED_TOKEN_MATERIAL,  # key-name removal (1)
            "safe_field": "keep me",
        },
    }
    expected_count = 3
    event = _response_event(body)

    with caplog.at_level(logging.INFO):
        result = handler(event, None)

    _assert_valid_envelope(result)

    # Sanity: the returned body actually dropped every credential-shaped item and
    # kept the benign ones (this is what makes the count N == 3 meaningful).
    transformed = _transformed_body(result)
    assert "session_token" not in transformed["result"]
    assert transformed["result"]["safe_field"] == "keep me"
    kept_list = transformed["result"]["content"]
    assert {"type": "text", "text": "Document body text"} in kept_list
    assert all("access_key_id" not in item for item in kept_list)
    assert all(_ACCESS_KEY_ID_SHAPED not in item.values() for item in kept_list)

    # The finding log is count-only: exactly one bare-integer line equal to N.
    _assert_count_only_log(caplog, expected_count)

    # No removed key name, no removed value, and no body content leaked.
    _assert_no_leak(
        caplog,
        forbidden=[
            # Credential key names.
            "access_key_id",
            "secret_access_key",
            "session_token",
            "tenant_credentials",
            "context",
            # Credential / removed values.
            _LEAKED_KEY_MATERIAL,
            _LEAKED_TOKEN_MATERIAL,
            _ACCESS_KEY_ID_SHAPED,
            # Surrounding body content.
            "Document body text",
            "safe_field",
            "keep me",
            "jsonrpc",
        ],
    )


# ---------------------------------------------------------------------------
# Exception path — no raise, valid envelope, count-only log, no leak
# ---------------------------------------------------------------------------


def test_exception_path_returns_envelope_and_logs_count_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed ``mcp`` shape trips the internal read yet never leaks or raises.

    ``mcp.gatewayResponse`` is a list, so the handler's defensive
    ``gateway_response.get("body", {})`` raises ``AttributeError`` before the scrub
    runs, exercising the ENVELOPE-read half of the ``except`` branch. Because the
    read failed before ``original_body`` was ever assigned, there is no body to
    withhold and the handler passes its empty default through — this is the one
    exception path that stays fail-OPEN, and it is safe precisely because it holds
    nothing. (The fail-CLOSED half, where the scrub itself raises, is covered in
    ``tests/test_response_interceptor_fail_closed.py``.)

    The list embeds credential-shaped strings so the test can prove the exception
    handler echoes none of them. The handler must (a) return a valid
    ``interceptorOutputVersion: "1.0"`` envelope without raising and (b) emit only
    the detail-free ``response_scrub error=1 stage=envelope withheld=0 removed=0``
    line — no body, key, or value.
    """
    # gatewayResponse is a list (not a dict): `.get("body", ...)` raises before scrub.
    event = {
        "mcp": {
            "gatewayResponse": [
                "access_key_id",
                _ACCESS_KEY_ID_SHAPED,
                {"secret_access_key": _LEAKED_SECRET_MATERIAL},
            ]
        }
    }

    with caplog.at_level(logging.INFO):
        # Must not raise out to the gateway (the handler's robustness contract).
        result = handler(event, None)

    # (a) A valid, well-formed output envelope is still returned.
    _assert_valid_envelope(result)
    assert result["mcp"]["transformedGatewayResponse"]["statusCode"] == _DEFAULT_STATUS_CODE

    # Nothing was read, so the empty default is echoed — not the malformed input.
    assert _transformed_body(result) == {}

    # The error line is present, detail-free, and marks the body as NOT withheld
    # (there was none to withhold).
    _assert_error_log(caplog, stage="envelope", withheld=0, removed=0)

    # (b) No body / key / value logged on the exception path.
    _assert_no_leak(
        caplog,
        forbidden=[
            "access_key_id",
            "secret_access_key",
            "session_token",
            "tenant_credentials",
            "context",
            _ACCESS_KEY_ID_SHAPED,
            _LEAKED_SECRET_MATERIAL,
            "gatewayResponse",
        ],
    )


# ---------------------------------------------------------------------------
# Property — all credential material is stripped, everything else preserved
#
# A concrete @example pins the value-shape branch: access-key-id-shaped strings
# under a non-credential key, as a list element, and nested inside a list-element
# dict — none of them under a credential key name.
# ---------------------------------------------------------------------------

_EXAMPLE_DIRTY: dict[str, Any] = {
    "note": "ASIAIOSFODNN7EXAMPLE",  # value-shape under a NON-credential key
    "items": [
        "keep",  # ordinary element, preserved
        "AKIAABCDEFGHIJKLMNOP",  # value-shape as a list element, removed
        {"inner_note": "ASIA1234567890123456"},  # value-shape nested, removed
    ],
    "context": {  # credential key name -> whole subtree removed
        "tenant_credentials": {"access_key_id": "AKIA0000000000000000"}
    },
    "keepme": "hello",
    "count": 7,
}
_EXAMPLE_CLEAN: dict[str, Any] = {
    "items": ["keep", {}],  # AKIA element dropped; inner dict emptied but kept
    "keepme": "hello",
    "count": 7,
}


@settings(max_examples=100, deadline=None)
@given(node=_credential_bearing_bodies())
@example(node=(_EXAMPLE_DIRTY, _EXAMPLE_CLEAN))
def test_response_interceptor_strips_all_credential_material(
    node: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The RESPONSE interceptor scrubber removes all credential-shaped material at
    any depth, preserves everything else, and never mutates its input.

    Model-based: the generator builds ``(dirty, clean)`` in lock-step —
    non-credential content is placed identically into both, while injected
    credential material goes into ``dirty`` only. ``clean`` is therefore the exact
    expected output, constructed WITHOUT applying any deletion logic, so the
    equality assertion is not circular.
    """
    dirty, expected_clean = node
    snapshot = copy.deepcopy(dirty)

    scrubbed, removed_count = scrub(dirty)

    # Removal AND preservation in one shot: `expected_clean` was built with all
    # credential material omitted and every non-credential piece retained, so exact
    # equality proves both halves of the property. Deletion (not redaction) is
    # implied: removed keys are simply absent from `expected_clean` rather than
    # carrying a masked placeholder.
    assert scrubbed == expected_clean

    # No credential key name survives anywhere at any depth.
    surviving_keys = set(_iter_keys(scrubbed))
    assert CREDENTIAL_KEY_NAMES.isdisjoint(surviving_keys)

    # No access-key-id-shaped string survives as a value/element at any depth,
    # regardless of the enclosing key.
    for text in _iter_string_values(scrubbed):
        assert ACCESS_KEY_ID_RE.match(text) is None

    # The input is never mutated (the scrub works on a deep copy).
    assert dirty == snapshot

    # The count is a non-negative integer (count-only contract; no names/values).
    assert isinstance(removed_count, int)
    assert removed_count >= 0


# ---------------------------------------------------------------------------
# Property — no-op on credential-free bodies, never errors on any shape
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    body=_json_bodies(),
    status_code=st.integers(min_value=100, max_value=599),
    envelope_kind=st.sampled_from(_ENVELOPE_KINDS),
    gateway_request=_gateway_requests(),
)
def test_response_interceptor_noop_and_robust(
    body: Any,
    status_code: int,
    envelope_kind: str,
    gateway_request: dict[str, Any],
) -> None:
    """RESPONSE interceptor is a no-op on credential-free bodies and never errors.

    For all credential-free bodies (including ``initialize`` / ``tools/list`` /
    ``notifications/initialized`` / ``ping`` and degenerate shapes) and every
    envelope kind, the handler returns a well-formed envelope and never raises; a
    present body is returned deep-equal to the input body; and the result is
    identical whether or not ``mcp.gatewayRequest`` is present.
    """
    # Meta-check: the generated body is genuinely credential-free, so the deep-equal
    # no-op below is a real no-op (not the scrubber silently removing material and
    # the test passing for the wrong reason).
    assert _is_credential_free(body)

    env_without = _build_input(envelope_kind, body, status_code, gateway_request=None)
    env_with = _build_input(
        envelope_kind, body, status_code, gateway_request=gateway_request
    )
    snap_without = copy.deepcopy(env_without)
    snap_with = copy.deepcopy(env_with)

    # Reaching past these calls proves the handler never raised on either shape (a
    # full body, or a degenerate/malformed envelope).
    result_without = handler(env_without, None)
    result_with = handler(env_with, None)

    # A well-formed interceptorOutputVersion "1.0" envelope on every shape.
    _assert_valid_envelope(result_without)
    _assert_valid_envelope(result_with)

    # The result is identical with or without mcp.gatewayRequest — the method gate
    # is a pure optimization and never changes the outcome.
    assert result_with == result_without

    # The handler operates on a deep copy, so neither input is mutated — the no-op
    # must not disturb the caller's event.
    assert env_without == snap_without
    assert env_with == snap_with

    # When a body is present, it round-trips DEEP-EQUAL (structural no-op) and the
    # original status code is echoed back unchanged.
    if envelope_kind == "full":
        transformed = result_without["mcp"]["transformedGatewayResponse"]
        assert transformed["body"] == body
        assert transformed["statusCode"] == status_code
