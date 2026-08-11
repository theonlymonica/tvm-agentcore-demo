"""
RESPONSE interceptor Lambda entry point — credential-shaped material scrubber.

This is the zip-packaged RESPONSE interceptor for multi-tenant data isolation. It
wraps the pure recursive scrubber in
``response_interceptor/credential_scrubber.py`` with the gateway envelope
handling: it reads the tool reply from the gateway response, runs the scrub, and
returns the scrubbed reply in a ``transformedGatewayResponse`` envelope. It makes
NO allow/block decision and NEVER rejects, errors, or blocks a response on the
basis of its method, shape, or content — the load-bearing regression guard
against a prior interceptor that classified every message as unclassifiable and
blocked ``initialize``, breaking the MCP handshake.

Flat-import / packaging contract:
    The RESPONSE interceptor is a SEPARATE zip Lambda, distinct from the
    container-image REQUEST interceptor in ``interceptor/``. Its CDK handler is
    ``handler.handler`` and ``lambda_.Code.from_asset("response_interceptor")``
    zips this directory's contents at the archive ROOT, so ``handler.py`` and
    ``credential_scrubber.py`` sit side by side as top-level modules at runtime.
    The scrubber is therefore imported FLAT — ``from credential_scrubber import
    scrub`` — not as a package submodule.

RESPONSE interceptor I/O contract — see
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-types.html:
    Input payload (``interceptorInputVersion: "1.0"``) carries, under ``mcp``:
      * ``rawGatewayRequest.body`` — the raw request body string;
      * ``gatewayRequest`` — ``path`` / ``httpMethod`` / ``headers`` (headers
        present only when ``passRequestHeaders=true``) / ``body`` (the parsed
        JSON-RPC request, whose ``method`` is available regardless of
        ``passRequestHeaders``);
      * ``gatewayResponse`` — ``statusCode`` / ``headers`` / ``body`` (the tool
        reply this interceptor scrubs).
    Output payload (``interceptorOutputVersion: "1.0"``) returns
    ``mcp.transformedGatewayResponse`` as ``{statusCode, body}``; when that field
    is present the gateway responds with it immediately. This handler reads
    ``mcp.gatewayResponse.body`` + ``mcp.gatewayResponse.statusCode`` (defaulting
    the status to 200 when absent) and writes the scrubbed body back at
    ``mcp.transformedGatewayResponse``.

Non-``tools/call`` pass-through:
    The credential removal is a STRUCTURAL NO-OP on responses that carry no
    credential-shaped key or value — ``initialize``, ``tools/list``,
    ``notifications/initialized``, and ``ping`` replies (this proven pass-through
    method set is confirmed by the retained comments in
    ``interceptor/handler.py``). Such bodies round-trip deep-equal to the input.
    An OPTIONAL method-gate optimization (:func:`_should_skip_scan`) reads
    ``mcp.gatewayRequest.body.method`` ONLY IF present and skips the scan for
    non-``tools/call`` methods; correctness NEVER depends on ``mcp.gatewayRequest``
    being present, and the default behaviour is to run the no-op-safe scrub on
    EVERY response body.

Robustness — never raise, but fail CLOSED once scrubbing starts:
    Every read uses a defensive ``.get()`` accessor and the whole body traversal is
    wrapped, so the handler never raises out to the gateway. What it returns on the
    exception path depends on how far it got, because the three failures are not
    equally safe:

      * a failure while reading the ENVELOPE (before the scrub) passes through the
        default empty body — nothing was read, so nothing can leak;
      * a failure INSIDE the scrub (:class:`credential_scrubber.ScrubError`, e.g. an
        over-deep body, or any unforeseen defect) means the reply is
        unvouched-for. The handler WITHHOLDS it, returning a JSON-RPC error body
        instead of the original. Echoing the original was a scrubber bypass: the
        only credential control on the response path could be defeated by making it
        throw, and the count-only ``removed=0`` log made the bypass
        indistinguishable from a genuinely clean reply. The exception path now logs
        ``error=1`` with the stage that failed;
      * a failure AFTER the body was cleared for return — the scrub completed, or the
        method gate passed the reply through — returns that cleared body. It is safe
        and already computed, so neither withholding it nor falling back to the
        unscrubbed original would be right.

    Withholding cannot break the MCP handshake: ``initialize`` / ``tools/list`` /
    ``notifications/initialized`` / ``ping`` replies exit earlier through the method
    gate, and are small flat structures with nothing for the scrubber to trip on. The
    no-blocking contract forbids blocking a message because of its method, shape or
    content — not reporting an error when the interceptor's own processing failed.

Logging:
    COUNT-ONLY. The handler logs at most the integer number of removed
    keys/values and NEVER logs the response body, the tool reply, any key, any
    value, or any removed name/value — on any path, including the exception path.

Functions:
    handler: Lambda entry point — read the tool reply, scrub it, return the
        ``transformedGatewayResponse`` envelope; no-op-safe and non-blocking.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from credential_scrubber import has_credential_shape, scrub

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Required interceptor output envelope version (RESPONSE interceptor contract).
_INTERCEPTOR_OUTPUT_VERSION = "1.0"

#: Status code echoed back when ``mcp.gatewayResponse.statusCode`` is absent.
_DEFAULT_STATUS_CODE = 200

#: JSON-RPC method for tool invocations. Used ONLY by the optional method-gate
#: optimization (:func:`_should_skip_scan`); the default behaviour scrubs every
#: response body regardless of method.
_TOOLS_CALL_METHOD = "tools/call"

#: JSON-RPC error code returned in place of a withheld reply. -32603 is the
#: standard "Internal error" code, which is what an interceptor that could not
#: complete its own processing is reporting.
_WITHHELD_ERROR_CODE = -32603

#: Client-facing message for a withheld reply. Deliberately GENERIC: the caller may
#: be acting on attacker-influenced document content, and naming the scrubber would
#: confirm one sits in the path and let its depth cap be bisected. The specifics go
#: to CloudWatch (``error=1 withheld=1``), not to the client. Fixed and content-free
#: either way — it names no key and no value.
_WITHHELD_ERROR_MESSAGE = "internal error: the response could not be processed"

#: Cap on how many per-entry errors a withheld JSON-RPC BATCH reply is answered
#: with. One error per entry lets a client correlate each id it is waiting on, but a
#: batch of N tiny entries would otherwise inflate into N ~120-byte objects, so the
#: array is TRUNCATED to the first `_MAX_WITHHELD_BATCH_ITEMS` entries. Ids beyond
#: that get no error and those calls fall back to the client's own timeout — the
#: deliberate trade for keeping a withheld reply's size bounded (~8 KB) no matter how
#: large the batch was.
_MAX_WITHHELD_BATCH_ITEMS = 64


class _Unset:
    """Type of the :data:`_UNSET` sentinel."""


#: Sentinel distinguishing "the scrub has not produced a body" from a scrub that
#: legitimately produced ``None`` / ``{}``. A plain ``None`` default could not.
_UNSET = _Unset()

#: Longest JSON-RPC string id echoed back on the withheld path. A correlation id is
#: short; anything longer is not one, and the withheld body is built from a reply the
#: scrubber never vouched for, so what gets copied out of it is kept minimal.
_MAX_ECHOED_ID_CHARS = 128

#: Characters a string id may contain to be echoed back. Deliberately narrow — the
#: shapes real MCP clients use (``req-42``, a UUID, ``call:7``) and nothing that
#: could carry markup, quoting or prose out of an unvouched body.
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """RESPONSE interceptor Lambda entry point — scrub the tool reply.

    Reads the tool reply body from ``mcp.gatewayResponse.body`` and the status
    code from ``mcp.gatewayResponse.statusCode`` (defaulting to 200), runs the
    credential scrubber over the body, and returns the scrubbed reply in an
    ``interceptorOutputVersion: "1.0"`` envelope at
    ``mcp.transformedGatewayResponse``.

    Never rejects, errors, or blocks on the basis of method, shape, or content, and
    never raises out to the gateway: a credential-free body round-trips deep-equal
    to the input, and a malformed envelope yields a no-op pass-through. When the
    SCRUB itself fails, however, the handler fails CLOSED — it withholds the
    unvouched-for reply and returns a JSON-RPC error body (see the module
    docstring's Robustness section). Logging is count-only and never includes the
    body, keys, values, or removed names/values.

    Args:
        event: The RESPONSE interceptor payload from the gateway (see the module
            docstring for the contract).
        context: The Lambda context (unused).

    Returns:
        An ``interceptorOutputVersion: "1.0"`` envelope carrying
        ``mcp.transformedGatewayResponse`` as ``{"statusCode", "body"}``.
    """
    # Defaults so the fail-safe (except) path always has a well-formed body and
    # status to echo back.
    status_code: Any = _DEFAULT_STATUS_CODE
    # `original_body` is published ONLY on the line before the scrub is entered, so
    # "we are past the envelope read" and "we hold a body" are the same fact rather
    # than two statements that happen to be ordered that way.
    original_body: Any = {}
    # Flips to True the instant the scrub is entered. It is the fail-open /
    # fail-closed boundary: before it, a failure means we could not read the
    # ENVELOPE (nothing to leak); after it, a failure means the SCRUBBER could not
    # vouch for a body we do hold, and echoing that body would be a bypass.
    scrub_started = False
    # Set to a body we are willing to return: the SCRUBBED body once the scrub
    # succeeds, or the pass-through body once the method gate has cleared it. If
    # anything after that point fails, this is a body we HAVE vouched for, so the
    # except path returns it — withholding a clean reply would be a needless outage,
    # and echoing `original_body` there would be a leak.
    vouched_body: Any = _UNSET
    try:
        # Defensive `.get()` accessors: a missing/unexpected shape never raises.
        mcp = event.get("mcp", {}) or {}
        gateway_response = mcp.get("gatewayResponse", {}) or {}
        body = gateway_response.get("body", {})
        status_code = gateway_response.get("statusCode", _DEFAULT_STATUS_CODE)

        # Optional method-gate optimization. If — and ONLY IF — the
        # originating request method is present and is not "tools/call", skip the
        # scan and pass the body through untouched. The scan is a structural
        # no-op on such bodies anyway (initialize / tools/list /
        # notifications/initialized / ping), so this is purely a cost saving. The
        # interceptor NEVER depends on mcp.gatewayRequest being present; when the
        # method is unavailable the default is to scrub every response body.
        if _should_skip_scan(mcp):
            # Pass-through by design, so this body is one we are willing to return:
            # publish it before the log/return so a fault in either still yields the
            # reply rather than the empty default. Returning `{}` for an `initialize`
            # reply is exactly the handshake-breaking shape the no-blocking contract
            # exists to prevent, and this is the branch the handshake travels through.
            vouched_body = body
            _log("response_scrub removed=%d", 0)
            return _transformed_response(status_code, body)

        # Default path: run the no-op-safe recursive scrub. A credential-free body
        # round-trips deep-equal to the input; a body carrying credential material
        # has every credential-shaped key and value deleted at any depth on a deep
        # copy, so the input is never mutated.
        original_body = body
        scrub_started = True
        scrubbed_body, removed_count = scrub(original_body)
        vouched_body = scrubbed_body
        # Count-only logging: log at most the integer removal count; never the
        # body, its keys, its values, or removed names/values.
        _log("response_scrub removed=%d", removed_count)
        return _transformed_response(status_code, scrubbed_body)
    except Exception:  # noqa: BLE001 - never reject/error/block
        # Robustness contract: never raise out to the gateway. The
        # RETURNED BODY, though, depends on how far we got — the cases are not
        # equally safe:
        #
        #   * vouched_body set -> return the SCRUBBED body. The scrub completed; only
        #     something after it failed. The reply is safe and already computed.
        #   * scrub_started -> FAIL CLOSED. The scrubber raised (an over-deep body,
        #     or any unforeseen defect), so the reply is unvouched-for and MAY hold
        #     credential material. Echoing it is exactly the bypass this branch
        #     used to be: the only credential control on the response path,
        #     defeated by making it throw. Withhold the body and return a
        #     JSON-RPC error instead. This cannot break the MCP handshake:
        #     initialize / tools/list / ping replies either exit above via the
        #     method gate or are small flat structures with nothing for the
        #     scrubber to trip on, and the no-blocking contract forbids blocking on
        #     a message's method/shape/content — not returning an error when our own
        #     processing failed.
        #   * neither -> pass through. The envelope read failed, so `original_body`
        #     is still the `{}` default; there is nothing to withhold, and echoing
        #     it keeps the no-op contract for a malformed envelope.
        #
        # Either way the log stays count-only and detail-free — but it now carries
        # `error=1`, so a scrub that failed is no longer indistinguishable in
        # CloudWatch from a body that was genuinely clean.
        if vouched_body is not _UNSET:
            _log("response_scrub error=1 stage=post_scrub withheld=0 removed=%d", 0)
            return _transformed_response(status_code, vouched_body)
        if scrub_started:
            _log("response_scrub error=1 stage=scrub withheld=1 removed=%d", 0)
            return _transformed_response(status_code, _withheld_body(original_body))
        _log("response_scrub error=1 stage=envelope withheld=0 removed=%d", 0)
        return _transformed_response(status_code, original_body)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log(message: str, *args: Any) -> None:
    """Emit one count-only log line, swallowing any logging failure.

    The handler's contract is that it never raises out to the gateway, and the
    recovery paths log too — so an unguarded ``logger.info`` in the ``except`` block
    would be the one place a broken log handler could still escape and turn a
    recoverable reply into a gateway 5xx. Logging is observability, never a reason to
    fail a response.

    Args:
        message: A ``%``-style format string. Callers pass ONLY count/flag literals
            — never body content, key names or values.
        *args: Format arguments for ``message``.
    """
    try:
        logger.info(message, *args)
    except Exception:  # noqa: BLE001 - logging must never break the response
        pass


def _should_skip_scan(mcp: dict[str, Any]) -> bool:
    """Return whether the optional method gate lets us skip the credential scan.

    Reads ``mcp.gatewayRequest.body.method`` defensively and returns ``True`` only
    when a string method is present AND is not ``tools/call``. The scan
    is a structural no-op on those non-``tools/call`` responses, so skipping is a
    pure optimization. When the request / body / method is absent — or the request
    shape is malformed — this returns ``False`` so the default no-op-safe scrub
    still runs on every response body; correctness never depends on
    ``mcp.gatewayRequest`` being present.

    Args:
        mcp: The ``mcp`` object from the interceptor input (already defaulted to a
            dict by the caller).

    Returns:
        ``True`` to skip the scan (non-``tools/call`` method present), else
        ``False`` (the safe default: scrub the response body).
    """
    try:
        gateway_request = mcp.get("gatewayRequest", {}) or {}
        request_body = gateway_request.get("body", {}) or {}
        method = request_body.get("method")
    except Exception:  # noqa: BLE001 - the gate must never decide by raising
        # A malformed gatewayRequest shape (e.g. body is a raw string) must not
        # disable scrubbing — fall back to the safe default (do not skip). Broad on
        # purpose: this helper is the handler's fail-open/fail-closed boundary, so
        # an escape from here would be classified as an envelope-read failure and
        # pass the body through unscrubbed.
        return False
    return isinstance(method, str) and method != _TOOLS_CALL_METHOD


def _withheld_body(original_body: Any) -> Any:
    """Build the replacement body for a reply the scrubber could not process.

    Returns a JSON-RPC error rather than an empty body: the caller is an MCP client
    waiting on a specific request id, so an error it can correlate ends the call
    cleanly, whereas ``{}`` is a protocol violation that can leave it waiting. The
    original request id is echoed when it is a JSON-RPC-legal scalar.

    A BATCH reply (a top-level array, legal in the protocol versions this gateway
    serves) is answered with an ARRAY of errors — one per entry — so the client still
    gets the shape it is parsing and can correlate each id. The array is truncated to
    the first :data:`_MAX_WITHHELD_BATCH_ITEMS` entries so a large batch of small
    entries cannot inflate a withheld reply; ids past that point are not answered and
    fall back to the client's own timeout.

    Reads ONLY top-level ids — no traversal — so it cannot re-trip the recursion
    that brought us here, and any failure still degrades to an id-less error rather
    than raising out of the handler's ``except`` block.

    Carries none of the original body: no key, no value, no content.

    Args:
        original_body: The unscrubbed reply body being withheld.

    Returns:
        A JSON-RPC error object, or a list of them for a batch reply.
    """
    try:
        if isinstance(original_body, list):
            entries = original_body[:_MAX_WITHHELD_BATCH_ITEMS] or [None]
            return [_withheld_error(entry) for entry in entries]
    except Exception:  # noqa: BLE001 - never raise out of the handler's except
        return [_withheld_error(None)]
    return _withheld_error(original_body)


def _withheld_error(entry: Any) -> dict[str, Any]:
    """Build one JSON-RPC error object, echoing ``entry``'s id only if it is vetted.

    The id is the ONE field copied out of a body the scrubber never vouched for, so
    it is vetted before being echoed, not merely type-checked. A response is not
    obliged to carry a sane id: an over-deep reply whose ``id`` is itself an access
    key would otherwise put that key straight into the withheld body — the same leak
    this whole path exists to prevent, through the field meant to make the error
    useful.

    Numbers are echoed as-is (a JSON number cannot carry credential material). A
    string is echoed only when it is short, drawn from a narrow id charset, and free
    of every credential shape :func:`credential_scrubber.has_credential_shape`
    recognises — a non-recursive check, so it is safe here even though the recursive
    scrub just failed. Anything else becomes ``null``: correlation is best-effort,
    secrecy is not.

    Args:
        entry: The withheld reply (or one entry of a withheld batch) whose top-level
            ``id`` may be echoed.

    Returns:
        A JSON-RPC error object reporting that the reply was withheld.
    """
    request_id: Any = None
    try:
        if isinstance(entry, dict):
            candidate = entry.get("id")
            # JSON-RPC ids are string, number or null. `bool` is a subclass of
            # `int`, so exclude it explicitly.
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                request_id = candidate
            elif (
                isinstance(candidate, str)
                and len(candidate) <= _MAX_ECHOED_ID_CHARS
                and _SAFE_ID_PATTERN.match(candidate) is not None
                and not has_credential_shape(candidate)
            ):
                request_id = candidate
    except Exception:  # noqa: BLE001 - never raise out of the handler's except
        request_id = None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": _WITHHELD_ERROR_CODE,
            "message": _WITHHELD_ERROR_MESSAGE,
        },
    }


def _transformed_response(status_code: Any, body: Any) -> dict[str, Any]:
    """Build the RESPONSE interceptor output envelope.

    Args:
        status_code: The status code to echo back (the original
            ``mcp.gatewayResponse.statusCode``, or 200 when it was absent).
        body: The (scrubbed, or on the fail-safe path the original) reply body.

    Returns:
        An ``interceptorOutputVersion: "1.0"`` envelope carrying
        ``mcp.transformedGatewayResponse`` as ``{"statusCode", "body"}``; when the
        gateway sees this field it responds with it immediately.
    """
    return {
        "interceptorOutputVersion": _INTERCEPTOR_OUTPUT_VERSION,
        "mcp": {
            "transformedGatewayResponse": {
                "statusCode": status_code,
                "body": body,
            }
        },
    }
