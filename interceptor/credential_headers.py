"""REQUEST-interceptor credential HEADER channel.

Why the credentials travel as headers, not in the body
-----------------------------------------------------
An earlier version of this architecture carried the vended STS credentials in the
request BODY, at ``params.arguments["context"]``, on the reading that a Lambda
target has no header channel. It does have one, confirmed by live measurement: an
interceptor-set header that the target allowlists arrives at the Lambda under
``context.client_context.custom["bedrockAgentCorePropagatedHeaders"]``.

Carrying them as headers instead buys two things the body could not:

1. **Nothing credential-shaped is in the request body at all.** This is the
   measured gain, and it is narrower than it first looks. The old design did NOT
   put the credential in the tool contract: it rode as an UNDECLARED
   ``params.arguments`` field, kept out of every ``inputSchema`` precisely so the
   model never saw a credential-shaped property (``cdk/gateway_resources.py``
   declares only ``doc_id`` / ``query`` / ``body``, and never declared
   ``context``). So "the credential left the tool contract" is NOT what changed —
   it was never there.

   What changed is that the credential is no longer in the BODY, and the body is
   what the Gateway's vended ``APPLICATION_LOGS`` copy verbatim into CloudWatch
   through their ``requestBody`` field. With the credential in the body, enabling
   that delivery wrote a live secret key into a log group — observed, not
   theorised. With the credential on a header, the same records were re-measured
   with the real credentials in flight and carry no header name and no header
   value. A second, smaller gain: nothing is written into ``params`` /
   ``arguments`` at all, so the forwarded body is deep-equal to the one the
   Gateway received, which is a property a test can assert rather than a
   convention to remember.

   AWS makes a related argument for propagating an identity token by header
   rather than by tool parameter — "If you put the id_token in the tool schema
   instead, the FM becomes responsible for passing it, which means the token
   lands in prompts, traces, memory, and logs." It describes a failure mode this
   design never had, since the field was undeclared; it is cited here as the
   general case, not as the defect this change fixed.
   https://aws.amazon.com/blogs/security/identity-aware-ai-data-agents-with-aws-lake-formation-and-trusted-identity-propagation/

2. **A client-supplied value cannot win — ON A CALL THE INTERCEPTOR HANDLES.**
   The Gateway merges interceptor headers with the target's
   ``metadataConfiguration.allowedRequestHeaders`` allowlist, and an
   interceptor-provided value takes PRECEDENCE over a client-provided one. A
   header absent from the allowlist is dropped outright. The interceptor sets all
   four on every scoped call, so a hostile client that sends its own
   ``x-tvm-served-scope`` is overwritten rather than honoured.

   The qualifier is load-bearing: precedence only applies to a header the
   interceptor actually WRITES. On a path where it writes none, a client-supplied
   value for an allowlisted name is the only value present and would propagate.
   That is why an unclassifiable ``tools/call`` now FAILS CLOSED instead of
   passing through (``interceptor/handler.py``, asserted by
   ``tests/test_unclassifiable_tools_call_fails_closed.py``) — the allowlist is on
   all three targets, so a pass-through is no longer the harmless thing it was
   when the credentials rode in the body.
   https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-headers.html

What this module deliberately does NOT do
-----------------------------------------
It never logs a header VALUE. :func:`header_size_report` returns byte LENGTHS
only — that is the measurement that decides whether a vended session token fits
the Gateway's per-value limit, and a length is not a secret.

Hard limits enforced here (both from the header-propagation reference above):

* a header value may be at most 4096 bytes;
* a target may allowlist at most 10 headers — this module emits 4, and the
  service maintains its own ``x-amzn-bedrock-agentcore-policy-session-id`` entry
  alongside them, so the budget is 5 of 10.

An oversized value RAISES rather than being truncated or dropped. A truncated
credential would reach the tool as an unusable one and surface as an opaque
``AccessDenied`` from DynamoDB; raising makes the interceptor fail closed, which
is the same answer the rest of this component gives to every other impossible
state. AWS does not document a maximum STS session-token size, so this guard is
the only thing standing between a future, larger token and a silent corruption:
the one related documented figure is ``AssumeRole``'s
``MinimumSessionTokenSize``, whose own ceiling is 4096 bytes — exactly the header
limit, meaning a token padded to the maximum would sit right at the edge.
https://docs.aws.amazon.com/sdk-for-ruby/v3/api/Aws/STS/Types/AssumeRoleRequest.html

Constants:
    SERVED_SCOPE_HEADER / ACCESS_KEY_ID_HEADER / SECRET_ACCESS_KEY_HEADER /
    SESSION_TOKEN_HEADER: the four propagated header names.
    CREDENTIAL_HEADERS: the full set, for the CDK allowlist and the tests.
    MAX_HEADER_VALUE_BYTES: the Gateway's per-value limit.

Functions:
    build_credential_headers: Build the four headers from a served scope + creds.
    header_size_report: Byte lengths per header (never values), for the log line.
"""

from __future__ import annotations

from typing import Any

#: Header carrying the authoritative, JWT-derived scope. NOT a secret, but just
#: as authoritative as the credentials: it decides which partition the tool's key
#: is built in, so it must travel on the same un-model-writable channel. Leaving
#: it in the body would have handed the model the one value that selects the
#: partition.
SERVED_SCOPE_HEADER = "x-tvm-served-scope"

#: The three vended STS credential fields. Names mirror the snake_case fields the
#: interceptor already produced, so the mapping to the boto3 Session keyword
#: arguments on the tool side is unchanged in meaning.
ACCESS_KEY_ID_HEADER = "x-tvm-access-key-id"
SECRET_ACCESS_KEY_HEADER = "x-tvm-secret-access-key"
SESSION_TOKEN_HEADER = "x-tvm-session-token"

#: Mapping of vended-credential field name -> propagated header name. The KEYS
#: are exactly the three fields `vend_scoped_credentials` returns.
CREDENTIAL_FIELD_TO_HEADER: dict[str, str] = {
    "access_key_id": ACCESS_KEY_ID_HEADER,
    "secret_access_key": SECRET_ACCESS_KEY_HEADER,
    "session_token": SESSION_TOKEN_HEADER,
}

#: Every header this interceptor propagates. The CDK allowlists exactly this set
#: on each Lambda target; tests/test_header_contract_parity.py pins the two to
#: each other so a header added here cannot be silently dropped by the Gateway.
CREDENTIAL_HEADERS: tuple[str, ...] = (
    SERVED_SCOPE_HEADER,
    ACCESS_KEY_ID_HEADER,
    SECRET_ACCESS_KEY_HEADER,
    SESSION_TOKEN_HEADER,
)

#: Gateway limit on a single propagated header value.
MAX_HEADER_VALUE_BYTES = 4096

#: Gateway limit on the number of allowlisted headers per target. Informational
#: here; asserted against CREDENTIAL_HEADERS in the CDK and in the tests.
MAX_ALLOWED_HEADERS_PER_TARGET = 10


class CredentialHeaderError(RuntimeError):
    """Raised when the vended credentials cannot be expressed as headers.

    The only cause is a value exceeding :data:`MAX_HEADER_VALUE_BYTES`. The
    interceptor turns this into its standard fail-closed short circuit, so the
    target is never called with a truncated credential.

    The message names the offending header and its SIZE, never its value.
    """


def build_credential_headers(
    served_scope: str,
    creds: dict[str, str],
) -> dict[str, str]:
    """Build the four propagated headers carrying the scope and vended credentials.

    Args:
        served_scope: The authoritative, JWT-derived scope.
        creds: The vended credentials as returned by
            ``interceptor.scoped_credentials.vend_scoped_credentials`` —
            ``access_key_id`` / ``secret_access_key`` / ``session_token``.

    Returns:
        A dict of header name -> value, ready for
        ``mcp.transformedGatewayRequest.headers``.

    Raises:
        CredentialHeaderError: If a credential field is missing/empty, or if any
            value exceeds the Gateway's 4096-byte per-value limit. Both are
            fail-closed: the caller short-circuits rather than calling the target
            with an incomplete or truncated credential.
    """
    if not (isinstance(served_scope, str) and served_scope.strip()):
        raise CredentialHeaderError("served scope is missing or not a string")

    headers: dict[str, str] = {SERVED_SCOPE_HEADER: served_scope.strip()}

    for field, header_name in CREDENTIAL_FIELD_TO_HEADER.items():
        value = creds.get(field)
        if not (isinstance(value, str) and value):
            # Names the FIELD, never the value. A missing field means the vend
            # returned an incomplete credential, which must not be papered over.
            raise CredentialHeaderError(
                f"vended credential field {field!r} is missing or not a string"
            )
        headers[header_name] = value

    for header_name, value in headers.items():
        size = len(value.encode("utf-8"))
        if size > MAX_HEADER_VALUE_BYTES:
            raise CredentialHeaderError(
                f"header {header_name!r} is {size} bytes, over the Gateway's "
                f"{MAX_HEADER_VALUE_BYTES}-byte per-value limit; refusing to "
                f"send a truncated credential"
            )

    return headers


def header_size_report(headers: dict[str, str]) -> dict[str, int]:
    """Return the UTF-8 byte length of each header value, plus the total.

    This is the measurement that answers whether a real vended session token fits
    the Gateway's 4096-byte per-value limit — a figure AWS does not document, so
    it has to be observed. Lengths only: no value is returned, and therefore no
    value can be logged by a caller that logs this dict.

    Args:
        headers: The header map built by :func:`build_credential_headers`.

    Returns:
        A dict of header name -> byte length, plus a ``total`` key carrying the
        sum. The keys are header names, so the report is self-describing in a log
        line without naming any value.
    """
    report = {
        name: len(value.encode("utf-8")) for name, value in sorted(headers.items())
    }
    report["total"] = sum(report.values())
    return report


def oversize_headers(headers: dict[str, Any]) -> dict[str, int]:
    """Return the headers whose values exceed the per-value limit, with their sizes.

    Exposed for tests and for an operator reading the measurement: it answers
    "which value would be refused, and by how much" without exposing any value.

    Args:
        headers: A header map.

    Returns:
        A dict of header name -> byte length for the offending headers only.
        Empty when every value fits.
    """
    over: dict[str, int] = {}
    for name, value in headers.items():
        if not isinstance(value, str):
            continue
        size = len(value.encode("utf-8"))
        if size > MAX_HEADER_VALUE_BYTES:
            over[name] = size
    return over
