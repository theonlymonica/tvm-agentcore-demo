"""
Pure recursive credential scrubber for the RESPONSE interceptor.

This module is the credential-removal core of the RESPONSE interceptor. It is a
**pure** module: it computes only from its argument, performs no I/O, and does no
logging. The Lambda entry point that wraps it lives in
``response_interceptor/handler.py``; this module never touches the gateway
envelope.

RESPONSE interceptor I/O contract (where the scrubbed body sits on the wire):
    The gateway delivers the tool reply to the RESPONSE interceptor at
    ``mcp.gatewayResponse.body`` (alongside ``mcp.gatewayResponse.statusCode``),
    and the interceptor returns the transformed reply at
    ``mcp.transformedGatewayResponse`` as ``{"statusCode", "body"}`` inside an
    ``interceptorOutputVersion: "1.0"`` envelope; when that field is present the
    gateway responds with it immediately. The handler reads
    ``mcp.gatewayResponse.body``, passes it to :func:`scrub`, and places the
    returned scrubbed body into ``mcp.transformedGatewayResponse.body``. This
    module operates purely on that body value and is unaware of the envelope.
    AWS documentation reference:
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-types.html

Removal rules (applied to a deep copy — the caller's value is never mutated):
    1. Key-name removal (authoritative). At ANY depth, DELETE (not redact / not
       mask) any dict key whose NORMALISED name matches one of
       :data:`_CREDENTIAL_KEY_NAMES`. Normalisation (see :func:`_normalise_key`)
       applies NFKC, casefolds, and strips ``_``, ``-``, ``.`` and whitespace, so
       ``access_key_id``, ``AccessKeyId``, ``accessKeyId``, ``ACCESS_KEY_ID`` and
       ``Access-Key-Id`` are all one name. Deleting the key drops its whole
       subtree in one removal.
    2. Value-shape removal (defense-in-depth HEURISTICS). At ANY depth and
       regardless of the enclosing key name, credential-shaped SUBSTRINGS are
       EXCISED from every string — an access-key ID, a secret-access-key-shaped
       run, a session-token-shaped run, and the SigV4 presigned-URL query
       parameters. When excision empties the string (i.e. the value WAS the
       credential), the whole dict key / list element is removed, so a whole-value
       credential still disappears entirely rather than leaving an empty
       placeholder behind.
    3. Embedded-JSON traversal. A string value that parses as a JSON object or
       array is scrubbed STRUCTURALLY (rules 1 and 2 applied to the parsed value)
       and re-serialised — this is the MCP ``content[].text`` shape, where a tool
       reply's real payload is a JSON-encoded string rather than a parsed dict. If
       nothing was removed the ORIGINAL string is returned byte-for-byte, so the
       deep-equal round-trip on credential-free bodies is preserved exactly.

Depth bound (why this module RAISES rather than degrading):
    The traversal is recursive and ``copy.deepcopy`` recurses too, so a body nested
    deeper than the interpreter's stack allows used to abort the scrub with
    ``RecursionError`` PART-WAY THROUGH. There is no safe way to report that as a
    value: a partially scrubbed body is indistinguishable from a clean one, and the
    handler's blanket ``except`` turned it into a pass-through of the original
    unscrubbed reply — the exception path doubled as a scrubber bypass. So
    :data:`_MAX_BODY_DEPTH` caps the nesting and :class:`DepthLimitExceeded` (a
    :class:`ScrubError`) is raised for anything past it, deterministically and
    before any copying. The budget spans the whole walk, including containers rule
    3 decodes out of embedded JSON strings, so chaining JSON-inside-JSON cannot
    win back the unbounded recursion; an embedded string the JSON DECODER cannot
    bottom out raises the same error rather than quietly falling back to a
    text-only scan that never applies rule 1. The handler treats the raise as
    fail-CLOSED: it withholds the reply.

Why substrings and normalised names (the gap this closes):
    Both rules used to be exact-match only: rule 2's regex was fully anchored
    (``^...$``) and rule 1's key set was case-sensitive snake_case. A key ID
    embedded in prose, a log line, a presigned URL or a JSON-encoded blob passed
    straight through, as did the raw STS ``AssumeRole`` / ``GetSessionToken``
    reply shape (``AccessKeyId`` / ``SecretAccessKey`` / ``SessionToken``) and the
    boto3 ``Session`` keyword names (``aws_access_key_id`` / ...) that the tool
    Lambdas actually build. A secret access key and a session token had no
    value-shape rule at all, so under any non-listed key — or as free text — the
    highest-value material had the weakest coverage.

HEURISTIC / grounding note:
    AWS documents credential FIELD NAMES but publishes no authoritative, stable
    regular expression for any credential STRING shape. Every value-shape rule
    below is therefore explicitly a defense-in-depth heuristic layered ON TOP OF
    the authoritative key-name removal (rule 1), and is NOT grounded in a
    published AWS specification:

      * access-key ID — ``ASIA``/``AKIA`` + 16 uppercase-alnum chars (20 total);
      * secret access key — a 40-char run of ``[A-Za-z0-9+/]``;
      * session token — a run of 300+ chars of ``[A-Za-z0-9+/=]``, with no upper
        bound, because AWS documents the token size as unbounded and warns against
        assuming a maximum.

    The two long-run heuristics additionally require the run to mix lowercase,
    uppercase AND digits. Randomly generated AWS secrets satisfy that with
    overwhelming probability, while single-case material that is NOT a credential
    — hex digests, uppercase identifiers, lowercase prose tokens — does not, which
    is what keeps the heuristics from eating legitimate document content. The
    secret-key rule is also bounded by non-charset characters on both sides, so a
    40-char window inside a longer base64 blob is never excised. The access-key-ID
    rule is deliberately NOT bounded: its 4-character prefix and fixed length are
    distinctive enough that requiring a boundary would only create a concatenation
    bypass.

    The accepted residual cost is that a very long mixed-class base64 run in
    legitimate content is treated as a session token and excised. That is the
    deliberate trade for not capping a length AWS refuses to bound, and it is
    pinned by a test so it cannot be mistaken for a regression.

Counting semantics (documented, consistent):
    :func:`scrub` returns an integer ``removed_count`` for count-only logging by
    the handler (the removed NAMES and VALUES are never returned or logged). Each
    of the following counts as exactly ONE removal:
        - one removed credential-key-name dict key (rule 1) — counted once even
          though it may drop a whole subtree;
        - one excised credential-shaped substring (rule 2). A string holding two
          credentials therefore counts two; a string that WAS one credential
          counts one, whether it was a dict value or a list element.
    A key removed by rule 1 is never also scanned under rule 2 (the value is
    dropped with the key), so removals are never double-counted.

Functions:
    scrub: Deep-copy the input once, remove all credential-shaped keys and
        values at any depth, and return ``(scrubbed_value, removed_count)``.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any, Callable
from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Rule 1: authoritative credential key names, stored in their NORMALISED form (see
# `_normalise_key`) and matched against normalised dict keys at any depth; a
# matching key is deleted outright — deleted, never redacted/masked. Normalisation
# is what makes one entry cover the snake_case, PascalCase, camelCase,
# SCREAMING_CASE and kebab-case spellings of the same field, so the raw STS reply
# shape and the boto3 keyword shape are covered by the same entries as the
# project's own wire contract.
#
# The families covered, and why each is here:
#   * the injected `context` object and its `tenant_credentials` child — this
#     repository's wire contract (see interceptor/handler.py);
#   * `credentials` — the STS `AssumeRole` / `GetSessionToken` reply member that
#     carries the three fields;
#   * access key id / secret access key / session token — the three fields, in
#     both the bare spelling and the `aws_`-prefixed boto3 `Session` keyword
#     spelling that tools/common/credentials_context.py maps them onto;
#   * `security_token` / `aws_security_token` — the legacy AWS SDK spelling of the
#     session token;
#   * the SigV4 HTTP/query parameter names, which can appear as headers or as
#     parsed query-string keys.
# ---------------------------------------------------------------------------
_CREDENTIAL_KEY_NAME_SPELLINGS: frozenset[str] = frozenset(
    {
        "context",
        "tenant_credentials",
        "credentials",
        "access_key_id",
        "secret_access_key",
        "session_token",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "security_token",
        "aws_security_token",
        "x-amz-credential",
        "x-amz-signature",
        "x-amz-security-token",
    }
)

#: Characters stripped from a key before comparison, so separator style and
#: incidental whitespace (including the zero-width characters that survive a
#: copy/paste) cannot defeat the match.
_KEY_NOISE_PATTERN = re.compile(r"[\s_\-.\u200b\u200c\u200d\ufeff]+")


def _normalise_key(key: str) -> str:
    """Return the comparison form of a dict key for the rule-1 name match.

    Applies NFKC compatibility normalisation (so fullwidth / compatibility
    variants collapse onto their ASCII spelling), strips separator and zero-width
    noise, and casefolds. ``AccessKeyId``, ``access_key_id``, ``ACCESS-KEY-ID``
    and ``accessKeyId`` all normalise to ``accesskeyid``.

    Args:
        key: A dict key encountered during traversal.

    Returns:
        The normalised key string used for the credential-name comparison.
    """
    return _KEY_NOISE_PATTERN.sub("", unicodedata.normalize("NFKC", key)).casefold()


#: Rule 1's comparison set: the spellings above, normalised once at import.
_CREDENTIAL_KEY_NAMES: frozenset[str] = frozenset(
    _normalise_key(name) for name in _CREDENTIAL_KEY_NAME_SPELLINGS
)

# ---------------------------------------------------------------------------
# Rule 2: DEFENSE-IN-DEPTH value-shape HEURISTICS. Each pattern is deliberately
# UNANCHORED and boundary-guarded so the shape is caught wherever it sits — a
# standalone value, a sentence, a log line, a URL, a JSON blob — while never
# excising a fragment of some longer unrelated token. See the module docstring for
# the explicit statement that none of these is grounded in a published AWS
# specification.
# ---------------------------------------------------------------------------

#: Access-key ID: `ASIA` (STS temporary) / `AKIA` (long-term) + 16 uppercase-alnum
#: chars. Deliberately UNBOUNDED, unlike the long-run heuristics below: the 4-char
#: prefix plus fixed 20-char length makes an occurrence overwhelmingly likely to be
#: a real key even when it is concatenated into a longer token, and requiring a
#: boundary would let `<junk>AKIA...<junk>` through — a real bypass. The long-run
#: rules below run FIRST, so a key ID appearing inside an excised secret or session
#: token is never reached by this rule.
_ACCESS_KEY_ID_PATTERN = re.compile(r"(?:ASIA|AKIA)[A-Z0-9]{16}")

#: Secret-access-key candidate: exactly 40 chars of the base64 alphabet AWS uses
#: for secrets, bounded by non-charset characters. Candidates are filtered by
#: `_mixes_character_classes` before removal.
_SECRET_ACCESS_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40}(?![A-Za-z0-9+/])"
)

#: Session-token candidate: a very long base64 run. Session tokens are hundreds of
#: characters; 300 is a deliberately conservative floor that no ordinary
#: identifier reaches. Candidates are filtered by `_mixes_character_classes`.
#:
#: There is NO upper bound, on purpose. AWS states: "The size of the security token
#: that STS API operations return is not fixed. We strongly recommend that you make
#: no assumptions about the maximum size."
#: (https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html)
#: Capping the length would therefore create a bypass that AWS's own documentation
#: predicts. The accepted cost is that a very long MIXED-CASE-plus-digits base64
#: run in legitimate content — an embedded attachment, say — is excised too. Single
#: case runs and hex are unaffected (see `_mixes_character_classes`), which covers
#: ordinary document text; see the matching test in
#: tests/test_response_interceptor_scrub_gaps.py, which pins this as intended
#: behaviour rather than an accident.
_SESSION_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/=]{300,}")

#: SigV4 presigned-URL query parameters. These carry credential material inside a
#: URL string, under no credential key name at all, so no other rule reaches them.
#: The whole `name=value` pair is excised (plus a trailing `&` when present).
_SIGV4_QUERY_PARAM_PATTERN = re.compile(
    r"(?i)(?<=[?&])(?:x-amz-credential|x-amz-signature|x-amz-security-token)"
    r"=[^&\s\"'<>]*&?"
)

#: Separator debris left behind once a query parameter is excised: `?&next=1`
#: collapses to `?next=1`, and a trailing `?` / `&` is dropped. Tidying is purely
#: cosmetic and never counted as a removal.
_COLLAPSE_QUERY_SEPARATOR_PATTERN = re.compile(r"\?&+")
_TRAILING_QUERY_SEPARATOR_PATTERN = re.compile(r"[?&]+(?=[\s\"'<>]|$)")

#: Upper bound on a string we will try to parse as embedded JSON (rule 3). Guards
#: against spending the interceptor's budget parsing a very large document body
#: that merely happens to start with `{`.
_MAX_EMBEDDED_JSON_CHARS = 1_048_576

#: Maximum number of nested containers the scrubber will traverse before refusing
#: to vouch for the body. Both the traversal below AND `copy.deepcopy` recurse once
#: per container, so without a cap a deeply nested body raises `RecursionError`
#: PART-WAY THROUGH the scrub — and the handler's blanket `except` used to turn
#: that into a pass-through of the ORIGINAL body, making the exception path a
#: silent scrubber bypass.
#:
#: The cap counts CONTAINERS (dicts and lists); scalars cannot recurse. It is a
#: budget spent across the WHOLE walk, including structures rule 3 decodes out of
#: embedded JSON strings, so a chain of JSON-inside-JSON cannot re-create the
#: unbounded recursion one decode at a time.
#:
#: Why 100. Measured on CPython 3.14 at the default 1000-frame recursion limit,
#: the recursive DEEP COPY gives out first — from ~500 nested containers — while
#: the traversal itself survives to ~1000 for a plain dict chain and fewer for
#: shapes that spend more frames per level (rule 3's decode-and-recurse adds
#: several). 100 therefore clears the binding ceiling by ~5x, and sits far above
#: any reply this system produces: the deepest real shape is
#: `result.content[].text` plus the payload decoded out of it, well under 10.
#:
#: The trade this accepts: a body nested 101..~500 deep used to scrub correctly and
#: now gets withheld instead. Nothing hand-written reaches that, but generated JSON
#: (an XML->JSON conversion, a deep org tree) conceivably could, so the ceiling is
#: a documented number to raise rather than a hidden one to discover.
_MAX_BODY_DEPTH = 100

#: Fixed, content-free messages for :class:`DepthLimitExceeded`. They name only the
#: constant or the failing stage, never the body, a key, or a value, so logging the
#: exception (or its traceback) cannot leak the reply.
_DEPTH_LIMIT_MESSAGE = f"body nests deeper than {_MAX_BODY_DEPTH} containers"
_EMBEDDED_DEPTH_MESSAGE = "embedded JSON nests deeper than the decoder can parse"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ScrubError(Exception):
    """The body could not be scrubbed to completion.

    Raised INSTEAD of returning a half-scrubbed body, so the caller never mistakes
    an aborted scrub for a clean one. The handler
    (``response_interceptor/handler.py``) fails CLOSED on this exception — it
    withholds the reply rather than passing the unscrubbed original through. The
    rule that the interceptor never rejects, errors or blocks is about not breaking
    the MCP protocol on the basis of a message's method, shape or content; it is
    not a licence to ship credentials the scrubber never got to see.

    Subclasses carry NO body content, key name or value — every message is a fixed
    string.
    """


class DepthLimitExceeded(ScrubError):
    """The body nests deeper than :data:`_MAX_BODY_DEPTH` containers.

    Signals the one input the recursive scrubber cannot process safely. Raised
    either by the pre-copy check (:func:`_assert_within_depth`) or by the traversal
    itself when rule 3 decodes further containers out of an embedded JSON string.
    """


def _mixes_character_classes(candidate: str) -> bool:
    """Return whether ``candidate`` mixes lowercase, uppercase AND digits.

    The discriminator for the two long-run heuristics (secret access key, session
    token). A randomly generated AWS secret mixes all three with overwhelming
    probability; single-case material that is not a credential — a hex digest, an
    uppercase identifier, a lowercase prose token — does not, and is left alone.

    Args:
        candidate: The matched run of base64-alphabet characters.

    Returns:
        ``True`` when at least one lowercase letter, one uppercase letter and one
        digit are present, otherwise ``False``.
    """
    has_lower = has_upper = has_digit = False
    for char in candidate:
        if char.islower():
            has_lower = True
        elif char.isupper():
            has_upper = True
        elif char.isdigit():
            has_digit = True
        if has_lower and has_upper and has_digit:
            return True
    return False


def _excise(
    pattern: re.Pattern[str],
    text: str,
    accept: Callable[[str], bool] | None = None,
) -> tuple[str, int]:
    """Remove every ``pattern`` match ``accept`` approves and count the removals.

    Excision removes the matched credential material outright and substitutes
    NOTHING — no mask, no ``[REDACTED]`` marker — so the delete-never-redact rule
    holds for substrings exactly as it does for whole keys.

    Args:
        pattern: The credential-shape pattern to search for (unanchored).
        text: The string to scrub.
        accept: Optional predicate applied to each matched substring; when it
            returns ``False`` the match is left in place and not counted. Used to
            filter the entropy-style long-run candidates.

    Returns:
        A ``(scrubbed_text, removed_count)`` tuple.
    """
    removed = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed
        if accept is not None and not accept(match.group(0)):
            return match.group(0)
        removed += 1
        return ""

    return pattern.sub(_replace, text), removed


def _scrub_text(text: str) -> tuple[str, int]:
    """Excise every credential-shaped substring from ``text`` (rule 2).

    Rules run longest-shape-first so a session token is excised whole rather than
    being nibbled by the 40-char secret-key rule, which would otherwise match
    windows inside it.

    Args:
        text: The string to scrub.

    Returns:
        A ``(scrubbed_text, removed_count)`` tuple. ``removed_count`` is ``0`` and
        the text is returned unchanged when nothing credential-shaped is present.
    """
    removed = 0

    text, count = _excise(_SIGV4_QUERY_PARAM_PATTERN, text)
    removed += count
    text, count = _excise(
        _SESSION_TOKEN_PATTERN, text, accept=_mixes_character_classes
    )
    removed += count
    text, count = _excise(
        _SECRET_ACCESS_KEY_PATTERN, text, accept=_mixes_character_classes
    )
    removed += count
    text, count = _excise(_ACCESS_KEY_ID_PATTERN, text)
    removed += count

    if removed:
        # Cosmetic only: tidy the separator debris an excised query parameter
        # leaves behind. Never counted as a removal.
        text = _COLLAPSE_QUERY_SEPARATOR_PATTERN.sub("?", text)
        text = _TRAILING_QUERY_SEPARATOR_PATTERN.sub("", text)

    return text, removed


def _assert_within_depth(value: Any) -> None:
    """Raise :class:`DepthLimitExceeded` if ``value`` nests past the container cap.

    Walks ITERATIVELY (an explicit stack of iterators, zero recursion) so the guard
    itself can never raise ``RecursionError`` on the very input it exists to reject.
    It runs BEFORE ``copy.deepcopy`` in :func:`scrub`, because deepcopy recurses once
    per container too and would otherwise exhaust the stack before any depth-aware
    code got to look at the body.

    Only CONTAINERS are pushed, and only the ancestor chain is retained, so the
    guard's own memory is bounded by :data:`_MAX_BODY_DEPTH` rather than by the size
    of the body. Enqueueing every scalar instead would let a wide, shallow reply — a
    long list of strings — make the guard allocate proportionally to the payload
    before the copy even starts. Scalars cannot recurse, so they are irrelevant to
    what this check is for.

    Args:
        value: The body about to be copied and scrubbed.

    Raises:
        DepthLimitExceeded: When containers nest more than
            :data:`_MAX_BODY_DEPTH` deep.
    """
    if not isinstance(value, (dict, list)):
        return
    # Each entry is an open container's child iterator, so len(stack) is exactly the
    # number of containers currently nested above the child being examined.
    stack: list[Iterator[Any]] = [iter(value.values() if isinstance(value, dict) else value)]
    while stack:
        if len(stack) > _MAX_BODY_DEPTH:
            raise DepthLimitExceeded(_DEPTH_LIMIT_MESSAGE)
        try:
            child = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(child, dict):
            stack.append(iter(child.values()))
        elif isinstance(child, list):
            stack.append(iter(child))


def _assert_depth_budget(depth: int) -> None:
    """Raise :class:`DepthLimitExceeded` when the traversal is too deep.

    Called on entry to each recursive step. ``depth`` is 0-based, so a container at
    ``depth`` is the ``depth + 1``-th nested container and the budget is spent once
    ``depth`` reaches :data:`_MAX_BODY_DEPTH` — the same container count
    :func:`_assert_within_depth` enforces.

    That pre-copy check already bounds the body handed to :func:`scrub`, so in
    practice this fires only on the containers rule 3 decodes out of an embedded JSON
    string mid-walk — depth the pre-copy check cannot see, since those containers do
    not exist yet when it runs.

    Args:
        depth: Container depth of the node about to be traversed.

    Raises:
        DepthLimitExceeded: When the container count would exceed
            :data:`_MAX_BODY_DEPTH`.
    """
    if depth >= _MAX_BODY_DEPTH:
        raise DepthLimitExceeded(_DEPTH_LIMIT_MESSAGE)


def has_credential_shape(text: str) -> bool:
    """Return whether ``text`` carries anything the value-shape rules would excise.

    A cheap, NON-RECURSIVE predicate (plain regex work, no traversal, no copying) so
    it is safe to call on the handler's failure path — including after the recursive
    scrub has already given up. It answers only "does this short string look like
    credential material", which is what the handler needs before echoing a field out
    of a body the scrubber never vouched for.

    Args:
        text: The string to test.

    Returns:
        ``True`` when at least one credential-shaped substring is present.
    """
    _, removed = _scrub_text(text)
    return removed > 0


def _scrub_embedded_json(text: str, depth: int) -> tuple[str, int] | None:
    """Structurally scrub ``text`` when it is a JSON-encoded object or array.

    This is rule 3. An MCP tool reply carries its real payload as a JSON-encoded
    string (``result.content[].text``), so without this the structural rules would
    never see the payload's keys at all — a ``SecretAccessKey`` field inside such a
    string would be invisible to rule 1.

    Only re-serialises when something was actually removed, so a credential-free
    body still round-trips BYTE-IDENTICAL and the deep-equal no-op is preserved
    (re-serialising would otherwise reformat whitespace).

    Args:
        text: A candidate string value.
        depth: Container depth of the string being decoded. The decoded containers
            continue the SAME depth budget, so JSON nested inside JSON cannot buy
            fresh recursion allowance.

    Returns:
        ``None`` when ``text`` is not JSON-encoded container (so the caller
        applies the plain-text rules instead); otherwise a
        ``(text, removed_count)`` tuple, where ``text`` is the re-serialised
        scrubbed JSON if anything was removed and the untouched original if not.

    Raises:
        DepthLimitExceeded: When the decoded value nests past the shared budget, or
            when the DECODER itself cannot reach the bottom of it. The second case
            must not degrade to the plain-text rules: rule 1 (credential key names)
            would never run on the payload, and the string would come back reporting
            ``removed=0`` — a clean-looking reply that was never structurally
            scrubbed. That is the same silent bypass this cap exists to close, so it
            is routed to the handler's fail-closed branch instead.
    """
    stripped = text.strip()
    if (
        not stripped
        or stripped[0] not in "{["
        or len(text) > _MAX_EMBEDDED_JSON_CHARS
    ):
        return None

    try:
        parsed = json.loads(text)
    except ValueError:
        # Not JSON after all (a document body that merely starts with `{`). The
        # caller falls back to the plain-text rules, which is correct here: there is
        # no structure to traverse.
        return None
    except RecursionError as exc:
        # The decoder ran out of stack, so the payload is nested beyond anything we
        # can inspect structurally. Refuse the body rather than silently downgrading
        # to a text-only scan.
        raise DepthLimitExceeded(_EMBEDDED_DEPTH_MESSAGE) from exc

    if isinstance(parsed, dict):
        removed = _scrub_dict(parsed, depth + 1)
    elif isinstance(parsed, list):
        removed = _scrub_list(parsed, depth + 1)
    else:  # pragma: no cover - a leading `{`/`[` cannot decode to a scalar.
        return None

    if removed == 0:
        return text, 0
    return json.dumps(parsed), removed


def _scrub_string_value(text: str, depth: int) -> tuple[str, int]:
    """Scrub one string value with rule 3 if it is JSON, else rule 2.

    Args:
        text: The string value or list element to scrub.
        depth: Container depth of the string, carried into rule 3's decoded
            containers.

    Returns:
        A ``(scrubbed_text, removed_count)`` tuple.

    Raises:
        DepthLimitExceeded: When rule 3 decodes containers past the depth budget.
    """
    embedded = _scrub_embedded_json(text, depth)
    if embedded is not None:
        return embedded
    return _scrub_text(text)


def _scrub_dict(node: dict[Any, Any], depth: int) -> int:
    """Scrub a dict IN PLACE and return the number of removals within it.

    Applies rule 1 (delete credential-name keys), then rules 2/3 to string values
    (removing the key outright when the value WAS the credential and excision
    emptied it), then recurses into any remaining nested dict/list values.
    Operates on the already-copied node — no further copying happens here.

    Args:
        node: A dict from the deep-copied body; mutated in place.
        depth: Container depth of ``node`` within the body being scrubbed.

    Returns:
        The count of removals at this level and below.

    Raises:
        DepthLimitExceeded: When ``depth`` exceeds :data:`_MAX_BODY_DEPTH`.
    """
    _assert_depth_budget(depth)
    removed = 0
    # Iterate over a snapshot of the keys so deletions during the loop are safe.
    for key in list(node.keys()):
        # Rule 1: normalised credential key name -> delete the key (and its whole
        # subtree) as ONE removal, without scanning the value.
        if isinstance(key, str) and _normalise_key(key) in _CREDENTIAL_KEY_NAMES:
            del node[key]
            removed += 1
            continue

        value = node[key]

        if isinstance(value, str):
            scrubbed, count = _scrub_string_value(value, depth)
            removed += count
            if count:
                if scrubbed.strip():
                    node[key] = scrubbed
                else:
                    # The value was nothing but credential material: drop the
                    # key/value pair rather than leave an empty placeholder.
                    del node[key]
            continue

        # Otherwise recurse into nested containers to any depth.
        if isinstance(value, dict):
            removed += _scrub_dict(value, depth + 1)
        elif isinstance(value, list):
            removed += _scrub_list(value, depth + 1)

    return removed


def _scrub_list(node: list[Any], depth: int) -> int:
    """Scrub a list IN PLACE and return the number of removals within it.

    List elements have no key name, so rule 1 does not apply to them directly;
    an element that is itself a dict or list is traversed (where rule 1 applies to
    its keys). A string element is scrubbed by rules 2/3, and is DROPPED when
    excision empties it. Operates on the already-copied node — no further copying
    happens here.

    Args:
        node: A list from the deep-copied body; mutated in place via slice assign.
        depth: Container depth of ``node`` within the body being scrubbed.

    Returns:
        The count of removals at this level and below.

    Raises:
        DepthLimitExceeded: When ``depth`` exceeds :data:`_MAX_BODY_DEPTH`.
    """
    _assert_depth_budget(depth)
    removed = 0
    kept: list[Any] = []
    for item in node:
        if isinstance(item, str):
            scrubbed, count = _scrub_string_value(item, depth)
            removed += count
            if count and not scrubbed.strip():
                # The element was nothing but credential material: drop it.
                continue
            kept.append(scrubbed if count else item)
            continue

        # Recurse into nested containers to any depth before keeping the element.
        if isinstance(item, dict):
            removed += _scrub_dict(item, depth + 1)
        elif isinstance(item, list):
            removed += _scrub_list(item, depth + 1)

        kept.append(item)

    # Rebind the list contents in place so the parent's reference stays valid.
    node[:] = kept
    return removed


def scrub(value: Any) -> tuple[Any, int]:
    """Remove all credential-shaped keys and values from ``value`` at any depth.

    Deep-copies ``value`` EXACTLY ONCE at the top so the caller's object is never
    mutated, then walks every nested dict and list to any depth applying the
    removal rules described in the module docstring. Deterministic and side-effect
    free: no logging, no I/O.

    A credential-free value round-trips unchanged (deep-equal to the input) with a
    ``removed_count`` of ``0`` — the removal is a structural no-op on bodies such
    as ``initialize`` / ``tools/list`` / ``notifications/initialized`` / ``ping``
    replies.

    A bare top-level STRING body is scrubbed too: it has no enclosing key or list
    to be removed from, so credential-shaped substrings are excised in place and
    the (possibly empty) string is returned. Other bare scalars are returned
    as-is. In practice the RESPONSE interceptor body is a JSON-RPC object (dict),
    so the traversal normally starts at a dict.

    Args:
        value: The response body to scrub (typically a JSON-RPC dict). May be any
            JSON-shaped value (dict, list, str, int, float, bool, None).

    Returns:
        A ``(scrubbed_value, removed_count)`` tuple where ``scrubbed_value`` is the
        deep-copied, credential-free value and ``removed_count`` is the integer
        number of removals. The tuple NEVER contains the removed key names or
        removed values.

    Raises:
        DepthLimitExceeded: When the body nests deeper than
            :data:`_MAX_BODY_DEPTH` containers, here or in a structure rule 3
            decodes out of an embedded JSON string. Raising is deliberate: the
            alternative — returning a body the recursion abandoned half-scrubbed —
            is indistinguishable to the caller from a clean scrub. The handler
            withholds the reply instead of shipping it.
    """
    # Reject an over-deep body BEFORE copying: `copy.deepcopy` recurses per
    # container too, so on such a body it, not the traversal, is what blows the
    # stack — and a RecursionError from here reads to the handler exactly like one
    # from anywhere else.
    _assert_within_depth(value)

    # Deep-copy ONCE at the top. Every mutation below targets this copy, so the
    # caller's `value` is never touched.
    scrubbed = copy.deepcopy(value)

    if isinstance(scrubbed, dict):
        removed_count = _scrub_dict(scrubbed, 0)
    elif isinstance(scrubbed, list):
        removed_count = _scrub_list(scrubbed, 0)
    elif isinstance(scrubbed, str):
        scrubbed, removed_count = _scrub_string_value(scrubbed, 0)
    else:
        # A bare non-string scalar cannot carry a credential shape.
        removed_count = 0

    return scrubbed, removed_count
