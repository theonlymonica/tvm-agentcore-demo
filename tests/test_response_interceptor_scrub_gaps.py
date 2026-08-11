"""Adversarial coverage for the RESPONSE interceptor credential scrubber.

Companion to ``tests/test_response_interceptor_strip.py``, which validates the
scrubber's original exact-match contract. That file restates the production
regex and key set as its own oracle, so it can only ever confirm that the
implementation matches itself — it passes unchanged whether or not a credential
in a slightly different shape survives.

This module deliberately shares NO constant with the production module. Every
credential here is a hand-written literal in a shape that a real AWS API
returns, and the primary assertion is a blunt one: after :func:`scrub`, the
literal must not appear ANYWHERE in the output, at any depth, inside any string.
That assertion cannot drift with the implementation.

The shapes covered are the ones the anchored/case-sensitive contract missed:

* the raw STS ``AssumeRole`` / ``GetSessionToken`` reply (PascalCase
  ``Credentials`` / ``AccessKeyId`` / ``SecretAccessKey`` / ``SessionToken``), plus
  the camelCase, SCREAMING_SNAKE and kebab-case spellings of the same fields;
* the boto3 ``Session`` keyword names (``aws_access_key_id`` /
  ``aws_secret_access_key`` / ``aws_session_token``) that the tool Lambdas build
  from the injected context;
* credentials EMBEDDED in a larger string — prose, a log line, a URL — rather
  than standing alone as a whole value;
* the SigV4 presigned-URL query parameters, which live inside a URL string under
  no credential key name at all;
* a bare secret access key and a bare session token, which previously had no
  value-shape rule whatsoever;
* a JSON-ENCODED payload carried in a string (the MCP ``content[].text`` shape),
  where the structural rules never saw the payload's keys.

A matching set of NEGATIVE cases pins the false-positive boundary, so the
heuristics cannot later be widened into something that shreds legitimate
document content: hex digests, single-case runs and near-miss prefixes must all
survive untouched.

Import resolution mirrors the sibling tests: the root ``conftest.py`` prepends
``response_interceptor/`` to ``sys.path``, so ``credential_scrubber`` resolves to
``response_interceptor/credential_scrubber.py``.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from credential_scrubber import scrub

# ---------------------------------------------------------------------------
# Fixtures — hand-written credential literals in real AWS shapes. These are the
# published AWS documentation example values (not live material); their only
# job is to be structurally indistinguishable from the real thing.
# ---------------------------------------------------------------------------

#: A long-term access key ID (`AKIA` prefix), 20 chars.
ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"

#: An STS temporary access key ID (`ASIA` prefix), 20 chars.
TEMP_ACCESS_KEY_ID = "ASIAY34FZKBOKMUTVV7A"

#: A secret access key: exactly 40 chars of the base64 alphabet, mixing case and
#: digits the way a randomly generated secret does.
SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

#: A session token: STS tokens are hundreds of base64 characters. Built long
#: enough to clear the production floor with room to spare, and mixed-class.
SESSION_TOKEN = "FwoGZXIvYXdzEBYaD" + ("aB3xY9" * 70) + "9Zx1="

#: Every literal that must never survive a scrub.
ALL_CREDENTIALS = (
    ACCESS_KEY_ID,
    TEMP_ACCESS_KEY_ID,
    SECRET_ACCESS_KEY,
    SESSION_TOKEN,
)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _flatten_strings(value: Any) -> list[str]:
    """Return every string reachable in ``value``, keys included, at any depth.

    Walks dicts and lists exhaustively so the leak check cannot be defeated by
    nesting. Non-string scalars contribute nothing.

    Args:
        value: Any JSON-shaped value.

    Returns:
        A flat list of every string found, including dict keys.
    """
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str):
                found.append(key)
            found.extend(_flatten_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_flatten_strings(child))
    return found


def assert_no_credential_survives(scrubbed: Any) -> None:
    """Fail if any credential literal appears anywhere in ``scrubbed``.

    Checks containment (not equality), so a credential embedded in a longer
    string is caught just like a standalone value.

    Args:
        scrubbed: The value returned by :func:`scrub`.
    """
    haystacks = _flatten_strings(scrubbed)
    for credential in ALL_CREDENTIALS:
        for haystack in haystacks:
            assert credential not in haystack, (
                f"credential of length {len(credential)} survived the scrub "
                f"inside a string of length {len(haystack)}"
            )


def scrub_and_assert_clean(body: Any) -> tuple[Any, int]:
    """Scrub ``body``, assert nothing leaked and the input was not mutated.

    Args:
        body: The response body to scrub.

    Returns:
        The ``(scrubbed, removed_count)`` tuple, for further assertions.
    """
    original = copy.deepcopy(body)
    scrubbed, removed = scrub(body)
    assert body == original, "scrub() mutated its argument instead of copying it"
    assert_no_credential_survives(scrubbed)
    assert removed > 0, "a body carrying credentials must report removals"
    return scrubbed, removed


# ---------------------------------------------------------------------------
# Gap B — key-name spellings the case-sensitive snake_case set missed
# ---------------------------------------------------------------------------


def test_raw_sts_pascalcase_reply_is_scrubbed() -> None:
    """The verbatim STS ``AssumeRole`` reply shape must not survive.

    ``AccessKeyId`` / ``SecretAccessKey`` / ``SessionToken`` under ``Credentials``
    is what boto3 hands back before this project maps it to snake_case. If a tool
    ever echoes that object, the scrubber is the only thing between it and the
    model.
    """
    body = {
        "jsonrpc": "2.0",
        "result": {
            "Credentials": {
                "AccessKeyId": TEMP_ACCESS_KEY_ID,
                "SecretAccessKey": SECRET_ACCESS_KEY,
                "SessionToken": SESSION_TOKEN,
                "Expiration": "2026-08-05T18:00:00Z",
            },
            "AssumedRoleUser": {"Arn": "arn:aws:sts::111122223333:assumed-role/r/s"},
        },
    }

    scrubbed, _ = scrub_and_assert_clean(body)

    assert "Credentials" not in scrubbed["result"]
    # Deleting the credential key drops its whole subtree, non-secret siblings
    # included; the unrelated sibling key is untouched.
    assert "AssumedRoleUser" in scrubbed["result"]
    assert scrubbed["jsonrpc"] == "2.0"


@pytest.mark.parametrize(
    "key_spelling",
    [
        "AccessKeyId",
        "accessKeyId",
        "ACCESS_KEY_ID",
        "Access-Key-Id",
        "access key id",
        "access_key_id",
        "SecretAccessKey",
        "secretAccessKey",
        "SECRET_ACCESS_KEY",
        "secret-access-key",
        "SessionToken",
        "sessionToken",
        "SESSION_TOKEN",
        "session-token",
        "SecurityToken",
        "security_token",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "awsAccessKeyId",
        "AWS_SECRET_ACCESS_KEY",
        "TenantCredentials",
        "tenant-credentials",
        "X-Amz-Security-Token",
        "x-amz-signature",
    ],
)
def test_credential_key_spelling_variants_are_removed(key_spelling: str) -> None:
    """Any separator/casing spelling of a credential field name is removed.

    The field names are enumerated by hand here rather than derived from the
    production set, so a spelling silently dropped from that set fails this test.

    Args:
        key_spelling: One concrete spelling of a credential field name.
    """
    body = {"result": {"payload": {key_spelling: SECRET_ACCESS_KEY, "ok": "keep me"}}}

    scrubbed, removed = scrub_and_assert_clean(body)

    assert key_spelling not in scrubbed["result"]["payload"]
    assert scrubbed["result"]["payload"]["ok"] == "keep me"
    assert removed == 1


def test_boto3_session_kwargs_dict_is_scrubbed() -> None:
    """The boto3 ``Session(**kwargs)`` shape the tool Lambdas build is removed.

    ``tools/common/credentials_context.py`` maps the injected snake_case fields
    onto ``aws_access_key_id`` / ``aws_secret_access_key`` / ``aws_session_token``.
    That dict exists at runtime inside every tool, so it is a realistic thing to
    find echoed in an error payload.
    """
    body = {
        "result": {
            "debug": {
                "aws_access_key_id": ACCESS_KEY_ID,
                "aws_secret_access_key": SECRET_ACCESS_KEY,
                "aws_session_token": SESSION_TOKEN,
                "region_name": "eu-west-1",
            }
        }
    }

    scrubbed, removed = scrub_and_assert_clean(body)

    assert scrubbed["result"]["debug"] == {"region_name": "eu-west-1"}
    assert removed == 3


def test_credential_key_with_zero_width_noise_is_removed() -> None:
    """Zero-width padding inside a key name does not defeat the name match."""
    body = {"result": {"access\u200bkey\u200bid": ACCESS_KEY_ID}}

    scrubbed, _ = scrub_and_assert_clean(body)

    assert scrubbed["result"] == {}


# ---------------------------------------------------------------------------
# Gap A — credentials embedded in a larger string
# ---------------------------------------------------------------------------


def test_access_key_id_embedded_in_prose_is_excised_in_place() -> None:
    """A key ID inside a sentence is excised while the sentence survives.

    The agent is instructed to answer in natural language, so prose is exactly
    where a credential would surface. The old anchored rule was all-or-nothing:
    unless the string WAS the key, nothing happened.
    """
    body = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"assumed the role and received {ACCESS_KEY_ID} "
                        f"with secret {SECRET_ACCESS_KEY} valid for one hour"
                    ),
                }
            ]
        }
    }

    scrubbed, removed = scrub_and_assert_clean(body)

    text = scrubbed["result"]["content"][0]["text"]
    # The surrounding prose is preserved — this is an excision, not a deletion of
    # the whole message.
    assert text.startswith("assumed the role and received")
    assert text.endswith("valid for one hour")
    assert removed == 2


def test_credentials_in_a_log_line_are_excised() -> None:
    """A credential pasted into a log line is caught despite the surrounding text."""
    body = {
        "result": {
            "stderr": (
                "[ERROR] boto3 call failed: "
                f"aws_access_key_id={ACCESS_KEY_ID} region=eu-west-1"
            )
        }
    }

    scrubbed, _ = scrub_and_assert_clean(body)

    assert "region=eu-west-1" in scrubbed["result"]["stderr"]


def test_access_key_id_concatenated_into_a_longer_token_is_excised() -> None:
    """A key ID glued to adjacent characters is still excised.

    Requiring a word boundary would turn simple concatenation into a bypass.
    """
    body = {"result": {"blob": f"prefix{ACCESS_KEY_ID}suffix"}}

    scrubbed, _ = scrub_and_assert_clean(body)

    assert scrubbed["result"]["blob"] == "prefixsuffix"


def test_presigned_url_parameters_are_removed() -> None:
    """SigV4 presigned-URL parameters are stripped; other parameters survive.

    These carry credential material inside a URL string, under no credential key
    name, so no key-name rule can reach them.
    """
    url = (
        "https://example-bucket.s3.eu-west-1.amazonaws.com/doc.pdf"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        f"&X-Amz-Credential={TEMP_ACCESS_KEY_ID}%2F20260805%2Feu-west-1%2Fs3"
        "&X-Amz-Date=20260805T170000Z"
        "&X-Amz-Signature=6f2c1b0d9e8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a"
        f"&X-Amz-Security-Token={SESSION_TOKEN}"
        "&x-id=GetObject"
    )
    body = {"result": {"download_url": url}}

    scrubbed, removed = scrub_and_assert_clean(body)

    cleaned = scrubbed["result"]["download_url"]
    assert "X-Amz-Credential" not in cleaned
    assert "X-Amz-Signature" not in cleaned
    assert "X-Amz-Security-Token" not in cleaned
    # Non-credential parameters and the URL itself are left intact.
    assert cleaned.startswith(
        "https://example-bucket.s3.eu-west-1.amazonaws.com/doc.pdf?"
    )
    assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in cleaned
    assert "x-id=GetObject" in cleaned
    assert removed == 3


def test_presigned_url_keeps_a_wellformed_query_string() -> None:
    """Excising the only query parameter leaves no dangling separator."""
    body = {"result": {"url": f"https://host/key?X-Amz-Signature={ACCESS_KEY_ID}"}}

    scrubbed, _ = scrub_and_assert_clean(body)

    assert scrubbed["result"]["url"] == "https://host/key"


# ---------------------------------------------------------------------------
# Gap A/B — bare secret access keys and session tokens (previously no rule)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("holder_key", ["note", "value", "body", "0"])
def test_bare_secret_access_key_under_any_key_is_removed(holder_key: str) -> None:
    """A secret access key is removed regardless of the key it sits under.

    Previously it was removed ONLY under the literal key ``secret_access_key``;
    the highest-value material had the weakest coverage.

    Args:
        holder_key: An arbitrary non-credential key name.
    """
    body = {"result": {holder_key: SECRET_ACCESS_KEY}}

    scrubbed, removed = scrub_and_assert_clean(body)

    assert scrubbed["result"] == {}
    assert removed == 1


def test_bare_session_token_is_removed() -> None:
    """A session token standing alone under a neutral key is removed."""
    body = {"result": {"opaque": SESSION_TOKEN}}

    scrubbed, removed = scrub_and_assert_clean(body)

    assert scrubbed["result"] == {}
    assert removed == 1


def test_credential_as_a_list_element_is_dropped() -> None:
    """A whole-value credential in a list is dropped, siblings keep their order."""
    body = {"result": ["first", SECRET_ACCESS_KEY, "second", ACCESS_KEY_ID, "third"]}

    scrubbed, removed = scrub_and_assert_clean(body)

    assert scrubbed["result"] == ["first", "second", "third"]
    assert removed == 2


# ---------------------------------------------------------------------------
# Rule 3 — JSON-encoded payload carried in a string (MCP content[].text)
# ---------------------------------------------------------------------------


def test_json_encoded_payload_in_a_text_field_is_structurally_scrubbed() -> None:
    """Credentials inside a JSON-ENCODED string are removed by the key-name rule.

    An MCP tool reply carries its payload as a JSON-encoded string, so without
    parsing it the structural rules never see the payload's keys — a
    ``SecretAccessKey`` field one level of encoding down would be invisible.
    """
    payload = {
        "body": "the quarterly report",
        "scope": "tenant-a",
        "Credentials": {
            "AccessKeyId": TEMP_ACCESS_KEY_ID,
            "SecretAccessKey": SECRET_ACCESS_KEY,
        },
    }
    body = {
        "jsonrpc": "2.0",
        "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
    }

    scrubbed, removed = scrub_and_assert_clean(body)

    decoded = json.loads(scrubbed["result"]["content"][0]["text"])
    assert decoded == {"body": "the quarterly report", "scope": "tenant-a"}
    assert removed == 1


def test_doubly_encoded_json_payload_is_scrubbed() -> None:
    """Two levels of JSON encoding do not hide a credential."""
    inner = json.dumps({"AccessKeyId": ACCESS_KEY_ID, "keep": "yes"})
    body = {"result": {"content": [{"text": json.dumps({"nested": inner})}]}}

    scrubbed, _ = scrub_and_assert_clean(body)

    decoded = json.loads(json.loads(scrubbed["result"]["content"][0]["text"])["nested"])
    assert decoded == {"keep": "yes"}


def test_credential_free_json_encoded_string_round_trips_byte_identical() -> None:
    """A clean JSON-encoded payload is returned byte-for-byte, not re-serialised.

    Guards the deep-equal no-op: re-serialising every JSON string would rewrite
    whitespace and key order and break it, so the original string must be handed
    back untouched when nothing was removed.
    """
    text = '{"body":  "hello",\n  "scope":   "tenant-a"}'
    body = {"result": {"content": [{"type": "text", "text": text}]}}

    scrubbed, removed = scrub(body)

    assert removed == 0
    assert scrubbed == body
    assert scrubbed["result"]["content"][0]["text"] == text


# ---------------------------------------------------------------------------
# Bare top-level string body
# ---------------------------------------------------------------------------


def test_bare_top_level_string_body_is_scrubbed() -> None:
    """A string-typed body is scanned, not waved through as an unscrubbable scalar."""
    scrubbed, removed = scrub(f"the key is {ACCESS_KEY_ID} ok")

    assert_no_credential_survives(scrubbed)
    assert removed == 1
    assert scrubbed == "the key is  ok"


def test_bare_top_level_non_string_scalars_are_unchanged() -> None:
    """Non-string scalars cannot carry a credential shape and are untouched."""
    for value in (None, True, 0, 17, 3.5):
        scrubbed, removed = scrub(value)
        assert scrubbed == value
        assert removed == 0


# ---------------------------------------------------------------------------
# Negative cases — the false-positive boundary the heuristics must respect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "value"),
    [
        # 40 chars of lowercase hex — a SHA-1 digest, single-case, no uppercase.
        ("sha1 digest", "356a192b7913b04c54574d18c28d46e6395428ab"),
        # 40 chars, all lowercase letters.
        ("lowercase run", "abcdefghijklmnopqrstuvwxyzabcdefghijklmn"),
        # 40 chars, all uppercase letters.
        ("uppercase run", "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMN"),
        # 40 chars, digits only.
        ("digit run", "1234567890" * 4),
        # 39 chars — one short of the secret-key length.
        ("39 mixed chars", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKE"),
        # A near-miss access-key prefix.
        ("wrong key prefix", "BKIAIOSFODNN7EXAMPLE"),
        # Right prefix, lowercase tail — real key IDs are uppercase-alnum.
        ("lowercase key tail", "AKIAiosfodnn7examplex"),
        # A UUID: separators break every credential run.
        ("uuid", "3f2504e0-4f89-11d3-9a0c-0305e82c3301"),
        # An ARN, which contains long alnum stretches but no credential shape.
        ("arn", "arn:aws:iam::111122223333:role/tenant-a-reader"),
        # A long single-case base64-ish run, well past the session-token floor.
        ("long lowercase run", "abcdefghij" * 40),
    ],
)
def test_non_credential_strings_survive_untouched(
    description: str, value: str
) -> None:
    """Legitimate content that merely resembles a credential is not excised.

    These pin the heuristics' false-positive boundary. Tool replies carry
    arbitrary document bodies (``tools/read_document`` returns the stored ``body``
    verbatim), so a heuristic that shreds hex digests, identifiers or ordinary
    prose would corrupt real content.

    Args:
        description: Human-readable label for the failure message.
        value: A non-credential string that must round-trip unchanged.
    """
    body = {"result": {"content": [{"type": "text", "text": value}]}}

    scrubbed, removed = scrub(body)

    assert removed == 0, f"{description} was wrongly treated as a credential"
    assert scrubbed == body


def test_secret_key_window_inside_a_longer_base64_blob_is_not_excised() -> None:
    """A 40-char window inside a longer base64 blob is not mistaken for a secret.

    The secret-key rule is length-exact and boundary-bounded precisely so an
    embedded image or attachment does not get sliced apart. (A blob long enough
    to look like a session token is a separate, deliberate case — see
    :func:`test_very_long_base64_run_is_excised_as_a_session_token`.)
    """
    blob = "aB3xY9" * 12  # 72 chars, mixed-class, well short of the token floor
    body = {"result": {"attachment": blob}}

    scrubbed, removed = scrub(body)

    assert removed == 0
    assert scrubbed["result"]["attachment"] == blob


def test_very_long_base64_run_is_excised_as_a_session_token() -> None:
    """A very long mixed-class base64 run IS excised — the documented trade-off.

    AWS states that the size of an STS security token "is not fixed" and warns
    against assuming a maximum, so the session-token heuristic has no upper length
    bound; capping it would create a bypass AWS's own documentation predicts. The
    cost is that a long mixed-case-plus-digits base64 blob in legitimate content
    is excised too.

    This test exists so that behaviour is a recorded decision rather than a
    surprise: if it ever starts failing, someone has added a length ceiling and
    needs to justify it against the AWS guidance above.
    """
    blob = "aB3xY9" * 200  # 1200 chars, mixed-class
    body = {"result": {"attachment": blob}}

    scrubbed, removed = scrub(body)

    assert removed == 1
    assert "attachment" not in scrubbed["result"]


# ---------------------------------------------------------------------------
# Counting semantics
# ---------------------------------------------------------------------------


def test_removed_count_counts_each_excised_occurrence() -> None:
    """Two credentials in one string count as two removals."""
    body = {"note": f"{ACCESS_KEY_ID} and {TEMP_ACCESS_KEY_ID}"}

    _, removed = scrub_and_assert_clean(body)

    assert removed == 2


def test_removed_count_counts_a_credential_key_subtree_once() -> None:
    """Deleting a credential key counts once, however large its subtree."""
    body = {
        "context": {
            "served_scope": "tenant-a",
            "tenant_credentials": {
                "access_key_id": ACCESS_KEY_ID,
                "secret_access_key": SECRET_ACCESS_KEY,
                "session_token": SESSION_TOKEN,
            },
        }
    }

    scrubbed, removed = scrub_and_assert_clean(body)

    assert scrubbed == {}
    assert removed == 1
