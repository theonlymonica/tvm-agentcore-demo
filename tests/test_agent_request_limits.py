"""Tests for the agent's ``/invocations`` request-body bound.

``POST /invocations`` reads the raw request body: the endpoint takes a Starlette
``Request`` (the runtime delivers the payload directly, and two payload shapes are
tolerated), so the ``InvocationRequest`` pydantic model never applies and carries
no size bound. Nothing downstream supplied one either -- uvicorn has no
request-body-size flag -- so ``await request.json()`` buffered a body of any size
inside the container.

``agent/request_limits.py`` is that bound, and it is deliberately stdlib-only:
the agent's own dependencies (fastapi, strands) are not part of the test
toolchain, which is a large part of why ``agent/`` had no tests at all. Keeping
the check in a framework-free module is what makes it directly testable.

Two layers, and the tests cover why each is needed:

* the ``Content-Length`` pre-check is cheap but trusts a client-supplied header;
* the streamed read is the real bound -- it holds when the header is absent
  (a chunked body has none), unparseable, or understated, and it abandons the
  read part-way rather than buffering the whole oversized body first.

The streamed helper is a coroutine driven here with ``asyncio.run`` rather than
an ``async def`` test, because the suite has no pytest-asyncio and this bound is
not worth adding a plugin for.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from request_limits import (
    DEFAULT_MAX_BODY_BYTES,
    MAX_BODY_BYTES_ENV_VAR,
    BodyTooLarge,
    MalformedBody,
    check_content_length,
    max_body_bytes,
    parse_bounded_json,
    read_bounded_body,
)


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    """Yield ``parts`` as an async stream, mimicking ``Request.stream()``."""
    for part in parts:
        yield part


class _CountingStream:
    """An async body stream that records how many chunks were actually pulled.

    The count is the evidence for the claim that matters: an oversized body is
    rejected part-way through, not buffered in full and measured afterwards.
    """

    def __init__(self, chunk: bytes, count: int) -> None:
        self._chunk = chunk
        self._count = count
        self.yielded = 0

    def __aiter__(self) -> "_CountingStream":
        return self

    async def __anext__(self) -> bytes:
        if self.yielded >= self._count:
            raise StopAsyncIteration
        self.yielded += 1
        return self._chunk


def _read(stream: object, limit: int | None = None) -> bytes:
    """Drive ``read_bounded_body`` to completion synchronously."""
    return asyncio.run(read_bounded_body(stream, limit))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The configured limit
# ---------------------------------------------------------------------------


class TestMaxBodyBytes:
    """The limit comes from the environment, with a safe default."""

    def test_unset_env_uses_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(MAX_BODY_BYTES_ENV_VAR, raising=False)

        assert max_body_bytes() == DEFAULT_MAX_BODY_BYTES

    def test_the_default_is_a_sane_size(self) -> None:
        # A prompt, a model id and a JWT. The invariant is the RANGE, not the
        # exact figure: retuning to 128 KiB or 512 KiB is a legitimate call, while
        # a default down at a few KB would reject real prompts and one up in the
        # megabytes would make the bound meaningless.
        assert 64 * 1024 <= DEFAULT_MAX_BODY_BYTES <= 1024 * 1024

    def test_env_override_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MAX_BODY_BYTES_ENV_VAR, "1024")

        assert max_body_bytes() == 1024

    def test_whitespace_around_the_value_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MAX_BODY_BYTES_ENV_VAR, "  2048  ")

        assert max_body_bytes() == 2048

    @pytest.mark.parametrize("value", ["", "   ", "abc", "1.5", "0x10", "1024b"])
    def test_an_unparseable_value_falls_back_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        # A typo in a deploy-time variable must not crash every invocation.
        monkeypatch.setenv(MAX_BODY_BYTES_ENV_VAR, value)

        assert max_body_bytes() == DEFAULT_MAX_BODY_BYTES

    def test_a_digit_separator_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # int() accepts Python's underscore separators, so "65_536" is a real
        # value rather than a typo. Documented rather than fought: it reads
        # clearly in a deploy config and resolves to what it looks like.
        monkeypatch.setenv(MAX_BODY_BYTES_ENV_VAR, "65_536")

        assert max_body_bytes() == 65536

    @pytest.mark.parametrize("value", ["0", "-1", "-999"])
    def test_a_non_positive_value_cannot_switch_the_bound_off(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        # The important one: honouring 0 would reject every request, and honouring
        # a negative would be worse still. Neither may become a way to disable or
        # invert the control from the environment.
        monkeypatch.setenv(MAX_BODY_BYTES_ENV_VAR, value)

        assert max_body_bytes() == DEFAULT_MAX_BODY_BYTES


# ---------------------------------------------------------------------------
# Content-Length pre-check
# ---------------------------------------------------------------------------


class TestCheckContentLength:
    """A declared oversize body is rejected before any of it is read."""

    def test_over_the_limit_raises(self) -> None:
        with pytest.raises(BodyTooLarge):
            check_content_length("1025", limit=1024)

    def test_exactly_the_limit_is_allowed(self) -> None:
        # Boundary: the limit is inclusive, so a body of exactly the maximum size
        # is a valid request rather than the first rejected one.
        check_content_length("1024", limit=1024)

    def test_under_the_limit_is_allowed(self) -> None:
        check_content_length("10", limit=1024)

    @pytest.mark.parametrize("header", [None, "", "not-a-number", "12x"])
    def test_absent_or_unparseable_headers_pass_to_the_streamed_check(
        self, header: str | None
    ) -> None:
        # Deliberately not an error. The header is client-controlled, so this
        # check can only ever be an optimisation; refusing a request for a
        # malformed header would reject chunked bodies, which carry none at all.
        # The streamed read is what actually enforces the bound.
        check_content_length(header, limit=1024)

    def test_the_error_names_the_limit_but_not_the_declared_size(self) -> None:
        with pytest.raises(BodyTooLarge) as excinfo:
            check_content_length("999999", limit=1024)

        message = str(excinfo.value)
        assert "1024" in message
        # Echoing the observed size back tells a prober how far it got.
        assert "999999" not in message
        assert excinfo.value.limit == 1024

    def test_the_env_limit_applies_when_none_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MAX_BODY_BYTES_ENV_VAR, "100")

        with pytest.raises(BodyTooLarge):
            check_content_length("101")


# ---------------------------------------------------------------------------
# Streamed read -- the load-bearing bound
# ---------------------------------------------------------------------------


class TestReadBoundedBody:
    """The bytes actually read are capped, whatever the headers claimed."""

    def test_a_small_body_is_returned_intact(self) -> None:
        body = _read(_chunks(b'{"prompt":', b' "hello"}'), 1024)

        assert body == b'{"prompt": "hello"}'

    def test_an_empty_body_is_returned_as_empty_bytes(self) -> None:
        assert _read(_chunks(), 1024) == b""

    def test_empty_chunks_are_skipped(self) -> None:
        # Starlette's stream ends with an empty chunk; it must not be mistaken for
        # content or otherwise disturb the total.
        assert _read(_chunks(b"ab", b"", b"cd", b""), 1024) == b"abcd"

    def test_exactly_the_limit_is_accepted(self) -> None:
        # Boundary: the last byte that fits must not be the first byte rejected.
        assert _read(_chunks(b"x" * 1024), 1024) == b"x" * 1024

    def test_one_byte_over_the_limit_is_rejected(self) -> None:
        with pytest.raises(BodyTooLarge):
            _read(_chunks(b"x" * 1025), 1024)

    def test_the_total_is_bounded_not_the_individual_chunk(self) -> None:
        # Each chunk is well under the limit; only the running total crosses it. A
        # per-chunk check would let an unbounded body through in small pieces,
        # which is exactly how a real streamed upload arrives.
        with pytest.raises(BodyTooLarge):
            _read(_chunks(*[b"x" * 100] * 11), 1024)

    def test_an_oversized_body_is_not_read_to_completion(self) -> None:
        # The point of the bound. 1000 chunks of 100 bytes would be ~100 KB
        # buffered; the read must abort at the chunk that crosses 1024 bytes --
        # the 11th -- and never pull the rest.
        stream = _CountingStream(b"x" * 100, count=1000)

        with pytest.raises(BodyTooLarge):
            _read(stream, 1024)

        assert stream.yielded == 11

    def test_a_body_with_no_content_length_is_still_bounded(self) -> None:
        # A chunked request carries no Content-Length, so the pre-check passes it
        # through untouched. If the streamed cap were missing, this is the shape
        # that would go unbounded.
        check_content_length(None, limit=1024)

        with pytest.raises(BodyTooLarge):
            _read(_chunks(b"x" * 4096), 1024)

    def test_a_lying_content_length_does_not_defeat_the_bound(self) -> None:
        # An understated header sails past the pre-check. The streamed read is
        # what catches the body that actually arrives.
        check_content_length("10", limit=1024)

        with pytest.raises(BodyTooLarge):
            _read(_chunks(b"x" * 5000), 1024)

    def test_the_env_limit_applies_when_none_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MAX_BODY_BYTES_ENV_VAR, "16")

        with pytest.raises(BodyTooLarge):
            _read(_chunks(b"x" * 17))

    def test_the_default_limit_admits_a_realistic_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Guards against over-tightening: a long prompt plus a JWT-sized token
        # must still be a valid request, or the bound breaks the product.
        monkeypatch.delenv(MAX_BODY_BYTES_ENV_VAR, raising=False)
        payload = (
            b'{"prompt": "' + b"a" * 8000 + b'", "user_jwt": "' + b"e" * 2000 + b'"}'
        )

        assert _read(_chunks(payload)) == payload


# ---------------------------------------------------------------------------
# The sequence the endpoint delegates to
# ---------------------------------------------------------------------------


class TestParseBoundedJson:
    """Bound, parse and shape-check in one call -- the endpoint's whole body path.

    ``main.py`` cannot be imported in this venv (fastapi is not a test
    dependency), so anything left inline in the endpoint is untestable. Pulling
    the sequence in here leaves the endpoint a two-branch mapping of these
    exception types onto 413 / 400, and puts the behaviour under test.
    """

    def _parse(
        self, content_length: str | None, *parts: bytes, limit: int | None = None
    ) -> dict[str, object]:
        """Drive ``parse_bounded_json`` to completion synchronously."""
        return asyncio.run(
            parse_bounded_json(content_length, _chunks(*parts), limit)
        )

    def test_a_valid_object_is_returned_parsed(self) -> None:
        body = self._parse(None, b'{"prompt": "hello", "user_jwt": "abc"}', limit=1024)

        assert body == {"prompt": "hello", "user_jwt": "abc"}

    def test_a_body_split_across_chunks_is_reassembled(self) -> None:
        # The streamed read must not corrupt a payload that arrives in pieces --
        # the normal case for anything larger than one network buffer.
        body = self._parse(None, b'{"prompt":', b' "hel', b'lo"}', limit=1024)

        assert body == {"prompt": "hello"}

    def test_an_oversized_body_raises_body_too_large(self) -> None:
        # The 413 branch.
        with pytest.raises(BodyTooLarge):
            self._parse(None, b'{"prompt": "' + b"x" * 2000 + b'"}', limit=1024)

    def test_a_declared_oversize_body_raises_before_reading(self) -> None:
        # The pre-check fires on the header alone, so the content is irrelevant --
        # note the body here would have fit.
        with pytest.raises(BodyTooLarge):
            self._parse("99999", b'{"prompt": "hi"}', limit=1024)

    @pytest.mark.parametrize(
        "payload",
        [b"", b"   ", b"{not json", b'{"prompt": "unterminated', b"undefined"],
    )
    def test_unparseable_bodies_raise_malformed_body(self, payload: bytes) -> None:
        # The 400 branch. An EMPTY body belongs here rather than being read as an
        # object with no fields -- json.loads(b"") raises, and a request with no
        # body is malformed, not a request with no prompt.
        with pytest.raises(MalformedBody, match="not valid JSON"):
            self._parse(None, payload, limit=1024)

    @pytest.mark.parametrize(
        "payload", [b"[]", b'["prompt"]', b'"a string"', b"42", b"null", b"true"]
    )
    def test_valid_json_that_is_not_an_object_raises_malformed_body(
        self, payload: bytes
    ) -> None:
        # Valid JSON, wrong shape. Without this guard a bare list or scalar would
        # reach body.get() and fail as a 500 rather than a 400 -- the endpoint used
        # to rely on the framework for a guarantee it no longer gets.
        with pytest.raises(MalformedBody, match="must be a JSON object"):
            self._parse(None, payload, limit=1024)

    def test_an_empty_json_object_is_accepted(self) -> None:
        # Shape is this function's concern; the endpoint's own prompt check is
        # what rejects an object with nothing useful in it.
        assert self._parse(None, b"{}", limit=1024) == {}

    def test_the_two_failure_modes_are_distinguishable(self) -> None:
        # The endpoint maps these onto different status codes, so they must not
        # collapse into one type.
        assert not issubclass(MalformedBody, BodyTooLarge)
        assert not issubclass(BodyTooLarge, MalformedBody)

    def test_the_malformed_message_names_no_body_content(self) -> None:
        # The detail is returned to the caller. It may say what was wrong; it must
        # not echo the payload back.
        with pytest.raises(MalformedBody) as excinfo:
            self._parse(None, b'{"secret": "SHOULD-NOT-APPEAR"}garbage', limit=1024)

        assert "SHOULD-NOT-APPEAR" not in str(excinfo.value)

    def test_the_env_limit_applies_when_none_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MAX_BODY_BYTES_ENV_VAR, "16")

        with pytest.raises(BodyTooLarge):
            self._parse(None, b'{"prompt": "a longer body than sixteen bytes"}')

    def test_a_lying_content_length_does_not_defeat_the_bound(self) -> None:
        # Understated header, oversized body: the pre-check passes it and the
        # streamed cap catches it. This is the shape a bypass attempt takes.
        with pytest.raises(BodyTooLarge):
            self._parse("10", b'{"prompt": "' + b"x" * 5000 + b'"}', limit=1024)


# ---------------------------------------------------------------------------
# The module has to reach the container
# ---------------------------------------------------------------------------


class TestDockerfileShipsTheModule:
    """A bound that is not in the image is not a bound."""

    @staticmethod
    def _copied_python_files(dockerfile: str) -> set[str]:
        """Return every ``.py`` source token named on a ``COPY`` line.

        Parsed by tokens rather than a single regex so the assertion survives the
        forms a Dockerfile legitimately takes: ``COPY --chown=app:app x.py ./``
        (which pairs with the non-root user work in flight), a multi-source
        ``COPY a.py b.py ./``, and a glob. The last token is the destination and
        flags are skipped; anything else ending in ``.py`` counts as copied.
        """
        copied: set[str] = set()
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if not stripped.startswith("COPY "):
                continue
            # Drop the COPY keyword and the destination, leaving the sources.
            sources = stripped.split()[1:-1]
            for token in sources:
                if token.startswith("--"):
                    continue
                if token == "." or token.endswith("*.py"):
                    # A whole-directory or glob copy carries every module.
                    return {"*"}
                if token.endswith(".py"):
                    copied.add(token)
        return copied

    def test_every_agent_module_is_copied_into_the_image(self) -> None:
        # agent/Dockerfile copies source files one COPY at a time, so a new module
        # is silently left out -- and main.py's import of it fails at container
        # start, after a deploy. This asserts the list stays complete.
        agent_dir = Path(__file__).resolve().parent.parent / "agent"
        dockerfile = (agent_dir / "Dockerfile").read_text(encoding="utf-8")

        copied = self._copied_python_files(dockerfile)
        present = {path.name for path in agent_dir.glob("*.py")}

        assert present, "no agent modules found -- the test is looking in the wrong place"
        if copied == {"*"}:
            # A glob or whole-directory COPY: every module ships by construction,
            # so there is no list to fall out of date.
            return
        assert present <= copied, f"not copied into the image: {sorted(present - copied)}"

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("COPY main.py ./", {"main.py"}),
            ("COPY --chown=app:app main.py ./", {"main.py"}),
            ("COPY --from=builder main.py /app/main.py", {"main.py"}),
            ("COPY main.py agent_core.py ./", {"main.py", "agent_core.py"}),
            ("COPY requirements.txt ./", set()),
            ("COPY *.py ./", {"*"}),
            ("COPY . ./", {"*"}),
        ],
    )
    def test_the_copy_parser_handles_the_forms_a_dockerfile_takes(
        self, line: str, expected: set[str]
    ) -> None:
        # The guard above is only worth having if it does not fire spuriously.
        # A --chown flag added alongside a non-root USER must not read as "the
        # module is missing".
        assert self._copied_python_files(line) == expected
