"""No module may take a credential from the request BODY, or describe it as the
live channel.

This scans SOURCE rather than behaviour, on purpose. A behavioural test can only
show that the body path is not taken on the inputs it happens to try; scanning
shows the path does not exist to be taken. It is the same reason the tool-side
readers no longer accept an ``event`` parameter: the absence should be a property
of the code, not of the test suite's imagination.

What is asserted:

* the retired ``tenant_credentials`` key name appears in no scanned module except
  as prose explaining that it is retired;
* ``build_tenant_context`` — the helper that packaged credentials into a body
  object — is gone, and nothing re-adds a function by that name;
* no tool handler reads a credential-looking key off its ``event``;
* **no comment or docstring describes the in-body channel in the PRESENT TENSE.**

Why that last one exists
------------------------
The first version of this file scanned two directories (``interceptor/`` and
``tools/``) for ONE string (``tenant_credentials``). That is too narrow twice
over, and both gaps were real: comments in ``interceptor/handler.py``,
``interceptor/audit_record.py`` and ``cdk/gateway_resources.py`` still presented
``arguments["context"]`` as the channel in use, and the CDK was not scanned at
all. Stale prose of that kind is not cosmetic — it is what a reviewer reads to
learn how the system works, and here it described a design the code no longer
implements while sounding authoritative.

The check is phrase-based rather than keyword-based for the same reason: the word
``context`` is legitimate throughout (the Lambda ``context`` parameter,
``client_context``, the propagated-headers key), so forbidding the WORD would be
noise. What is forbidden is a phrase that asserts the body carries the
credentials now.

The RESPONSE scrubber is deliberately exempt from the key-name rule: it still
lists the retired spellings because it guards output a tool produces, which this
refactor does not control.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories carrying the request path, PLUS the CDK that publishes the tool
#: schemas and the header allowlist. The CDK is included because the schema
#: comments are where a reader looks to learn what reaches a tool.
_SCANNED_DIRS = (
    _REPO_ROOT / "interceptor",
    _REPO_ROOT / "tools",
    _REPO_ROOT / "cdk",
)

#: Files allowed to NAME the retired key, because their whole job is to say it is
#: retired. Anything else naming it is a live reference.
_PROSE_EXEMPT = {
    "credentials_context.py",  # module docstring records what it stopped reading
    "scoped_credentials.py",  # AWS doc references + the deletion note
    "audit_record.py",  # explains which key it must never log
    # Found by WIDENING this scan to cdk/: its HISTORY block names the retired key
    # to record what the APPLICATION_LOGS prohibition used to be about. That is
    # the history the next reader needs, so it is exempt rather than rewritten.
    "gateway_wiring.py",
}

#: Phrases that assert the body IS the credential channel. Each is matched
#: case-insensitively. A phrase in the PAST tense ("used to ride", "rode",
#: "retired", "no longer") is legitimate and must not match, which is why these
#: are anchored on present-tense verbs.
_PRESENT_TENSE_BODY_CLAIMS = (
    r"are\s+injected\s+by\s+the\s+REQUEST\s+interceptor",
    r"injects\s+(it|them|the\s+\w+)\s+into\s+the\s+tool\s+arguments",
    r"adds\s+exactly\s+one\s+new\s+key",
    r"arrive\s+via\s+the\s+REQUEST\s+interceptor\s+at",
    r"the\s+``?context``?\s+object\s+is\s+injected",
    r"BEFORE\s+the\s+interceptor\s+writes\s+its\s+``?context``?\s+object",
)


def _python_sources() -> list[Path]:
    """Return every scanned Python source, excluding caches and synth output."""
    found: list[Path] = []
    for directory in _SCANNED_DIRS:
        found.extend(
            path
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts and "cdk.out" not in path.parts
        )
    return found


def test_sources_were_found() -> None:
    """Guard against a vacuous scan.

    Without this, a wrong path would make every assertion below pass over an empty
    list and report a property nobody measured.
    """
    sources = _python_sources()
    assert len(sources) >= 12, [str(p) for p in sources]


def test_the_cdk_is_actually_in_scope() -> None:
    """The widened scope is real, not just declared.

    The original gap was that ``cdk/`` was never scanned. Asserting the directory
    is in the list would prove nothing if the glob missed it, so this names a file
    that must be present.
    """
    names = {p.name for p in _python_sources()}
    assert "gateway_resources.py" in names, sorted(names)


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: p.name)
def test_no_live_reference_to_the_retired_body_key(path: Path) -> None:
    """``tenant_credentials`` appears in no module outside explanatory prose."""
    if path.name in _PROSE_EXEMPT:
        pytest.skip(f"{path.name} names the retired key only to explain it")
    assert "tenant_credentials" not in path.read_text(encoding="utf-8"), path


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: p.name)
def test_no_comment_presents_the_body_as_the_live_channel(path: Path) -> None:
    """No prose asserts, in the present tense, that credentials ride in the body.

    This is the assertion the narrow first version lacked. It caught nothing on
    the day it was written only because the stale comments had just been fixed;
    its job is to stop them coming back, and to fail on a NEW module that
    describes the retired design.
    """
    source = path.read_text(encoding="utf-8")
    offenders = [
        pattern
        for pattern in _PRESENT_TENSE_BODY_CLAIMS
        if re.search(pattern, source, re.IGNORECASE)
    ]
    assert offenders == [], (
        f"{path.name} describes the in-body credential channel as current: "
        f"{offenders}. The credentials travel as propagated headers; say so, or "
        "put the claim in the past tense if it is history."
    )


def test_the_present_tense_check_has_teeth() -> None:
    """A phrase check that matches nothing would be indistinguishable from none.

    Each pattern is run against a string that MUST match it, so a regex typo
    (an escaped delimiter that cannot survive, a tightened quantifier) is caught
    here rather than discovered the next time a stale comment slips through.
    """
    must_match = {
        r"are\s+injected\s+by\s+the\s+REQUEST\s+interceptor": (
            "the vended tenant_credentials are injected by the REQUEST interceptor"
        ),
        r"injects\s+(it|them|the\s+\w+)\s+into\s+the\s+tool\s+arguments": (
            "and injects it into the tool arguments for the scoped tool set"
        ),
        r"adds\s+exactly\s+one\s+new\s+key": (
            "the interceptor adds exactly one new key, `context` (below)"
        ),
        r"arrive\s+via\s+the\s+REQUEST\s+interceptor\s+at": (
            'the scope and credentials arrive via the REQUEST interceptor at '
            'arguments["context"]'
        ),
        r"the\s+``?context``?\s+object\s+is\s+injected": (
            "Tools for which the ``context`` object is injected authoritatively"
        ),
        r"BEFORE\s+the\s+interceptor\s+writes\s+its\s+``?context``?\s+object": (
            "as the model supplied them, BEFORE the interceptor writes its "
            "``context`` object"
        ),
    }
    assert set(must_match) == set(_PRESENT_TENSE_BODY_CLAIMS), (
        "every pattern needs a sample that must match it"
    )
    for pattern, sample in must_match.items():
        assert re.search(pattern, sample, re.IGNORECASE), pattern


def test_past_tense_history_is_not_flagged() -> None:
    """Describing the retired design as history must stay legal.

    Several modules explain what the body channel WAS and why it went away. A
    check that flagged those would push authors to delete the history instead of
    marking it, which is the opposite of what this file is for.
    """
    history = (
        "Until this change the vended STS credentials rode in the request BODY, "
        'at params.arguments["context"], on the belief that a Lambda '
        "target has no header channel. Under the earlier in-body design they rode "
        'as an UNDECLARED arguments["context"] field. build_tenant_context used to '
        "assemble the context object the interceptor wrote into the body."
    )
    offenders = [
        p for p in _PRESENT_TENSE_BODY_CLAIMS if re.search(p, history, re.IGNORECASE)
    ]
    assert offenders == [], offenders


def test_build_tenant_context_is_gone() -> None:
    """Nothing defines a helper that packages credentials into a body object.

    Deleted rather than kept unused: a helper that still knows how to do it is a
    path back to the body, which is exactly what the header channel removes.
    """
    for path in _python_sources():
        source = path.read_text(encoding="utf-8")
        assert "def build_tenant_context" not in source, path


@pytest.mark.parametrize(
    "handler_path",
    [
        _REPO_ROOT / "tools" / "read_document" / "handler.py",
        _REPO_ROOT / "tools" / "search_documents" / "handler.py",
        _REPO_ROOT / "tools" / "reply" / "handler.py",
    ],
    ids=["read_document", "search_documents", "reply"],
)
def test_no_tool_reads_a_credential_off_the_event(handler_path: Path) -> None:
    """No tool handler subscripts its event for a credential or the scope.

    The tools read ``event`` only for their declared schema arguments. Anything
    else would be a value the MODEL wrote.
    """
    source = handler_path.read_text(encoding="utf-8")
    for forbidden in (
        'event.get("context")',
        'event["context"]',
        'event.get("served_scope")',
        'event["served_scope"]',
        'event.get("access_key_id")',
        'event["access_key_id"]',
    ):
        assert forbidden not in source, f"{handler_path.name}: {forbidden}"
