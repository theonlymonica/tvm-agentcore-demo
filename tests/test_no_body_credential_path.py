"""No module may take a credential from the request BODY.

This scans SOURCE rather than behaviour, on purpose. A behavioural test can only
show that the body path is not taken on the inputs it happens to try; scanning
shows the path does not exist to be taken. It is the same reason the tool-side
readers no longer accept an ``event`` parameter: the absence should be a property
of the code, not of the test suite's imagination.

What is asserted:

* the retired ``tenant_credentials`` key name appears in no tool or interceptor
  module except as prose explaining that it is retired;
* ``build_tenant_context`` — the helper that packaged credentials into a body
  object — is gone, and nothing re-adds a function by that name;
* no tool handler reads a credential-looking key off its ``event``.

The RESPONSE scrubber is deliberately exempt: it still lists the retired spellings
because it guards output a tool produces, which this refactor does not control.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Modules that carry the request path. The response scrubber is NOT here.
_SCANNED_DIRS = (
    _REPO_ROOT / "interceptor",
    _REPO_ROOT / "tools",
)

#: Files allowed to NAME the retired key, because their whole job is to say it is
#: retired. Anything else naming it is a live reference.
_PROSE_EXEMPT = {
    "credentials_context.py",  # module docstring records what it stopped reading
    "scoped_credentials.py",  # AWS doc references + the deletion note
    "audit_record.py",  # explains which key it must never log
}


def _python_sources() -> list[Path]:
    """Return every request-path Python source, excluding caches."""
    found: list[Path] = []
    for directory in _SCANNED_DIRS:
        found.extend(
            path
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    return found


def test_sources_were_found() -> None:
    """Guard against a vacuous scan.

    Without this, a wrong path would make every assertion below pass over an empty
    list and report a property nobody measured.
    """
    sources = _python_sources()
    assert len(sources) >= 8, [str(p) for p in sources]


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: p.name)
def test_no_live_reference_to_the_retired_body_key(path: Path) -> None:
    """``tenant_credentials`` appears in no module outside explanatory prose."""
    if path.name in _PROSE_EXEMPT:
        pytest.skip(f"{path.name} names the retired key only to explain it")
    assert "tenant_credentials" not in path.read_text(encoding="utf-8"), path


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
