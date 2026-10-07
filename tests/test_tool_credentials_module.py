"""Module-absence unit tests for the tool-side scoped-credentials module.

Task 10.5 — proves the dead session-policy code that task 10.1 removed from the
*tool-side* module ``tools/common/scoped_credentials.py`` is truly gone. Under
the header wire contract the tool no longer assembles its own
STS session policy: the REQUEST interceptor performs ``sts:AssumeRole`` with the
inline ``dynamodb:LeadingKeys`` policy and hands the tool short-lived, partition-
confined credentials via the injected ``context`` object. The tool therefore has
no use for a policy builder or the read/write action lists, and this module
requires that those identifiers no longer appear in the tool-side module.

The module under test is imported exactly as the tool handlers import it at
runtime — flat, ``import common.scoped_credentials`` — which the root
``conftest.py`` enables by prepending the ``tools/`` directory to ``sys.path``.

Two concerns are covered:

1. **Forbidden identifiers absent (Req 8.4).** ``build_session_policy``,
   ``READ_ACTIONS`` and ``WRITE_ACTIONS`` are not module attributes
   (``hasattr`` is ``False`` for each), are absent from the module's ``__all__``
   export list, and — as a secondary, source-level guard — are not *defined*
   anywhere in the module (no function definition, module-level assignment, or
   re-import binds those names). The module file is located via
   ``common.scoped_credentials.__file__`` so no absolute path is hardcoded.

2. **Live tool-side API still present (positive control).** Removing the dead
   code must not have taken the real, in-use API with it: the module still
   exposes ``documents_table_from_event``, ``served_scope_from_event`` and the
   ``ScopedCredentialsError`` class the handlers depend on.

AWS grounding (verified via the AWS Documentation MCP server, per the workspace
``aws-docs-lookup`` rule): the session-policy construction that once lived here
belongs to STS ``AssumeRole`` with an inline session policy — the vended session
is the INTERSECTION of the role identity policy and that inline policy — which is
now performed in the interceptor, not the tool:
https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_assumerole.html
"""

from __future__ import annotations

import ast

import common.scoped_credentials as tool_creds

# Identifiers that task 10.1 removed from the tool-side module and that
# this module forbids from reappearing. None of these identifiers match the
# task 13.1 static-grep acceptance patterns, so naming them here is safe.
_FORBIDDEN_IDENTIFIERS = ("build_session_policy", "READ_ACTIONS", "WRITE_ACTIONS")

# The live tool-side API the handlers still depend on — a positive control so a
# future over-eager cleanup that also deletes the real API is caught here.
#
# Inverted for the F9 header channel: the readers now take the Lambda CONTEXT
# rather than the event, and were RENAMED to say so. The old names are pinned
# ABSENT just below, because a reader that still accepts an event is a path by
# which a model-written body field could supply a credential — the exact thing the
# header channel removes.
_LIVE_API = (
    "documents_table_from_context",
    "served_scope_from_context",
    "ScopedCredentialsError",
)

#: Names that MUST NOT come back. Asserted, not merely deleted: a reinstated
#: event-based reader would restore the body path in silence.
_RETIRED_API = (
    "documents_table_from_event",
    "served_scope_from_event",
    "context_credentials_from_event",
    "validated_context",
)


def _module_source() -> str:
    """Read the tool-side scoped-credentials module source from disk.

    Uses ``common.scoped_credentials.__file__`` so no absolute path is hardcoded.

    Returns:
        The full text of ``tools/common/scoped_credentials.py``.
    """
    with open(tool_creds.__file__, "r", encoding="utf-8") as handle:
        return handle.read()


def _defined_names(source: str) -> set[str]:
    """Return every name the module *defines or binds* at any level.

    Collects the names introduced by function/class definitions, assignment
    targets (plain and annotated), and ``import`` / ``from ... import`` aliases.
    This targets this test's real subject — the identifiers as code
    constructs — rather than incidental mentions inside docstrings or comments,
    which the AST does not surface as defined names.

    Args:
        source: Python source text of the module under test.

    Returns:
        The set of all defined/bound identifier names found in ``source``.
    """
    tree = ast.parse(source)
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_assignment_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_assignment_target_names(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])

    return names


def _assignment_target_names(target: ast.AST) -> set[str]:
    """Return the identifier names bound by an assignment target.

    Handles a bare ``Name`` target as well as tuple/list unpacking targets so
    that a name introduced via ``A, B = ...`` is detected.

    Args:
        target: The AST node of an assignment target.

    Returns:
        The set of names bound by ``target`` (empty for unsupported targets such
        as attribute or subscript assignments, which cannot bind a bare name).
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        found: set[str] = set()
        for element in target.elts:
            found.update(_assignment_target_names(element))
        return found
    return set()


def test_forbidden_identifiers_not_module_attributes() -> None:
    """``build_session_policy`` / ``READ_ACTIONS`` / ``WRITE_ACTIONS`` are absent.

    The primary assertion: none of the removed identifiers resolve as attributes
    of the imported tool-side module.

    """
    for identifier in _FORBIDDEN_IDENTIFIERS:
        assert not hasattr(tool_creds, identifier), (
            f"{identifier!r} should have been removed from "
            f"tools/common/scoped_credentials.py (Req 8.4)"
        )


def test_forbidden_identifiers_absent_from_dunder_all() -> None:
    """The removed identifiers are not re-exported via ``__all__``.

    If the module defines an ``__all__`` list, none of the forbidden identifiers
    may appear in it. (Skipped implicitly — the loop is a no-op — when the module
    declares no ``__all__``.)

    """
    exported = getattr(tool_creds, "__all__", [])
    for identifier in _FORBIDDEN_IDENTIFIERS:
        assert identifier not in exported, (
            f"{identifier!r} must not be exported from "
            f"tools/common/scoped_credentials.py __all__ (Req 8.4)"
        )


def test_forbidden_identifiers_not_defined_in_source() -> None:
    """Secondary source-level guard: the identifiers are not *defined* anywhere.

    Parses the module source (located via ``__file__``, no hardcoded path) and
    asserts none of the forbidden identifiers are introduced by a function/class
    definition, a module-level assignment, or a re-import. Incidental mentions in
    docstrings or comments are ignored because the AST does not treat them as
    defined names.

    """
    defined = _defined_names(_module_source())
    for identifier in _FORBIDDEN_IDENTIFIERS:
        assert identifier not in defined, (
            f"{identifier!r} is still defined in "
            f"tools/common/scoped_credentials.py (Req 8.4)"
        )


def test_live_tool_side_api_still_importable() -> None:
    """Positive control: the in-use tool-side API survived the cleanup.

    Removing the dead session-policy code must not remove the real API the tool
    handlers call: the scope reader, the table builder, and the fail-closed error
    class are all still present on the module.

    """
    for identifier in _LIVE_API:
        assert hasattr(tool_creds, identifier), (
            f"{identifier!r} is part of the live tool-side API and must remain "
            f"importable from tools/common/scoped_credentials.py"
        )


def test_retired_event_based_readers_are_gone() -> None:
    """The must-fail half of the pin above: no event-based reader survives.

    A reader that takes the EVENT can be handed a credential the model wrote into
    the request body, which is precisely the path the header channel exists to
    remove. Its absence is therefore asserted rather than assumed — deleting a
    function is not a guarantee that nobody re-adds one by the same name.
    """
    for identifier in _RETIRED_API:
        assert not hasattr(tool_creds, identifier), (
            f"{identifier!r} reads credentials from the event and was retired with "
            f"the in-body wire contract; re-adding it restores the body path"
        )
