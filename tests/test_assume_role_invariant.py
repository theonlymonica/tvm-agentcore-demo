"""Structural invariant: no ``AssumeRole`` may skip either scope gate.

WHY A STRUCTURAL TEST AND NOT A BEHAVIOURAL ONE
-----------------------------------------------
The concern here is not a live bypass — it is that ONE line of code carries the
whole tenant boundary. The minimum that answers such a concern is an enforced
invariant (a test or a lint) that ``AssumeRole`` on these roles is never called
without a session policy. This module is that invariant, extended to also cover
the scope session tag.

The behavioural tests in ``tests/test_scope_tag_abac.py`` prove that the CURRENT
call site passes both ``Policy`` and ``Tags``. They cannot prove anything about a
call site added tomorrow. This module parses every production source file and
fails if ANY ``.assume_role(...)`` call lacks either keyword — so a new caller, a
refactor, or a copy-paste into another module is caught at test time rather than
discovered as cross-tenant access in production.

WHAT IT DOES NOT CATCH (be honest about the hole)
-------------------------------------------------
It is a syntactic check on keyword PRESENCE, not on value correctness. A call
passing ``Tags=[]`` or ``Policy=""`` satisfies this test; the runtime guard for
that is ``interceptor/scoped_credentials._scope_tag_or_raise`` plus the identity
policy's ``Null`` presence checks, and the trust policy rejects the untagged
assume outright. It also cannot see a call built dynamically (``getattr(client,
"assume_role")(**kwargs)``) — the AST walk looks for an attribute call named
``assume_role``. Both are known residual gaps of this approach.

Test directories are excluded on purpose: test fakes legitimately DEFINE
``assume_role`` and drive it through the production helper, so scanning them would
report the fakes rather than real callers.
"""

from __future__ import annotations

import ast
import functools
import os
from typing import Iterator

#: Repository root (this file lives in ``tests/``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Directories scanned for production ``assume_role`` call sites.
_SCANNED_DIRS = ("interceptor", "tools", "cdk", "response_interceptor")

#: Never walked: dependency/build trees and the test suite itself (see docstring).
_EXCLUDED_DIRS = frozenset(
    {".git", ".venv", "__pycache__", "cdk.out", "node_modules", "tests"}
)

#: The STS call under invariant, and the keywords every such call MUST carry.
_ASSUME_ROLE_ATTR = "assume_role"
_REQUIRED_KEYWORDS = ("Policy", "Tags")


def _iter_python_files() -> Iterator[str]:
    """Yield every scanned production ``.py`` file path.

    Yields:
        Absolute paths to Python source files under :data:`_SCANNED_DIRS`.
    """
    for top in _SCANNED_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(_REPO_ROOT, top)):
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
            for filename in filenames:
                if filename.endswith(".py"):
                    yield os.path.join(dirpath, filename)


@functools.cache
def _assume_role_calls() -> tuple[tuple[str, int, frozenset[str]], ...]:
    """Find every ``*.assume_role(...)`` call in production source.

    Returns:
        A tuple of ``(repo-relative path, line number, keyword names)`` — one
        entry per call site found.
    """
    found: list[tuple[str, int, frozenset[str]]] = []
    for path in _iter_python_files():
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == _ASSUME_ROLE_ATTR:
                keywords = frozenset(
                    kw.arg for kw in node.keywords if kw.arg is not None
                )
                found.append(
                    (os.path.relpath(path, _REPO_ROOT), node.lineno, keywords)
                )
    return tuple(found)


def test_at_least_one_call_site_is_found() -> None:
    """Guard the guard: an empty scan would make every assertion below vacuous."""
    assert _assume_role_calls(), (
        "no assume_role call site found — the scan is broken (or the vending path "
        "moved), which would silently disable this invariant"
    )


def test_every_assume_role_passes_a_session_policy_and_a_scope_tag() -> None:
    """Both gates, at every call site, enforced structurally.

    ``Policy`` is the original per-request ``LeadingKeys`` session policy.
    ``Tags`` carries the ``scope`` session tag the roles' identity and trust
    policies require. A call site missing either one either re-opens the
    table-wide access finding (missing ``Policy``) or is rejected by the trust
    policy at runtime (missing ``Tags``).
    """
    offenders = [
        f"{path}:{lineno} missing {sorted(set(_REQUIRED_KEYWORDS) - keywords)}"
        for path, lineno, keywords in _assume_role_calls()
        if not set(_REQUIRED_KEYWORDS).issubset(keywords)
    ]

    assert not offenders, (
        "every sts:AssumeRole on the scoped Documents roles must pass BOTH the "
        "inline LeadingKeys session policy (Policy=) and the scope session tag "
        f"(Tags=). Offending call sites: {offenders}"
    )


def test_vending_remains_a_single_call_site() -> None:
    """One assume path is a security property, not an accident.

    The risk this guards against is a future refactor, an added caller, or an
    exception path. A second call site is not forbidden by itself — but it MUST
    be a deliberate decision, so this test fails and forces the reviewer to look
    at it (and to confirm the new site is covered by the gates above).
    """
    sites = [f"{path}:{lineno}" for path, lineno, _ in _assume_role_calls()]

    assert len(sites) == 1, (
        "credential vending is expected to have exactly ONE sts:AssumeRole call "
        "site (interceptor/scoped_credentials.py). If you are intentionally "
        f"adding another call site, update this test. Found: {sites}"
    )
