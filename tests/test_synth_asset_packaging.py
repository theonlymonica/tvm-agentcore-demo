"""Asset-determinism guards.

``Code.from_asset`` bundles its source directory verbatim, so the ``__pycache__``
that ``pytest`` writes into every packaged directory landed in the asset. The
fingerprint moved, and ``cdk diff`` reported five Lambda functions and the agent
container image as changed after a test run with no source edit — which is worse
than the wasted republication, because a diff that is always noisy is a diff
nobody reads.

The fix (``cdk/asset_packaging.py`` plus ``exclude=`` on both container assets) is
an omission being filled in, and an omission recurs. These guards are the teeth, in
the repo's established mechanically-enforced-constraint style (compare
``tests/test_synth_log_delivery.py`` for the APPLICATION_LOGS prohibition):

* :class:`TestAssetsExcludeBytecode` synthesizes the REAL stack with bytecode
  caches planted in every directory named by :data:`PACKAGED_SOURCE_DIRS` — at the
  root AND one level down — and asserts no staged asset carries them.
* :class:`TestAssetHashesAreTestRunInvariant` asserts the property the deploy diff
  depends on: planting caches does not move any asset fingerprint.
* :class:`TestSingleAssetConstructionPoint` is the structural guard: a new zip
  Lambda cannot reintroduce a bare ``Code.from_asset`` and quietly reopen the hole.
* :class:`TestGuardHasTeeth` proves the sweep above is not vacuous, by synthesizing
  a stub stack that DOES bundle bytecode and asserting it is flagged.

Scope of the behavioural guards: they cover the directories listed in
:data:`PACKAGED_SOURCE_DIRS`. **Adding a new asset root means adding it to that
tuple** — otherwise the new asset is covered only incidentally (if the suite
happens to import it *and* the environment writes bytecode, which it does not
under ``PYTHONDONTWRITEBYTECODE`` / ``PYTHONPYCACHEPREFIX``).

Why synthesize instead of asserting on the source
-------------------------------------------------
A source-level assertion ("every call passes ``exclude=``") would pass while the
patterns themselves were ineffective. That is not hypothetical: the two asset kinds
default to different ignore modes, so the slash-free ``__pycache__`` pattern reaches
nested caches in a zip bundle but NOT in a container context (see
``cdk/asset_packaging.py``). These tests assert on the STAGED bundle, which is the
artifact that actually ships, and the nested plants are what catch that class of
mistake.
"""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import aws_cdk as cdk
import aws_cdk.aws_lambda as lambda_
import pytest

from synth_helpers import FROZEN_STACK_NAME

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CDK_DIR = _REPO_ROOT / "cdk"

#: Every directory packaged into a deployed artifact that is ALSO an import target
#: of the test suite — i.e. every directory where a ``pytest`` run can leave
#: bytecode a naive asset bundle would pick up. The first four are sub-packages of
#: the shared ``tools/`` bundle (one asset, three Lambdas); ``response_interceptor``
#: and ``cdk/seed`` are zip assets of their own; ``agent`` and ``interceptor`` are
#: the two container build contexts.
#:
#: ``interceptor`` is included even though it has its own
#: ``interceptor/.dockerignore``: that file only reaches root-level caches, so the
#: nested plant below is what covers it.
PACKAGED_SOURCE_DIRS = (
    "tools/common",
    "tools/read_document",
    "tools/reply",
    "tools/search_documents",
    "response_interceptor",
    "cdk/seed",
    "agent",
    "interceptor",
)

#: Subdirectory planted inside each packaged directory so a cache exists ONE LEVEL
#: DOWN as well as at the root. This is the plant that distinguishes the two ignore
#: modes: a slash-free pattern reaches this depth under ``IgnoreMode.GLOB`` (zip)
#: but not under ``IgnoreMode.DOCKER`` (containers).
_NESTED_PROBE_DIR = "nested_cache_probe"

#: Bundle entries that must never appear in a staged asset.
_FORBIDDEN_SUFFIXES = (".pyc", ".pyo")
_FORBIDDEN_DIR = "__pycache__"


@pytest.fixture
def planted_bytecode() -> Iterator[list[Path]]:
    """Plant marker bytecode in every packaged directory, then clean up exactly it.

    Reproduces the state a local ``pytest`` run leaves behind without depending on
    the ambient interpreter having actually written caches — a run under
    ``PYTHONDONTWRITEBYTECODE`` or ``PYTHONPYCACHEPREFIX`` (which some agent and CI
    environments set) writes none, and these guards must still be meaningful there.

    Cleanup is deliberately conservative, because this fixture writes into the real
    source tree and a developer's genuine caches may sit in the same directories:

    * the marker filename is unique per fixture invocation, so two concurrent
      pytest processes cannot delete each other's markers;
    * a pre-existing file is never overwritten (the name makes that near-impossible,
      but the check keeps the guarantee explicit);
    * teardown removes only the marker files it created, then ``rmdir``s only the
      directories it created — an empty-directory removal, never a recursive one,
      so a cache another process created in the meantime is preserved;
    * setup runs INSIDE the ``try``, so an interruption part-way through still
      cleans up what it had already planted.

    Yields:
        The marker file paths that were planted.
    """
    marker_name = f"asset_guard_{os.getpid()}_{uuid4().hex}.cpython-314.pyc"
    created_dirs: list[Path] = []
    created_files: list[Path] = []
    try:
        for relative in PACKAGED_SOURCE_DIRS:
            for parent in (
                _REPO_ROOT / relative,
                _REPO_ROOT / relative / _NESTED_PROBE_DIR,
            ):
                cache_dir = parent / _FORBIDDEN_DIR
                for candidate in (parent, cache_dir):
                    if not candidate.exists():
                        candidate.mkdir(parents=True)
                        created_dirs.append(candidate)
                marker = cache_dir / marker_name
                if marker.exists():  # pragma: no cover — unique name makes this dead
                    raise AssertionError(f"marker collision at {marker}")
                marker.write_bytes(b"# synthetic bytecode planted by the guards\n")
                created_files.append(marker)
        yield created_files
    finally:
        for marker in created_files:
            marker.unlink(missing_ok=True)
        # Deepest-first, so a probe dir empties before its parent is tried.
        for cache_dir in reversed(created_dirs):
            try:
                cache_dir.rmdir()
            except OSError:
                # Not empty: another process put something here. Leave it alone.
                pass


def _synthesize(outdir: Path) -> Path:
    """Synthesize the real ``ToxicFlowStack`` into ``outdir``.

    Args:
        outdir: Cloud-assembly output directory. Assets are STAGED here (one
            ``asset.<hash>`` entry each), which is what these tests inspect —
            ``Template.from_stack`` alone never writes them to disk.

    Returns:
        The path to the cloud assembly directory.
    """
    from toxic_flow_stack import ToxicFlowStack

    app = cdk.App(outdir=str(outdir))
    ToxicFlowStack(
        app,
        FROZEN_STACK_NAME,
        env=cdk.Environment(account="123456789012", region="us-east-1"),
    )
    return Path(app.synth().directory)


def _bytecode_offenders(assembly_dir: Path) -> list[str]:
    """Return every bytecode entry found inside a staged asset.

    Args:
        assembly_dir: A cloud-assembly directory.

    Returns:
        Assembly-relative paths of offending entries. Empty when clean.
    """
    offenders: list[str] = []
    for asset_dir in sorted(
        p for p in assembly_dir.iterdir() if p.is_dir() and p.name.startswith("asset.")
    ):
        for root, dir_names, file_names in os.walk(asset_dir):
            if _FORBIDDEN_DIR in dir_names:
                offenders.append(str(Path(root, _FORBIDDEN_DIR).relative_to(assembly_dir)))
            for name in file_names:
                if name.endswith(_FORBIDDEN_SUFFIXES):
                    offenders.append(str(Path(root, name).relative_to(assembly_dir)))
    return offenders


def _staged_asset_count(assembly_dir: Path) -> int:
    """Count staged asset directories, so a vacuous sweep can be detected.

    Args:
        assembly_dir: A cloud-assembly directory.

    Returns:
        The number of ``asset.<hash>`` directories.
    """
    return sum(
        1 for p in assembly_dir.iterdir() if p.is_dir() and p.name.startswith("asset.")
    )


def _asset_fingerprints(assembly_dir: Path) -> dict[str, str]:
    """Return each asset's fingerprint, keyed by the construct that owns it.

    Keying by the construct path (rather than by asset id) matters for the failure
    message: the asset id IS the fingerprint, so an id-keyed mapping produces a NEW
    key when a hash moves and can only ever report ``None`` for the new value.

    Reads the assembly's ``*.assets.json`` so BOTH asset kinds are covered in one
    mapping: zip ``files`` (which surface as a Lambda ``Code.S3Key``) and
    ``dockerImages`` (which surface as a container image tag). A mapping built from
    ``Code.S3Key`` alone would miss both container images, including the agent's.

    Args:
        assembly_dir: The cloud-assembly directory.

    Returns:
        Mapping of ``"<kind>:<construct-path>"`` to the asset's fingerprint.

    Raises:
        AssertionError: If an asset carries no ``displayName``. Deliberately loud
            rather than falling back to the staged source path — for a container
            asset that path is ``asset.<hash>``, which would silently restore the
            hash-keying this function exists to avoid.
    """
    manifest = json.loads((assembly_dir / f"{FROZEN_STACK_NAME}.assets.json").read_text())
    fingerprints: dict[str, str] = {}
    for kind in ("files", "dockerImages"):
        for asset_id, entry in manifest.get(kind, {}).items():
            source = entry.get("source", {})
            if not (source.get("path") or source.get("directory")):
                continue  # CDK-internal asset with no local source
            label = entry.get("displayName")
            assert label, (
                f"{kind} asset {asset_id[:12]}… has no displayName, so it cannot be "
                "keyed by construct path; update this helper rather than keying by "
                "asset id (the id is the fingerprint)"
            )
            fingerprints[f"{kind}:{label}"] = asset_id
    return fingerprints


class TestAssetsExcludeBytecode:
    """No staged asset may contain Python bytecode (the behavioural guard)."""

    def test_no_staged_asset_contains_bytecode(
        self, planted_bytecode: list[Path], tmp_path: Path
    ) -> None:
        """Synthesize with caches planted at root AND one level down; assert none ship."""
        assert planted_bytecode, "fixture planted nothing — the guard would be vacuous"

        assembly_dir = _synthesize(tmp_path / "assembly")
        assert _staged_asset_count(assembly_dir), "no assets staged — sweep would be vacuous"

        offenders = _bytecode_offenders(assembly_dir)
        assert not offenders, (
            "staged asset bundles contain Python bytecode, so their fingerprints "
            "depend on whether the tests were run before synthesis. "
            "Route zip assets through cdk/asset_packaging.python_lambda_code and "
            "pass exclude=ASSET_EXCLUDE to container assets — note container "
            "contexts need the '**/' patterns to reach nested caches. Offending "
            f"entries: {offenders}"
        )


class TestAssetHashesAreTestRunInvariant:
    """Asset fingerprints must not move when the packaged tree gains bytecode."""

    def test_fingerprints_unchanged_by_planted_bytecode(
        self, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        """Synthesize, plant bytecode, synthesize again, and compare fingerprints.

        Writing bytecode into the packaged directories must not change what the
        next ``cdk diff`` reports.

        Note on the baseline: within a full-suite run, earlier tests have already
        imported the packaged modules, so the first synthesis here is not
        necessarily a pristine tree. What this therefore pins is that planting
        ADDITIONAL caches — including nested ones — moves nothing. That is the
        property that has teeth; a stricter "identical to a pristine checkout"
        assertion is not available from inside the suite that does the importing.
        """
        before = _asset_fingerprints(_synthesize(tmp_path / "before"))
        assert before, "no assets found in the first assembly"

        # Plant only for the second synthesis, so the first is the comparison base.
        request.getfixturevalue("planted_bytecode")
        after = _asset_fingerprints(_synthesize(tmp_path / "after"))

        moved = {
            key: {"before": before[key], "after": after.get(key, "<asset absent>")}
            for key in before
            if before[key] != after.get(key)
        }
        assert not moved, (
            "asset fingerprints moved after bytecode was written into the packaged "
            "directories, so a source-free `pytest` run would make `cdk diff` "
            f"report changed Lambdas / container images. Moved: {moved}"
        )


class TestSingleAssetConstructionPoint:
    """``python_lambda_code`` is the only permitted zip-asset construction point."""

    def test_no_bare_code_from_asset_under_cdk(self) -> None:
        """Assert no module under ``cdk/`` calls ``Code.from_asset`` directly.

        The structural half of the guard. The behavioural tests catch a regression
        only for a call site packaging a directory listed in
        :data:`PACKAGED_SOURCE_DIRS`; this catches ANY new bare call, which is the
        shape the original defect had — five individually-plausible call sites, none
        passing an exclusion.

        The scan is recursive (``rglob``) so ``cdk/seed/*.py`` is covered too, and it
        matches both the attribute form (``lambda_.Code.from_asset``) and the
        directly-imported form (``from aws_cdk.aws_lambda import Code``), which the
        repo's flat-import convention makes a realistic bypass.
        """
        offenders: list[str] = []
        for module_path in sorted(_CDK_DIR.rglob("*.py")):
            if module_path.name == "asset_packaging.py":
                continue  # the single sanctioned call site
            if "cdk.out" in module_path.parts:
                continue  # synthesized output, not source
            tree = ast.parse(module_path.read_text(), filename=str(module_path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "from_asset"):
                    continue
                owner = func.value
                is_attribute_form = isinstance(owner, ast.Attribute) and owner.attr == "Code"
                is_name_form = isinstance(owner, ast.Name) and owner.id == "Code"
                if is_attribute_form or is_name_form:
                    relative = module_path.relative_to(_REPO_ROOT)
                    offenders.append(f"{relative}:{node.lineno}")

        assert not offenders, (
            "bare Code.from_asset call(s) found — these bundle __pycache__ into the "
            "asset and make the fingerprint depend on whether the tests were run. "
            "Use cdk.asset_packaging.python_lambda_code instead. Found "
            f"at: {offenders}"
        )

    def test_exclude_list_carries_the_cache_patterns_at_both_depths(self) -> None:
        """Pin the patterns' intent, including the container-only ``**/`` forms.

        Deliberately narrow: it asserts the required patterns are PRESENT, not that
        the list is exactly this, so adding a pattern is not a test edit.
        Effectiveness is proven by the staged-bundle assertions, not here. The
        ``**/`` entries are pinned because dropping them is silent — it breaks only
        container contexts, and only once a subpackage exists.
        """
        from asset_packaging import ASSET_EXCLUDE, python_lambda_code

        assert callable(python_lambda_code)
        for pattern in ("__pycache__", "*.pyc", "*.pyo", "**/__pycache__", "**/*.pyc"):
            assert pattern in ASSET_EXCLUDE, f"{pattern!r} missing from ASSET_EXCLUDE"


class TestGuardHasTeeth:
    """The sweep rejects a bundle that really does carry bytecode (not vacuous)."""

    def test_sweep_flags_a_bare_from_asset_bundle(self, tmp_path: Path) -> None:
        """Bundle a directory containing ``__pycache__`` with NO exclusion, and
        assert :func:`_bytecode_offenders` reports it.

        Without this, :class:`TestAssetsExcludeBytecode` passing would be equally
        consistent with a sweep that can never fail — the failure mode
        ``tests/test_synth_log_delivery.py::TestGuardHasTeeth`` exists to rule out.
        Uses a throwaway source tree and stack, so it asserts on the mechanism
        rather than on the production stack's (correct) configuration.
        """
        source = tmp_path / "handler_src"
        (source / _FORBIDDEN_DIR).mkdir(parents=True)
        (source / "handler.py").write_text("def handler(event, context):\n    return {}\n")
        (source / _FORBIDDEN_DIR / "handler.cpython-314.pyc").write_bytes(b"stale bytecode\n")

        app = cdk.App(outdir=str(tmp_path / "assembly"))
        stack = cdk.Stack(app, "BareAssetStack")
        lambda_.Function(
            stack,
            "BareAssetFunction",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(source)),  # deliberately unexcluded
        )
        assembly_dir = Path(app.synth().directory)

        offenders = _bytecode_offenders(assembly_dir)
        assert offenders, (
            "the bytecode sweep did not flag a bundle built with a bare "
            "Code.from_asset — it cannot be trusted to catch a real regression"
        )
