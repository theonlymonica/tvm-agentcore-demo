"""Deterministic Lambda asset packaging.

Every zip-asset Lambda in this stack is packaged from a source directory that is
ALSO an import target for the test suite. ``pytest`` imports those modules,
CPython writes ``__pycache__/*.pyc`` next to them, and ``Code.from_asset`` — which
bundles the directory tree verbatim unless told otherwise — folds that bytecode
into the asset. The asset fingerprint changes, so ``cdk diff`` reports the
function's ``Code.S3Key`` as changed and the next deploy republishes it.

Observed on 2026-08-09: a comment-only edit to three Cedar policies produced a
ten-resource changeset instead of three, because one local ``pytest`` run had
moved the hashes of all five zip Lambdas (the three tool functions share the
``tools/`` bundle, hence a single shared hash) plus the agent container image.

Why this matters more than the wasted republication
---------------------------------------------------
The republication itself is harmless. The damage is to ``cdk diff`` as a
pre-deploy check: once the Lambda hashes move on every run, the habit becomes
"ignore the Lambda lines", and that is precisely the habit under which a REAL
unintended code change ships unnoticed. During the 2026-08-09 deploy this noise
sat in the same diff as a genuine unintended change (a wrong Bedrock model id),
and the noise is what made the diff hard to read.

Secondary cost: the published artifact carried ``.pyc`` files compiled by
whichever local interpreter happened to run the tests. Python ignores bytecode
whose magic number or source mtime does not match, so this was not a runtime
hazard — but it made the artifact non-reproducible, since the same commit
produced different bundles depending on whether someone had run the tests first.

Why a helper rather than an ``exclude=`` at each call site
----------------------------------------------------------
The defect was an omission, and an omission recurs the moment a sixth Lambda is
added. Routing every zip asset through :func:`python_lambda_code` makes the
exclusion the default rather than something each author must remember, and
``tests/test_synth_asset_packaging.py`` fails the build if a bare
``lambda_.Code.from_asset`` reappears anywhere under ``cdk/``.

Container assets share this list
--------------------------------
``ASSET_EXCLUDE`` is also passed to BOTH container assets — the agent runtime
(``AgentRuntimeArtifact.from_asset``, ``cdk/runtime_resources.py``) and the REQUEST
interceptor (``DockerImageCode.from_image_asset``, ``cdk/scoped_credentials_stack.py``) —
which have the same exposure via their staged build contexts.

``interceptor/.dockerignore`` already excluded root-level caches, which is why that
image was the ONE asset that did not churn on 2026-08-09. It is kept (it also trims
``Dockerfile`` from the image layer, which ``exclude`` here does not) and the
``exclude`` is added ALONGSIDE it, because a slash-free ``.dockerignore`` pattern
does not reach a nested cache. Editing that file instead would change its contents
and move the one asset hash that was already correct.

The agent got no ``.dockerignore`` for the same reason: one was written first and
then discarded after measurement — adding ``agent/.dockerignore`` moved that asset's
hash even with the file excluding itself, because any new file in the context
changes the fingerprint, so the fix would have forced exactly the one-time container
rebuild it exists to prevent. Passing ``exclude=`` is hash-neutral, adds no file, and
keeps one source of truth.

Documentation references:
  - aws_cdk.aws_lambda.Code.from_asset / AssetOptions.exclude (glob patterns
    excluded from the bundle, matched against paths relative to the asset root):
    https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_s3_assets/AssetOptions.html
  - CPython bytecode caching (``__pycache__``, invalidation by magic number and
    source mtime):
    https://docs.python.org/3/reference/import.html#cached-bytecode-invalidation
"""

from __future__ import annotations

import aws_cdk.aws_lambda as lambda_

#: Patterns excluded from every asset in this stack — zip bundles AND the two
#: container build contexts.
#:
#: The ``**/`` duplicates are load-bearing, not belt-and-braces. The two asset
#: kinds default to DIFFERENT ignore modes:
#:
#: * zip assets (``Code.from_asset``) default to ``IgnoreMode.GLOB``, where a
#:   slash-free pattern matches at any depth — ``__pycache__`` alone already
#:   covers ``tools/common/__pycache__``;
#: * container assets (``DockerImageAsset``, which backs both
#:   ``AgentRuntimeArtifact.from_asset`` and ``DockerImageCode.from_image_asset``)
#:   default to ``IgnoreMode.DOCKER``, following ``.dockerignore`` semantics, where
#:   a slash-free pattern matches ONLY at the context root.
#:
#: So without the ``**/`` forms, a cache one directory down inside a container
#: context is bundled. That is latent today because ``agent/`` and ``interceptor/``
#: are both flat — the first subpackage added under either would silently reopen
#: the churn on the asset with the most expensive remedy (image rebuild, ECR push,
#: new AgentCore runtime version).
#:
#: Passing ``ignore_mode=IgnoreMode.GLOB`` instead would be the tidier fix and is
#: deliberately NOT used: ``ignore_mode`` IS part of the asset fingerprint
#: (measured: the agent image moves ``a19d810b…`` -> ``c1970e59…``), so it would
#: force exactly the one-time container rebuild this change avoids. ``exclude``
#: patterns are not fingerprinted, so the list can grow for free.
#:
#: ``*.pyc`` / ``*.pyo`` catch stray bytecode written outside a cache directory,
#: and ``.pytest_cache`` catches pytest's scratch directory, which is written at
#: pytest's ROOTDIR (pinned to the repository root by ``pytest.ini``) — not inside
#: any asset root, so it is cheap insurance rather than a fix for something
#: observed.
#:
#: Deliberately NOT excluded: ``*.md``, tests, or anything else non-essential.
#: This list exists to make bundles DETERMINISTIC, not to minimise them —
#: trimming files that are stable across runs would change the asset hashes
#: without buying reproducibility, and every such exclusion is a new way to
#: accidentally omit a module the handler imports at runtime.
ASSET_EXCLUDE = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    # Depth coverage for IgnoreMode.DOCKER (container contexts) — see above.
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
]


def python_lambda_code(directory: str) -> lambda_.Code:
    """Bundle ``directory`` as a Lambda zip asset, excluding build droppings.

    The single construction point for every zip-asset Lambda in this stack, so
    the :data:`ASSET_EXCLUDE` patterns cannot be forgotten at a new call site.

    Args:
        directory: Filesystem path to the asset root — the directory whose
            contents become the archive root (so the handler string is relative
            to this directory, not to the repository root).

    Returns:
        The ``lambda_.Code`` to pass as the function's ``code=``.
    """
    return lambda_.Code.from_asset(directory, exclude=ASSET_EXCLUDE)
