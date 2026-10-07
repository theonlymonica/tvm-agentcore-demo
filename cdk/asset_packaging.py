"""Deterministic Lambda asset packaging.

``Code.from_asset`` bundles a source directory VERBATIM unless told otherwise, so
any ``__pycache__/*.pyc`` CPython leaves next to the handlers gets folded into
the asset. The fingerprint changes, ``cdk diff`` reports the function's
``Code.S3Key`` as changed, and the next deploy republishes it.

The republication is harmless; the damage is to ``cdk diff`` as a pre-deploy
check. Once Lambda hashes move on every local run the habit becomes "ignore the
Lambda lines", and that is the habit under which a real unintended code change
ships unnoticed. A second cost: the published artifact would carry ``.pyc`` files
compiled by whichever local interpreter ran last, making the same commit produce
different bundles.

Routing every zip asset through :func:`python_lambda_code` makes the exclusion
the default rather than something each author must remember. ``ASSET_EXCLUDE`` is
also passed to BOTH container assets — the agent runtime
(``AgentRuntimeArtifact.from_asset``, ``cdk/runtime_resources.py``) and the REQUEST
interceptor (``DockerImageCode.from_image_asset``, ``cdk/scoped_credentials_stack.py``)
— which have the same exposure via their staged build contexts.

``interceptor/.dockerignore`` is kept alongside it: it also trims ``Dockerfile``
from the image layer, which ``exclude`` here does not, and a slash-free
``.dockerignore`` pattern does not reach a nested cache. The agent has no
``.dockerignore`` on purpose — any new file in a build context changes that
asset's fingerprint, so adding one would force the very container rebuild the
exclusion exists to avoid. Passing ``exclude=`` is hash-neutral and adds no file.

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
#: ``*.pyc`` / ``*.pyo`` catch stray bytecode written outside a cache directory.
#:
#: Deliberately NOT excluded: ``*.md`` or anything else non-essential. This list
#: exists to make bundles DETERMINISTIC, not to minimise them — trimming files
#: that are stable across runs would change the asset hashes without buying
#: reproducibility, and every such exclusion is a new way to accidentally omit a
#: module the handler imports at runtime.
ASSET_EXCLUDE = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
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
