"""Root pytest configuration: import-path setup and shared Hypothesis profiles.

This repository has no single source root — Python code spans ``tools/``,
``interceptor/``, ``shared/``, ``cdk/`` and ``agent/``. The two subsystems
under test use two different import conventions:

* the interceptor imports its own modules package-style
  (``from interceptor.tool_classifier import ...``), which needs the repository
  root on ``sys.path``;
* the tool handlers import their shared helpers flat
  (``from common.scoped_credentials import ...``), which needs the ``tools/``
  directory on ``sys.path``; and
* the RESPONSE interceptor (a zip Lambda whose contents sit at the archive root,
  handler ``handler.handler``) imports its scrubber flat
  (``from credential_scrubber import scrub``), which needs the
  ``response_interceptor/`` directory on ``sys.path``; and
* the agent container, whose modules sit together at ``/app`` and import each
  other flat (``from request_limits import ...``), which needs the ``agent/``
  directory on ``sys.path``.

To let the test suite import each subsystem exactly as it imports itself at
runtime, this root ``conftest.py`` prepends the repository root, the ``tools/``
directory, the ``response_interceptor/`` directory and the ``agent/`` directory
to ``sys.path`` before any test module is collected.

It also registers reusable Hypothesis profiles for the property-based tests
that later tasks add. The Hypothesis import is optional so that the path setup
still applies even in an environment where ``hypothesis`` is not yet installed.

See ``requirements-dev.txt`` for the test dependencies and how to install them
into a Python 3.14 virtual environment.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Import-path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
_RESPONSE_INTERCEPTOR_DIR = os.path.join(_REPO_ROOT, "response_interceptor")
_AGENT_DIR = os.path.join(_REPO_ROOT, "agent")


def _prepend_sys_path(path: str) -> None:
    """Prepend ``path`` to ``sys.path`` if it exists and is not already present.

    Args:
        path: Absolute filesystem path to add to the front of ``sys.path``.
    """
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


# Repository root first so ``interceptor.*``, ``tools.*`` and ``shared.*`` all
# resolve as packages; then ``tools/`` so the handlers' own ``common.*`` imports
# resolve exactly as they do inside the Lambda runtime;
# ``response_interceptor/`` so the RESPONSE interceptor's flat
# ``from credential_scrubber import scrub`` (and ``import handler``) resolve as
# they do at the zip Lambda's archive root; and ``agent/`` so the agent
# container's flat sibling imports (``from request_limits import ...``, as
# ``main.py`` does at /app) resolve the same way here. Only the agent's
# framework-free modules are importable in the test venv -- ``main.py`` and
# ``agent_core.py`` need fastapi / strands, which the test toolchain does not
# install.
_prepend_sys_path(_AGENT_DIR)
_prepend_sys_path(_RESPONSE_INTERCEPTOR_DIR)
_prepend_sys_path(_TOOLS_DIR)
_prepend_sys_path(_REPO_ROOT)


# ---------------------------------------------------------------------------
# Hypothesis profiles (optional import — path setup above must never depend on it)
# ---------------------------------------------------------------------------

try:
    from hypothesis import HealthCheck, settings

    # "ci" — deterministic, more examples, function-scoped-fixture health check
    # relaxed so property tests may reuse the moto/boto fixtures defined in
    # tests/conftest.py without Hypothesis complaining.
    settings.register_profile(
        "ci",
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    # "dev" — fast local feedback (fewer examples).
    settings.register_profile(
        "dev",
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    # Default profile is selected via the HYPOTHESIS_PROFILE env var, falling
    # back to "dev" for quick local runs.
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
except ImportError:  # pragma: no cover - hypothesis not installed yet
    # The suite's import-path setup is the load-bearing part of this file and
    # must succeed regardless of whether Hypothesis is present.
    pass
