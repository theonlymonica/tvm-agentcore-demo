"""The four header names must agree on all THREE sides of the wire.

The interceptor emits them, the tool reads them, and the CDK allowlists them on
every target. The literals are duplicated in three places on purpose — the CDK app
runs from ``cdk/`` with its own virtualenv and cannot import the Lambda packages —
so this test is what makes the duplication safe.

A mismatch fails CLOSED but confusingly: the Gateway DROPS any interceptor-supplied
header the target has not allowlisted, so a rename on one side alone produces a
tool reporting "propagated credential context is missing" and nothing anywhere
saying "a header name disagrees". That is the failure this file exists to turn into
a red test instead of a debugging session.

The synthesized-template check also asserts the allowlist is on ALL THREE targets,
not just the one the original measurement used.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from common.credentials_context import (
    HEADER_TO_SESSION_KWARG,
    SERVED_SCOPE_HEADER as TOOL_SERVED_SCOPE_HEADER,
)
from interceptor.credential_headers import (
    CREDENTIAL_HEADERS,
    MAX_ALLOWED_HEADERS_PER_TARGET,
    SERVED_SCOPE_HEADER as INTERCEPTOR_SERVED_SCOPE_HEADER,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE = _REPO_ROOT / "cdk" / "cdk.out" / "ToxicFlowStack.template.json"

#: The three tool targets. Every one of them must allowlist the headers: each
#: scoped tool is handed its own vended credentials.
_TOOL_TARGET_NAMES = {"ReadDocument", "SearchDocuments", "Reply"}


def _cdk_allowlist() -> list[str]:
    """Return the CDK-side allowlist literal, read from the CDK module's source.

    Imported by exec rather than by ``import``: ``cdk/gateway_target_headers.py``
    sits in a directory that is not a package and is not on this suite's path.
    Reading the literal keeps the assertion about the DEPLOYED value rather than a
    copy of it.

    Returns:
        The ``CREDENTIAL_REQUEST_HEADERS`` tuple as a list.
    """
    module_path = _REPO_ROOT / "cdk" / "gateway_target_headers.py"
    namespace: dict[str, object] = {}
    source = module_path.read_text(encoding="utf-8")
    # Strip the module's own import of `typing`, which is all it needs at import
    # time; exec'ing the whole file is safe because it defines constants and one
    # function and performs no I/O.
    exec(compile(source, str(module_path), "exec"), namespace)  # noqa: S102
    return list(namespace["CREDENTIAL_REQUEST_HEADERS"])  # type: ignore[arg-type]


def test_interceptor_and_tool_agree_on_the_scope_header() -> None:
    assert INTERCEPTOR_SERVED_SCOPE_HEADER == TOOL_SERVED_SCOPE_HEADER


def test_interceptor_and_tool_agree_on_the_credential_headers() -> None:
    """The three credential headers the interceptor sends are the three the tool maps."""
    interceptor_credential_headers = set(CREDENTIAL_HEADERS) - {
        INTERCEPTOR_SERVED_SCOPE_HEADER
    }
    assert interceptor_credential_headers == set(HEADER_TO_SESSION_KWARG)


def test_cdk_allowlist_matches_the_interceptor() -> None:
    """The CDK allowlists exactly what the interceptor sends — no more, no fewer.

    Fewer means the Gateway drops a header and the tool fails closed. More means a
    header is allowlisted that nothing sets, which lets a CLIENT supply it: the
    allowlist is what admits a client header, and interceptor precedence only wins
    for headers the interceptor actually sets.
    """
    assert set(_cdk_allowlist()) == set(CREDENTIAL_HEADERS)


def test_allowlist_fits_the_per_target_budget() -> None:
    """Four headers plus the service's own entry must stay inside the limit of 10."""
    # +1 for the service-maintained x-amzn-bedrock-agentcore-policy-session-id,
    # which was OBSERVED on the live targets and is not in the template.
    assert len(CREDENTIAL_HEADERS) + 1 <= MAX_ALLOWED_HEADERS_PER_TARGET


def test_no_header_name_is_reserved_or_malformed() -> None:
    """Names satisfy the Gateway's ^[a-zA-Z0-9_-]+$ rule and claim no AWS prefix."""
    import re  # noqa: PLC0415

    pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
    for name in CREDENTIAL_HEADERS:
        assert pattern.match(name), name
        assert not name.lower().startswith("x-amzn-"), name
        assert len(name) <= 100, name


@pytest.mark.skipif(
    not _TEMPLATE.exists(),
    reason="run `cdk synth` first; this asserts the SYNTHESIZED template",
)
def test_every_tool_target_allowlists_the_headers() -> None:
    """All three Lambda tool targets carry the allowlist in the synthesized template.

    Asserted against the template rather than the Python, because the template is
    what CloudFormation applies.
    """
    template = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    targets = {
        resource["Properties"]["Name"]: resource["Properties"]
        .get("MetadataConfiguration", {})
        .get("AllowedRequestHeaders")
        for resource in template["Resources"].values()
        if resource["Type"] == "AWS::BedrockAgentCore::GatewayTarget"
    }

    assert _TOOL_TARGET_NAMES <= set(targets), targets.keys()
    for name in _TOOL_TARGET_NAMES:
        assert targets[name] is not None, f"{name} has no allowlist"
        assert set(targets[name]) == set(CREDENTIAL_HEADERS), name


@pytest.mark.skipif(
    not _TEMPLATE.exists(),
    reason="run `cdk synth` first; this asserts the SYNTHESIZED template",
)
def test_probe_marker_header_is_gone() -> None:
    """The measurement's throwaway marker header is not in the template.

    It proved the channel exists and has no business surviving into the design.
    """
    assert "x-test-marker" not in _TEMPLATE.read_text(encoding="utf-8")
