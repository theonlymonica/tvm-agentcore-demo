"""Shared test helpers for the credential-HEADER channel.

The Gateway hands an allowlisted request header to a Lambda target inside the
Lambda CLIENT CONTEXT, at
``context.client_context.custom["bedrockAgentCorePropagatedHeaders"]`` — a shape
established by live measurement against a deployed gateway,
not by the service reference.

Every test that needs that shape builds it HERE rather than inline, for the same
reason the production readers live in one module: if the shape is written out in
six test files and the real one changes, five of them keep passing against a shape
the Gateway no longer sends.

Functions:
    lambda_context: Build a context carrying an arbitrary propagated-header map.
    credential_context: Build a context carrying a complete, valid credential set.
    headerless_context: Build a context whose client context has no headers at all.
    no_client_context: Build a context with no client context, as a direct
        (non-Gateway) invocation produces.
"""

from __future__ import annotations

from typing import Any, Optional

from common.credentials_context import PROPAGATED_HEADERS_KEY

#: The four header names, duplicated from the production modules on purpose: a
#: test that imported them could not catch a rename that broke the wire contract,
#: because it would rename with it. tests/test_header_contract_parity.py is where
#: the literals are pinned to the production constants.
SERVED_SCOPE_HEADER = "x-tvm-served-scope"
ACCESS_KEY_ID_HEADER = "x-tvm-access-key-id"
SECRET_ACCESS_KEY_HEADER = "x-tvm-secret-access-key"
SESSION_TOKEN_HEADER = "x-tvm-session-token"


class _ClientContext:
    """Minimal stand-in for the Lambda ``ClientContext`` object."""

    def __init__(self, custom: Any) -> None:
        self.custom = custom
        self.env = None
        self.client = None


class _LambdaContext:
    """Minimal stand-in for the Lambda context object passed to a handler."""

    def __init__(self, client_context: Optional[_ClientContext]) -> None:
        self.client_context = client_context


def lambda_context(headers: Any) -> Any:
    """Build a Lambda context carrying ``headers`` as the propagated-header map.

    Args:
        headers: Whatever should sit under the propagated-headers key. Deliberately
            untyped so a test can pass a malformed value (None, a string, a list)
            to exercise the fail-closed branches.

    Returns:
        A context object shaped like the one a Gateway-invoked Lambda receives.
    """
    return _LambdaContext(_ClientContext({PROPAGATED_HEADERS_KEY: headers}))


def credential_context(
    scope: str = "payments-core",
    *,
    suffix: Any = "",
) -> Any:
    """Build a Lambda context carrying a complete, valid credential header set.

    Args:
        scope: The served scope to put in the scope header.
        suffix: Appended to each credential value, so a test driving several
            requests can tell one request's credentials from another's.

    Returns:
        A context object carrying all four headers as non-empty strings.
    """
    tag = str(suffix)
    return lambda_context(
        {
            SERVED_SCOPE_HEADER: scope,
            ACCESS_KEY_ID_HEADER: f"ASIAFAKEKEY{tag}",
            SECRET_ACCESS_KEY_HEADER: f"fake-secret{tag}",
            SESSION_TOKEN_HEADER: f"fake-session-token{tag}",
        }
    )


def headerless_context() -> Any:
    """Build a context whose client context carries no propagated headers.

    Returns:
        A context object with an empty ``custom`` map — the shape a target with no
        allowlist receives, which must fail closed.
    """
    return _LambdaContext(_ClientContext({}))


def no_client_context() -> Any:
    """Build a context with NO client context at all.

    This is what a DIRECT Lambda invocation produces — someone calling the tool
    outside the Gateway, which is exactly the caller that must get nothing.

    Returns:
        A context object whose ``client_context`` is None.
    """
    return _LambdaContext(None)
