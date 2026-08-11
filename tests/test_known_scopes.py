"""Tests for the interceptor's known-scope boundary (Interceptor Hardening Item 3).

The interceptor derives ``served_scope`` only for Cognito groups in
``jwt_claims._DEFAULT_KNOWN_SCOPES`` (``payments-core`` / ``billing-internal``).
``infra-secrets`` and ``hr-data`` are DynamoDB partitions but NOT Cognito groups, so
they can never appear in a token's ``cognito:groups`` and can never become
``served_scope``. These tests turn that implicit boundary into an enforced one:

* foreign-partition group names fail closed (resolve to ``None``); and
* ``_DEFAULT_KNOWN_SCOPES`` cannot drift from ``cdk/auth_resources.SCOPE_GROUPS``
  (the Cognito groups the stack provisions) without this test failing.

The drift test reads ``SCOPE_GROUPS`` from the CDK source via ``ast`` rather than
importing ``cdk/auth_resources.py`` (which imports ``aws_cdk`` at module top), keeping
this a pure unit test with no CDK/AWS dependency.
"""

from __future__ import annotations

import ast
import os

from interceptor.jwt_claims import _DEFAULT_KNOWN_SCOPES, _scope_from_cognito_groups

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AUTH_RESOURCES = os.path.join(_REPO_ROOT, "cdk", "auth_resources.py")


class TestForeignPartitionsFailClosed:
    """A token carrying a foreign-partition group name must resolve to None."""

    def test_infra_secrets_fails_closed(self) -> None:
        assert _scope_from_cognito_groups(["infra-secrets"]) is None

    def test_hr_data_fails_closed(self) -> None:
        assert _scope_from_cognito_groups(["hr-data"]) is None

    def test_both_foreign_partitions_fail_closed(self) -> None:
        assert _scope_from_cognito_groups(["infra-secrets", "hr-data"]) is None

    def test_known_scopes_still_resolve(self) -> None:
        # Positive control: the two real scope groups resolve exactly.
        assert _scope_from_cognito_groups(["payments-core"]) == "payments-core"
        assert _scope_from_cognito_groups(["billing-internal"]) == "billing-internal"

    def test_known_scope_alongside_foreign_ignores_foreign(self) -> None:
        # A foreign name is not in the known set, so it does not count as a match;
        # exactly one known match remains -> that scope. The foreign name never
        # becomes served_scope.
        assert _scope_from_cognito_groups(["payments-core", "infra-secrets"]) == "payments-core"


def _read_cdk_scope_groups() -> frozenset[str]:
    """Read ``SCOPE_GROUPS`` from cdk/auth_resources.py via ast (no import).

    Returns:
        The value of the module-level ``SCOPE_GROUPS`` assignment as a
        frozenset of strings.

    Raises:
        AssertionError: If the assignment cannot be found.
    """
    with open(_AUTH_RESOURCES, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=_AUTH_RESOURCES)
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "SCOPE_GROUPS":
                return frozenset(ast.literal_eval(value))
    raise AssertionError(
        f"SCOPE_GROUPS not found in {_AUTH_RESOURCES}"
    )


class TestKnownScopeDrift:
    """The interceptor's known-scope set must equal the provisioned Cognito groups."""

    def test_known_scopes_match_provisioned_scope_groups(self) -> None:
        provisioned = _read_cdk_scope_groups()
        assert _DEFAULT_KNOWN_SCOPES == provisioned, (
            "interceptor _DEFAULT_KNOWN_SCOPES has drifted from "
            "cdk/auth_resources.SCOPE_GROUPS: "
            f"{set(_DEFAULT_KNOWN_SCOPES)} != {set(provisioned)}"
        )

    def test_foreign_partitions_absent_from_both(self) -> None:
        # Belt-and-suspenders: neither constant may contain a foreign partition.
        provisioned = _read_cdk_scope_groups()
        for foreign in ("infra-secrets", "hr-data"):
            assert foreign not in _DEFAULT_KNOWN_SCOPES
            assert foreign not in provisioned
