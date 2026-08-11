"""Property test — the session policy is pinned in space and time.

The property: for ANY served scope, for BOTH the read action set
(``dynamodb:GetItem`` / ``dynamodb:Query``) and the write action set
(``dynamodb:UpdateItem``), for ANY table ARN, and for ANY injected
``expires_at`` string formatted ``%Y-%m-%dT%H:%M:%SZ``, the **pure**
``interceptor.scoped_credentials.build_session_policy`` output — parsed from its
returned JSON string — has a single ``Allow`` statement whose ``Condition``
carries all THREE guards:

* ``ForAllValues:StringEquals`` on ``dynamodb:LeadingKeys`` equal to
  ``[served_scope]`` (the SPATIAL guard);
* a ``Null`` presence guard on ``dynamodb:LeadingKeys`` equal to ``"false"``
  (the ``ForAllValues`` footgun guard); and
* a ``DateLessThan`` on ``aws:CurrentTime`` (the TEMPORAL guard)

where the ``aws:CurrentTime`` value parses under ``%Y-%m-%dT%H:%M:%SZ`` and
equals the injected ``expires_at`` exactly (the builder is deterministic and has
no wall-clock dependence); the statement's ``Effect`` is ``Allow`` and its
``Resource`` is the supplied table ARN; and the statement's ``Action`` list NEVER
contains ``dynamodb:Scan`` for any generated action set.

The builder is pure — it computes no time itself and takes ``expires_at`` as an
injected argument — so this property is checked directly against the builder
with generated inputs; no time mocking is needed. The ±2-second wall-clock
tolerance is intentionally NOT part of this pure-builder property; it is
exercised by the separate vend-path timing/config unit test in
``tests/test_vend_timing.py``.

AWS documentation references:

* ``dynamodb:LeadingKeys`` is the fine-grained partition-key IAM condition key,
  qualified with the ``ForAllValues`` set operator:
  https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/specifying-conditions.html
* ``Null`` is the presence-check condition operator (``"false"`` requires the key
  to be present):
  https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_condition_operators.html
* ``DateLessThan`` is a Date condition operator used with the ``aws:CurrentTime``
  global condition key; the value is an ISO 8601 date/time string
  (``%Y-%m-%dT%H:%M:%SZ``):
  https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_aws-dates.html
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from interceptor.scoped_credentials import (
    READ_ACTIONS,
    WRITE_ACTIONS,
    build_session_policy,
)

# The whole-second UTC ISO 8601 format the `DateLessThan` / `aws:CurrentTime`
# condition value uses (ending in a literal `Z`).
_EXPIRES_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# The keyless action the policy must NEVER grant: `dynamodb:Scan` reads the whole
# table and carries no partition key, so the `LeadingKeys` guard cannot confine
# it. Its absence for every action set is the "never widens to Scan" half of the
# property.
_FORBIDDEN_ACTION = "dynamodb:Scan"


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


def _served_scopes() -> st.SearchStrategy[str]:
    """Arbitrary served-scope strings (non-empty; the confined partition value)."""
    return st.text(min_size=1, max_size=64)


def _table_arns() -> st.SearchStrategy[str]:
    """Arbitrary DynamoDB table ARNs (well-formed ``arn:aws:dynamodb:...`` shape).

    The builder treats the ARN as an opaque ``Resource`` value, so the exact
    account / region / table-name segments are generated freely; the property
    only asserts the emitted ``Resource`` equals whatever ARN was supplied.
    """
    region = st.sampled_from(["us-east-1", "eu-west-1", "ap-southeast-2"])
    account = st.text(alphabet="0123456789", min_size=12, max_size=12)
    table = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.",
        min_size=3,
        max_size=40,
    )
    return st.builds(
        lambda r, a, t: f"arn:aws:dynamodb:{r}:{a}:table/{t}",
        region,
        account,
        table,
    )


def _action_sets() -> st.SearchStrategy[list[str]]:
    """Draw either the read action set or the write action set.

    A fresh copy is returned so a test can never mutate the module-level
    ``READ_ACTIONS`` / ``WRITE_ACTIONS`` lists through an alias.
    """
    return st.sampled_from([READ_ACTIONS, WRITE_ACTIONS]).map(list)


def _expires_at() -> st.SearchStrategy[str]:
    """Arbitrary well-formed ``expires_at`` strings (``%Y-%m-%dT%H:%M:%SZ``).

    Generates arbitrary UTC datetimes and formats them to the whole-second UTC
    ISO 8601 string (ending in ``Z``) the ``DateLessThan`` condition expects. The
    year is bounded at four digits (>= 1000) so ``strftime`` always zero-pads the
    year to four characters and the round-trip through ``strptime`` is exact on
    every platform.
    """
    return st.datetimes(
        min_value=datetime(1000, 1, 1, 0, 0, 0),
        max_value=datetime(9999, 12, 31, 23, 59, 59),
    ).map(lambda dt: dt.strftime(_EXPIRES_AT_FORMAT))


# ---------------------------------------------------------------------------
# The session-policy property
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    served_scope=_served_scopes(),
    table_arn=_table_arns(),
    actions=_action_sets(),
    expires_at=_expires_at(),
)
def test_session_policy_dual_condition(
    served_scope: str,
    table_arn: str,
    actions: list[str],
    expires_at: str,
) -> None:
    """The session policy is pinned in space and time and never widens to ``Scan``.

    For all served scopes, both action sets, all table ARNs, and any injected
    ``expires_at``, the pure ``build_session_policy`` output's single ``Allow``
    statement carries the three Condition guards (``ForAllValues:StringEquals`` on
    ``dynamodb:LeadingKeys`` equal to ``[served_scope]``, the ``Null`` presence
    guard on ``dynamodb:LeadingKeys`` equal to ``"false"``, and ``DateLessThan`` on
    ``aws:CurrentTime``); the ``aws:CurrentTime`` value parses under
    ``%Y-%m-%dT%H:%M:%SZ`` and equals the injected ``expires_at`` exactly; the
    ``Effect`` is ``Allow`` and the ``Resource`` is the supplied ARN; and the
    ``Action`` list never contains ``dynamodb:Scan``.
    """
    policy_json = build_session_policy(served_scope, table_arn, actions, expires_at)

    # The STS `Policy` parameter is a JSON *string*; parse it before asserting.
    policy: dict[str, Any] = json.loads(policy_json)

    # --- Exactly one Allow statement. ------------------------------------------
    statements = policy["Statement"]
    assert isinstance(statements, list)
    assert len(statements) == 1
    statement = statements[0]

    # --- Effect / Resource. -----------------------------------------------------
    assert statement["Effect"] == "Allow"
    assert statement["Resource"] == table_arn

    # --- Action set never widens to a keyless Scan. -----------------------------
    action_list = statement["Action"]
    assert _FORBIDDEN_ACTION not in action_list
    # The generated set is exactly one of the two sanctioned action sets.
    assert action_list == actions
    assert action_list in (READ_ACTIONS, WRITE_ACTIONS)

    condition = statement["Condition"]

    # --- SPATIAL guard: ForAllValues:StringEquals on LeadingKeys = [scope]. -----
    assert "ForAllValues:StringEquals" in condition
    assert condition["ForAllValues:StringEquals"]["dynamodb:LeadingKeys"] == [
        served_scope
    ]

    # --- Footgun guard: Null presence check requiring LeadingKeys to exist. -----
    assert "Null" in condition
    assert condition["Null"]["dynamodb:LeadingKeys"] == "false"

    # --- TEMPORAL guard: DateLessThan on aws:CurrentTime = injected expiry. -----
    assert "DateLessThan" in condition
    current_time = condition["DateLessThan"]["aws:CurrentTime"]
    # Parses under the whole-second UTC ISO 8601 format...
    parsed = datetime.strptime(current_time, _EXPIRES_AT_FORMAT)
    assert isinstance(parsed, datetime)
    # ...and is the injected expiry verbatim (the builder is a pure passthrough).
    assert current_time == expires_at
