"""Least-privilege Bedrock invoke grants for the agent Runtime's execution role.

The agent invokes exactly ONE model: the one resolved from ``bedrock_model_id``
(config.json / the ``BEDROCK_MODEL_ID`` environment variable). This module turns
that single identifier into the smallest IAM statement set that still works for a
cross-Region inference profile, so the Runtime role cannot invoke an arbitrary or
more expensive foundation model.

Why two statements are needed for a geo inference profile
---------------------------------------------------------
A geographic cross-Region inference profile id (``us.``/``eu.``/``apac.``/... plus
the foundation-model id) is a *routing* resource: the caller invokes the profile
in the source Region and Bedrock dispatches the request to the underlying
foundation model in one of the profile's destination Regions. An identity policy
must therefore allow BOTH:

1. the inference-profile ARN — account-scoped, in the source Region:
   ``arn:aws:bedrock:<region>:<account>:inference-profile/<profile-id>``
2. the underlying foundation-model ARN — NOT account-scoped (note the empty
   account segment ``::``), in every destination Region:
   ``arn:aws:bedrock:<dest-region>::foundation-model/<model-id>``

Granting only (1) fails with AccessDenied at dispatch time.

Rather than hardcode a destination-Region list (AWS may add Regions to a profile,
which would silently break invocation), statement (2) keeps a Region wildcard but
pins the *model* and adds a ``bedrock:InferenceProfileArn`` condition, so the
foundation model can only be invoked when the request is routed through THIS
profile — not called directly, and not in some other Region on its own. That is
the pattern AWS documents for least-privilege cross-Region inference.

References:
  - Prerequisites for inference profiles (both ARN forms required):
    https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html
  - Securing Bedrock cross-Region inference (the ``bedrock:InferenceProfileArn``
    condition scoping foundation-model access to a profile):
    https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-cross-region-inference-geographic-and-global/
  - Identity-based policy examples for Amazon Bedrock:
    https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html
"""

from __future__ import annotations

from aws_cdk import aws_iam as iam

# The two invoke actions the Strands BedrockModel provider may use. The Converse
# path can stream internally, so the streaming action is granted alongside the
# non-streaming one on the same (pinned) resources.
INVOKE_ACTIONS = [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream",
]

# Geography prefixes that mark a model id as a system-defined cross-Region
# inference profile rather than a bare foundation model. Matched as a literal
# leading segment; anything else is treated as a plain foundation-model id.
CRIS_GEO_PREFIXES = ("us.", "eu.", "apac.", "au.", "jp.", "ca.", "global.")


def split_model_id(model_id: str) -> tuple[str | None, str]:
    """Split a configured model id into its geo prefix and foundation-model id.

    Args:
        model_id: The configured Bedrock model identifier — either a bare
            foundation-model id (``anthropic.claude-sonnet-4-6``) or a
            cross-Region inference profile id (``us.anthropic.claude-sonnet-4-6``).

    Returns:
        A ``(geo_prefix, foundation_model_id)`` tuple. ``geo_prefix`` is the
        matched prefix including its trailing dot (e.g. ``"us."``) for an
        inference profile id, or ``None`` for a bare foundation-model id.
    """
    for prefix in CRIS_GEO_PREFIXES:
        if model_id.startswith(prefix):
            return prefix, model_id[len(prefix):]
    return None, model_id


def _validate(model_id: str) -> str:
    """Return the stripped model id, rejecting values that would widen the grant.

    A configured value containing a wildcard (or whitespace) would silently
    re-introduce the over-broad grant this module exists to remove, so it is
    rejected at synth time rather than deployed.

    Args:
        model_id: The raw configured model identifier.

    Returns:
        The stripped model id.

    Raises:
        ValueError: If the id is empty, or contains a wildcard or inner
            whitespace.
    """
    candidate = model_id.strip()
    if not candidate:
        raise ValueError("bedrock_model_id must not be empty.")
    if "*" in candidate or "?" in candidate:
        raise ValueError(
            "bedrock_model_id must name a single model — wildcards are not "
            f"permitted (got {candidate!r})."
        )
    if any(ch.isspace() for ch in candidate):
        raise ValueError(
            f"bedrock_model_id must not contain whitespace (got {candidate!r})."
        )
    return candidate


def model_resource_arns(
    model_id: str, region: str, account: str
) -> tuple[str | None, str]:
    """Compute the pinned ARNs for one configured model.

    Args:
        model_id: The configured Bedrock model identifier.
        region: The source Region the agent invokes from.
        account: The AWS account id (may be a CDK token).

    Returns:
        A ``(inference_profile_arn, foundation_model_arn)`` tuple. The profile
        ARN is ``None`` when the id is a bare foundation-model id; the
        foundation-model ARN is always present.

    Raises:
        ValueError: If ``model_id`` is empty or wildcarded (see ``_validate``).
    """
    candidate = _validate(model_id)
    geo, foundation_model_id = split_model_id(candidate)

    if geo is None:
        # Bare foundation model: single Region, no routing indirection.
        return None, f"arn:aws:bedrock:{region}::foundation-model/{foundation_model_id}"

    profile_arn = f"arn:aws:bedrock:{region}:{account}:inference-profile/{candidate}"
    # Region wildcard is intentional and safe here: the destination Region set of
    # a system-defined profile is AWS-managed and may change, and the statement
    # built below constrains this ARN to requests routed through profile_arn.
    return profile_arn, f"arn:aws:bedrock:*::foundation-model/{foundation_model_id}"


def invoke_statements(
    model_id: str, region: str, account: str
) -> list[iam.PolicyStatement]:
    """Build the least-privilege Bedrock invoke statements for one model.

    Args:
        model_id: The configured Bedrock model identifier.
        region: The source Region the agent invokes from.
        account: The AWS account id (may be a CDK token).

    Returns:
        One statement for a bare foundation-model id; two statements for a
        cross-Region inference profile id (the profile, plus the underlying
        foundation model conditioned on that profile).

    Raises:
        ValueError: If ``model_id`` is empty or wildcarded.
    """
    profile_arn, foundation_model_arn = model_resource_arns(model_id, region, account)

    if profile_arn is None:
        return [
            iam.PolicyStatement(
                sid="InvokePinnedFoundationModel",
                actions=INVOKE_ACTIONS,
                resources=[foundation_model_arn],
            )
        ]

    return [
        iam.PolicyStatement(
            sid="InvokePinnedInferenceProfile",
            actions=INVOKE_ACTIONS,
            resources=[profile_arn],
        ),
        iam.PolicyStatement(
            sid="InvokePinnedModelViaInferenceProfileOnly",
            actions=INVOKE_ACTIONS,
            resources=[foundation_model_arn],
            conditions={
                "StringEquals": {"bedrock:InferenceProfileArn": profile_arn}
            },
        ),
    ]
