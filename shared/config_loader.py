"""
Configuration loader.

This module reads and validates config.json from the repository root. Every
starting component (the CDK stack) imports load_config() before performing any
action. The loader halts with a specific error when:

- The config file is missing or unparseable
- A required field is missing or empty
- An optional field is present but malformed (a blank or ill-shaped aws_region)

Resource names are kept in code, not in config.
No credentials are stored in config.
aws_region is optional and defaults to DEFAULT_AWS_REGION when the key is
ABSENT; a present value must be a well-formed region name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when config.json is missing, unparseable, or invalid."""

    pass


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration.

    Attributes:
        aws_region: AWS region for deployment (DEFAULT_AWS_REGION when the
            config omits the key).
        bedrock_model_id: Bedrock model identifier for the agent.
    """

    aws_region: str
    bedrock_model_id: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_FILENAME = "config.json"

# The region used when config.json omits ``aws_region``. Named rather than
# inlined so the fallback is visible at module level and testable by reference.
DEFAULT_AWS_REGION = "us-east-1"

# Structural shape of an AWS region name: a two-letter geography, one or more
# lowercase words, then a single-digit ordinal --  us-east-1, eu-west-3,
# ap-southeast-7, us-gov-east-1, us-iso-east-1, il-central-1, mx-central-1.
# Deliberately a shape check and not an allow-list (see _validate_region). The
# ordinal is one digit because no AWS region has ever carried two; if that
# changes, this pattern must widen -- and it will fail loudly at config load with
# a message naming the value, which is a far cheaper failure than accepting a
# nonexistent region and hitting an endpoint-resolution error at runtime.
_REGION_PATTERN = re.compile(r"[a-z]{2}(?:-[a-z]+)+-\d")


def _resolve_config_path() -> Path:
    """Return the absolute path to config.json at the repository root."""
    return _REPO_ROOT / _CONFIG_FILENAME


def _read_json(path: Path) -> dict[str, Any]:
    """Read and parse config.json, raising ConfigError on failure.

    Args:
        path: Absolute path to config.json.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        ConfigError: If the file is missing or not valid JSON.
    """
    if not path.is_file():
        raise ConfigError(
            f"Configuration file not found: {path}. "
            "config.json must exist at the repository root."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(
            f"Configuration file is not valid JSON: {path}. Error: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Configuration file must contain a JSON object, got {type(data).__name__}."
        )

    return data


def _validate_required_string(data: dict[str, Any], field: str) -> str:
    """Validate that a field is present and non-empty.

    Args:
        data: Parsed config dictionary.
        field: Field name to check.

    Returns:
        The validated string value.

    Raises:
        ConfigError: If the field is missing, None, or empty/whitespace-only.
    """
    value = data.get(field)
    if value is None:
        raise ConfigError(
            f"Required configuration field '{field}' is missing. "
            f"Add '{field}' to config.json."
        )
    if not isinstance(value, str):
        raise ConfigError(
            f"Required configuration field '{field}' must be a string, "
            f"got {type(value).__name__}."
        )
    if not value.strip():
        raise ConfigError(
            f"Required configuration field '{field}' is empty. "
            f"Provide a non-empty value for '{field}' in config.json."
        )
    return value


def _validate_region(data: dict[str, Any]) -> str:
    """Validate ``aws_region``, applying the documented default when absent.

    The field is optional: an absent key means "use the default", and that
    default is the named ``DEFAULT_AWS_REGION`` constant rather than a literal
    buried in the branch. Every OTHER shape is rejected loudly:

    * a non-string value (as before);
    * a present-but-blank string — a blank value is a malformed edit, not an
      omission, and silently rewriting it to the default hid the mistake;
    * a string that is not shaped like a region. Without this check
      ``"us-east-99"`` and ``"not-a-region"`` were accepted verbatim at load
      time and failed much later, as an opaque endpoint-resolution error from
      whichever boto call happened to run first.

    The check is structural, not an allow-list of live regions: a hardcoded list
    goes stale every time AWS launches a region, and this loader has no way to
    refresh it offline. It therefore accepts any well-formed region name --
    including one that does not exist yet -- and rejects only names that cannot
    be a region at all.

    Args:
        data: Parsed config dictionary.

    Returns:
        The validated region string, or ``DEFAULT_AWS_REGION`` when the key is
        absent.

    Raises:
        ConfigError: If the value is present but not a well-formed region string.
    """
    if "aws_region" not in data or data["aws_region"] is None:
        return DEFAULT_AWS_REGION

    aws_region = data["aws_region"]
    if not isinstance(aws_region, str):
        raise ConfigError(
            f"Configuration field 'aws_region' must be a string, "
            f"got {type(aws_region).__name__}."
        )

    if not aws_region.strip():
        raise ConfigError(
            "Configuration field 'aws_region' is present but empty. Either remove "
            f"it to accept the default ({DEFAULT_AWS_REGION}) or provide a region "
            "such as 'eu-west-1'."
        )

    if not _REGION_PATTERN.fullmatch(aws_region):
        raise ConfigError(
            f"Configuration field 'aws_region' is not a well-formed AWS region: "
            f"{aws_region!r}. Expected a name such as 'us-east-1', 'eu-west-3' or "
            "'us-gov-east-1'."
        )

    return aws_region


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load and validate config.json, returning an AppConfig instance.

    This function must be called before any provisioning, setup, or console
    action. It halts the calling component with a ConfigError if the config
    is invalid.

    Args:
        config_path: Optional override for the config file path. When None,
            defaults to config.json at the repository root.

    Returns:
        A validated AppConfig dataclass.

    Raises:
        ConfigError: If the file is missing/unparseable, a required field is
            missing or empty, or ``aws_region`` is present but blank or not a
            well-formed region name.
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        path = _resolve_config_path()

    data = _read_json(path)

    aws_region = _validate_region(data)

    # Required fields that must be present and non-empty. ``bedrock_model_id``
    # gets presence-and-type validation only: the meaningful check is an
    # allow-list of permitted models, which belongs with the Bedrock IAM grant
    # rather than here (tracked separately as the model allow-list issue).
    bedrock_model_id = _validate_required_string(data, "bedrock_model_id")

    return AppConfig(
        aws_region=aws_region,
        bedrock_model_id=bedrock_model_id,
    )
