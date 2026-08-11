"""Validation tests for ``shared.config_loader``.

``load_config`` is the first thing every starting component runs -- the CDK app,
the synth tests, the operator console -- and had no dedicated coverage of its
validation contract. (``tests/test_bedrock_model_grant.py`` calls it, but only
incidentally, to read a model id off a valid config.) Its job is to fail loudly
at load time so a bad edit cannot become an opaque runtime error later, and two
paths did the opposite:

* ``aws_region`` accepted any string. ``"us-east-99"`` and ``"not-a-region"``
  passed validation verbatim, then surfaced much later as an endpoint-resolution
  failure from whichever boto call ran first -- precisely the deferred, opaque
  failure the loader exists to prevent.
* a present-but-blank ``aws_region`` was silently rewritten to the default, so a
  half-finished edit deployed to us-east-1 without a word.

The region check is deliberately a SHAPE check, not an allow-list of live
regions: an allow-list goes stale on every AWS region launch and this loader
cannot refresh one offline. So the tests below assert both directions -- every
well-formed name is accepted (including regions that do not exist yet), and
names that cannot be a region are rejected.

``bedrock_model_id`` keeps presence-and-type validation only. The meaningful
check there is an allow-list of permitted models, which belongs with the Bedrock
IAM grant rather than in the loader, and is tracked separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shared.config_loader import (
    DEFAULT_AWS_REGION,
    AppConfig,
    ConfigError,
    load_config,
)

_MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def _write_config(tmp_path: Path, data: Any) -> Path:
    """Write ``data`` as JSON to a temp config file and return its path."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidConfig:
    """A well-formed config loads into a frozen AppConfig."""

    def test_both_fields_are_returned(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path, {"aws_region": "eu-west-1", "bedrock_model_id": _MODEL_ID}
        )

        config = load_config(path)

        assert isinstance(config, AppConfig)
        assert config.aws_region == "eu-west-1"
        assert config.bedrock_model_id == _MODEL_ID

    def test_the_committed_example_config_is_valid(self) -> None:
        # config.example.json is what a clean checkout (and CI) copies into place.
        # If a validation rule ever rejected it, every fresh clone would fail at
        # the first CDK command with a ConfigError -- so it is worth an assertion
        # rather than an assumption.
        #
        # Asserts VALIDITY only: that load_config accepts it and both fields come
        # back populated. Pinning the region to the default here would fail if
        # someone changed the example to another perfectly valid region, which is
        # not what this test is about (see test_absent_key_uses_the_named_default).
        repo_root = Path(__file__).resolve().parent.parent

        config = load_config(repo_root / "config.example.json")

        assert config.aws_region
        assert config.bedrock_model_id

    def test_unknown_fields_are_ignored(self, tmp_path: Path) -> None:
        # Forward compatibility: an extra key is not an error, so a config written
        # for a newer revision still loads.
        path = _write_config(
            tmp_path,
            {
                "aws_region": "us-east-1",
                "bedrock_model_id": _MODEL_ID,
                "some_future_field": "ignored",
            },
        )

        assert load_config(path).aws_region == "us-east-1"


# ---------------------------------------------------------------------------
# aws_region -- the default
# ---------------------------------------------------------------------------


class TestRegionDefault:
    """The default applies to an ABSENT key, and only to an absent key."""

    def test_absent_key_uses_the_named_default(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, {"bedrock_model_id": _MODEL_ID})

        # Asserted against the constant rather than a repeated literal: the point
        # is that the fallback is the documented one, whatever its value.
        assert load_config(path).aws_region == DEFAULT_AWS_REGION

    def test_explicit_null_uses_the_default(self, tmp_path: Path) -> None:
        # JSON null reads as "not provided", which is the same intent as omitting
        # the key -- unlike a blank string, which is a malformed value.
        path = _write_config(
            tmp_path, {"aws_region": None, "bedrock_model_id": _MODEL_ID}
        )

        assert load_config(path).aws_region == DEFAULT_AWS_REGION

    @pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n"])
    def test_present_but_blank_is_rejected_not_defaulted(
        self, tmp_path: Path, blank: str
    ) -> None:
        # The behaviour change: a blank value used to become the default silently.
        # A key someone wrote and left empty is a mistake, and the loader now says
        # so instead of quietly deploying to another region.
        path = _write_config(
            tmp_path, {"aws_region": blank, "bedrock_model_id": _MODEL_ID}
        )

        with pytest.raises(ConfigError) as excinfo:
            load_config(path)

        message = str(excinfo.value)
        assert "aws_region" in message
        # The message must tell the reader how to resolve it, including what the
        # default would be if they meant to omit the field.
        assert DEFAULT_AWS_REGION in message


# ---------------------------------------------------------------------------
# aws_region -- the shape check
# ---------------------------------------------------------------------------


class TestRegionShape:
    """A present region must look like a region."""

    @pytest.mark.parametrize(
        "region",
        [
            "us-east-1",
            "us-east-2",
            "us-west-2",
            "eu-west-3",
            "eu-central-1",
            "eu-south-2",
            "ap-southeast-1",
            "ap-southeast-7",
            "ap-northeast-3",
            "ca-west-1",
            "sa-east-1",
            "il-central-1",
            "mx-central-1",
            "me-south-1",
            "af-south-1",
            # Partitions beyond aws: GovCloud, ISO, China. Their names carry an
            # extra segment, which a pattern written for the commercial shape only
            # would wrongly reject.
            "us-gov-west-1",
            "us-gov-east-1",
            "us-iso-east-1",
            "us-isob-east-1",
            "cn-north-1",
            "cn-northwest-1",
        ],
    )
    def test_real_regions_are_accepted(self, tmp_path: Path, region: str) -> None:
        path = _write_config(
            tmp_path, {"aws_region": region, "bedrock_model_id": _MODEL_ID}
        )

        assert load_config(path).aws_region == region

    def test_a_region_that_does_not_exist_yet_is_accepted(
        self, tmp_path: Path
    ) -> None:
        # The check is structural on purpose. A well-formed name for a region AWS
        # has not launched yet must load, because the alternative -- a hardcoded
        # allow-list -- would block every future launch until someone edited this
        # repository.
        path = _write_config(
            tmp_path, {"aws_region": "eu-north-2", "bedrock_model_id": _MODEL_ID}
        )

        assert load_config(path).aws_region == "eu-north-2"

    @pytest.mark.parametrize(
        "region",
        [
            "us-east-99",  # the shape is plausible, the ordinal is not
            "not-a-region",
            "useast1",
            "us_east_1",
            "US-EAST-1",  # regions are lowercase
            "us-east-1a",  # an availability zone, not a region
            "us-east-",
            "us-east",
            "1-east-us",
            " us-east-1",  # a stray space in a hand-edited file
            "us-east-1 ",
            "us-east-1,eu-west-1",
            "arn:aws:iam::123456789012:role/Example",
        ],
    )
    def test_malformed_regions_are_rejected(
        self, tmp_path: Path, region: str
    ) -> None:
        path = _write_config(
            tmp_path, {"aws_region": region, "bedrock_model_id": _MODEL_ID}
        )

        with pytest.raises(ConfigError) as excinfo:
            load_config(path)

        # The message must quote the offending value: "not a well-formed region"
        # with no value named sends the reader hunting through the file.
        assert region in str(excinfo.value)

    @pytest.mark.parametrize("region", [1, 1.5, True, ["us-east-1"], {"a": "b"}])
    def test_non_string_regions_are_rejected(
        self, tmp_path: Path, region: Any
    ) -> None:
        path = _write_config(
            tmp_path, {"aws_region": region, "bedrock_model_id": _MODEL_ID}
        )

        with pytest.raises(ConfigError, match="must be a string"):
            load_config(path)


# ---------------------------------------------------------------------------
# bedrock_model_id
# ---------------------------------------------------------------------------


class TestModelId:
    """The model id is required, must be a non-empty string, and is not shape-checked."""

    def test_missing_is_rejected(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, {"aws_region": "us-east-1"})

        with pytest.raises(ConfigError, match="bedrock_model_id"):
            load_config(path)

    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_empty_is_rejected(self, tmp_path: Path, value: str) -> None:
        path = _write_config(
            tmp_path, {"aws_region": "us-east-1", "bedrock_model_id": value}
        )

        with pytest.raises(ConfigError, match="bedrock_model_id"):
            load_config(path)

    @pytest.mark.parametrize("value", [None, 42, ["a"], {"a": "b"}])
    def test_non_string_is_rejected(self, tmp_path: Path, value: Any) -> None:
        path = _write_config(
            tmp_path, {"aws_region": "us-east-1", "bedrock_model_id": value}
        )

        with pytest.raises(ConfigError, match="bedrock_model_id"):
            load_config(path)

    def test_an_arbitrary_model_id_string_is_still_accepted(
        self, tmp_path: Path
    ) -> None:
        # Documents the deliberate boundary: the loader does NOT police which
        # model may be used. Restricting that is an allow-list enforced alongside
        # the Bedrock IAM grant, tracked as its own issue; asserting it here would
        # duplicate that decision in the wrong layer.
        path = _write_config(
            tmp_path,
            {"aws_region": "us-east-1", "bedrock_model_id": "anything-goes-here"},
        )

        assert load_config(path).bedrock_model_id == "anything-goes-here"


# ---------------------------------------------------------------------------
# File-level failures
# ---------------------------------------------------------------------------


class TestFileFailures:
    """A missing, unparseable or wrongly-typed file halts with a clear error."""

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "does-not-exist.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ConfigError, match="not valid JSON"):
            load_config(path)

    @pytest.mark.parametrize("payload", ["[]", '"a string"', "42", "null"])
    def test_a_json_document_that_is_not_an_object(
        self, tmp_path: Path, payload: str
    ) -> None:
        path = tmp_path / "config.json"
        path.write_text(payload, encoding="utf-8")

        with pytest.raises(ConfigError, match="must contain a JSON object"):
            load_config(path)

    def test_a_directory_is_not_mistaken_for_a_config_file(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "config.json"
        directory.mkdir()

        with pytest.raises(ConfigError, match="not found"):
            load_config(directory)
