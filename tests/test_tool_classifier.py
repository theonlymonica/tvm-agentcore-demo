"""Tests for ``interceptor.tool_classifier.classify_tool``.

The fix closes a scope-confinement bypass: the pre-fix ``classify_tool`` fell back to
``action_name.endswith(tool)`` after the ``___`` split, so a tool named to END in a
known tool name (e.g. ``evil___fake_read_document``) classified as that tool, entered
the scoped set, and would be handed vended credentials.

This module asserts the POST-fix exact-match behavior. The PRE-fix bypass evidence was
captured by running these same assertions against the unfixed classifier: the three
cases below failed with the classifier returning ``read_document``:
  - ``evil___fake_read_document`` -> ``read_document`` (should be UNCLASSIFIABLE)
  - ``x___notread_document``      -> ``read_document`` (should be UNCLASSIFIABLE)
  - ``read_document`` (no ``___``) -> ``read_document`` (should be UNCLASSIFIABLE)
After the fix, all classify as ``UNCLASSIFIABLE`` (fail closed).

Gateway tool names are always ``${target_name}___${tool_name}`` (AWS gateway tool-naming
rule), so an absent ``___`` delimiter is illegitimate and must fail closed.
"""

from __future__ import annotations

import pytest

from interceptor.tool_classifier import (
    TOOL_READ_DOCUMENT,
    TOOL_UNCLASSIFIABLE,
    classify_tool,
)


class TestClassifyToolExactMatch:
    """Post-fix: classification is an EXACT match on the segment after the final ``___``."""

    def test_real_gateway_name_classifies(self) -> None:
        # Deployed target name is "ReadDocument"; the MCP tool name is the gateway form.
        assert classify_tool("ReadDocument___read_document") == TOOL_READ_DOCUMENT

    def test_malicious_suffix_prefixed_name_fails_closed(self) -> None:
        # BYPASS CASE: ends with "read_document" but the trailing ___ segment is
        # "fake_read_document" (not an exact known tool) -> must be UNCLASSIFIABLE.
        assert classify_tool("evil___fake_read_document") == TOOL_UNCLASSIFIABLE

    def test_malicious_no_delimiter_before_suffix_fails_closed(self) -> None:
        # BYPASS CASE: "x___notread_document" -> trailing segment "notread_document".
        assert classify_tool("x___notread_document") == TOOL_UNCLASSIFIABLE

    def test_no_delimiter_fails_closed(self) -> None:
        # The gateway always emits "${target}___${tool}"; a bare name is illegitimate.
        assert classify_tool("read_document") == TOOL_UNCLASSIFIABLE

    def test_exact_match_negative_trailing_char(self) -> None:
        # "read_documentx" is not exactly a known tool name.
        assert classify_tool("read_document___read_documentx") == TOOL_UNCLASSIFIABLE

    def test_none_and_empty_fail_closed(self) -> None:
        assert classify_tool(None) == TOOL_UNCLASSIFIABLE
        assert classify_tool("") == TOOL_UNCLASSIFIABLE
