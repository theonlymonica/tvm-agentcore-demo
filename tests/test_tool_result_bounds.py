"""Bounded-result / bounded-write tests for search_documents and reply.

Both tools previously had an unbounded shape on their hot path:

* ``search_documents`` issued a partition ``Query`` with no ``Limit`` and walked
  every ``LastEvaluatedKey`` page, accumulating every title match into an
  uncapped list. Latency, consumed RCUs and the size of the result set injected
  into the model context all scaled with the partition size rather than with the
  number of results asked for -- and since the model picks the query string, an
  injected instruction could issue deliberately broad queries repeatedly to
  amplify it.
* ``reply`` checked ``body`` only for non-emptiness and appended it with
  ``list_append`` with no bound on the entry size or the resulting list length,
  so repeated or large replies grew the item toward DynamoDB's 400 KB item
  limit -- past which every further write to that document hard-fails.

These tests lock in the caps and, just as importantly, the boundaries: an exact
fit must NOT be reported as truncated, and a reply that exactly reaches a limit
must still be accepted. They also assert the negative side of each bound -- that
a rejected write consumed nothing -- since a bound that returns an error while
still writing would pass a naive assertion.

The moto-backed cases exercise the real DynamoDB semantics (pagination,
``size()`` in a ``ConditionExpression``). The page-cap case uses a recording
stub table instead, because proving "at most N Query requests are issued" is a
statement about the request sequence, and driving the real 100-item page size
past the page cap would need >500 seeded items to say the same thing far more
slowly.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

import reply.handler as reply_module
import search_documents.handler as search_module
from reply.handler import handler as reply_handler
from search_documents.handler import handler as search_documents_handler
from tests.conftest import SERVED_SCOPE, make_document

# Matches the credential shapes the other tool tests use, so a fixture never
# contradicts the wire contract. moto accepts any credential values.
_ACCESS_KEY_ID = "ASIAIOSFODNN7EXAMPLE"
_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEXAMPLETOKEN"


def _context(scope: str = SERVED_SCOPE) -> dict[str, Any]:
    """Return a well-formed injected ``context`` object for ``scope``."""
    return {
        "served_scope": scope,
        "tenant_credentials": {
            "access_key_id": _ACCESS_KEY_ID,
            "secret_access_key": _SECRET_ACCESS_KEY,
            "session_token": _SESSION_TOKEN,
        },
    }


def _matching_documents(count: int, *, scope: str = SERVED_SCOPE) -> list[dict[str, Any]]:
    """Return ``count`` served-scope documents whose titles all match "refund"."""
    return [
        make_document(scope, f"DOC-{index:04d}", title=f"Refund case {index:04d}")
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# search_documents -- result cap
# ---------------------------------------------------------------------------


class TestSearchResultCap:
    """The returned result set is capped and reports its own completeness."""

    def test_more_matches_than_the_cap_are_truncated(
        self, scoped_env, put_documents
    ) -> None:
        # Twice the cap in matching documents: the response carries exactly the
        # cap, flags itself partial, and explains why.
        put_documents(_matching_documents(search_module._MAX_RESULTS * 2))

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert len(result["results"]) == search_module._MAX_RESULTS
        assert result["truncated"] is True
        assert "note" in result
        # Every returned entry still belongs to the served partition and still
        # carries only the three published fields -- the cap must not widen the
        # per-entry projection.
        for entry in result["results"]:
            assert set(entry) == {"doc_id", "scope", "title"}
            assert entry["scope"] == SERVED_SCOPE

    def test_exactly_the_cap_is_not_reported_truncated(
        self, scoped_env, put_documents
    ) -> None:
        # Boundary: the cap is reached, but the partition holds no further match, so
        # the result set is complete. The trailing non-matching documents are the
        # point of the fixture -- they sort after the matches, so a cap tested per
        # ITEM rather than per MATCH would trip on them and wrongly report partial,
        # pushing the model into re-querying for results that do not exist. Without
        # them this test passes against that bug.
        documents = _matching_documents(search_module._MAX_RESULTS)
        documents += [
            make_document(SERVED_SCOPE, f"ZZZ-{index:04d}", title="Release status")
            for index in range(5)
        ]
        put_documents(documents)

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert len(result["results"]) == search_module._MAX_RESULTS
        assert result["truncated"] is False
        assert "note" not in result

    def test_one_match_past_the_cap_is_reported_truncated(
        self, scoped_env, put_documents
    ) -> None:
        # The other side of that boundary: a single genuine match beyond the cap is
        # enough to make the set partial, even though the partition is exhausted in
        # one page.
        put_documents(_matching_documents(search_module._MAX_RESULTS + 1))

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert len(result["results"]) == search_module._MAX_RESULTS
        assert result["truncated"] is True

    def test_fewer_matches_than_the_cap_are_complete(
        self, scoped_env, put_documents
    ) -> None:
        put_documents(_matching_documents(3))

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert len(result["results"]) == 3
        assert result["truncated"] is False

    def test_no_matches_is_complete_and_not_truncated(
        self, scoped_env, put_documents
    ) -> None:
        put_documents([make_document(SERVED_SCOPE, "PAY-001", title="Release status")])

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert result == {"results": [], "truncated": False}


# ---------------------------------------------------------------------------
# search_documents -- read cap (pages)
# ---------------------------------------------------------------------------


class _RecordingTable:
    """A stub DynamoDB ``Table`` that records every ``query`` it is handed.

    By default it returns ``items_per_page`` NON-matching items per page and always
    advertises another page via ``LastEvaluatedKey``, i.e. an effectively infinite
    partition. Only the page cap can end that walk, so the number of recorded calls
    is a direct measurement of the read bound.

    ``pages`` overrides that with an explicit page script, for the boundaries a real
    table cannot cheaply produce: a page carrying exactly the result cap plus a
    cursor, or a page whose ``Items`` key is empty or absent entirely. Each scripted
    page gets a ``LastEvaluatedKey`` unless it declares one, so every scripted case
    still advertises more data; the final scripted page repeats if the walk outlives
    the script.
    """

    def __init__(
        self,
        items_per_page: int = 3,
        pages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._items_per_page = items_per_page
        self._pages = pages

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        page_number = len(self.calls)
        cursor = {"scope": SERVED_SCOPE, "doc_id": f"cursor-{page_number}"}

        if self._pages is not None:
            page = dict(self._pages[min(page_number - 1, len(self._pages) - 1)])
            page.setdefault("LastEvaluatedKey", cursor)
            return page

        return {
            "Items": [
                # Titles deliberately do NOT contain the query, so no match ever
                # ends the walk early and the page cap is what stops it.
                {
                    "scope": SERVED_SCOPE,
                    "doc_id": f"DOC-{page_number}-{index}",
                    "title": "Release status",
                }
                for index in range(self._items_per_page)
            ],
            "LastEvaluatedKey": cursor,
        }


class TestSearchReadCap:
    """The partition walk is bounded in pages read, not just results returned."""

    @pytest.fixture
    def recording_table(self, monkeypatch: pytest.MonkeyPatch) -> _RecordingTable:
        table = _RecordingTable()
        monkeypatch.setattr(
            search_module, "documents_table_from_event", lambda _event: table
        )
        return table

    def test_page_cap_stops_an_endless_partition(
        self, scoped_env, recording_table: _RecordingTable
    ) -> None:
        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        # The read bound held: the walk stopped at the page cap rather than
        # following LastEvaluatedKey forever.
        assert len(recording_table.calls) == search_module._MAX_PAGE_READS
        # Nothing matched, yet the response is still flagged partial -- an empty
        # result set from a walk that stopped early must not read as "no such
        # document exists".
        assert result["results"] == []
        assert result["truncated"] is True
        assert "note" in result

    def test_every_query_carries_a_per_request_limit(
        self, scoped_env, recording_table: _RecordingTable
    ) -> None:
        search_documents_handler({"query": "refund", "context": _context()}, None)

        for call in recording_table.calls:
            assert call["Limit"] == search_module._PAGE_ITEM_LIMIT

    def test_pagination_still_chains_the_cursor(
        self, scoped_env, recording_table: _RecordingTable
    ) -> None:
        # The caps must not have broken pagination itself: the first request
        # carries no cursor and each later one resumes from the previous page.
        search_documents_handler({"query": "refund", "context": _context()}, None)

        assert "ExclusiveStartKey" not in recording_table.calls[0]
        for index, call in enumerate(recording_table.calls[1:], start=1):
            assert call["ExclusiveStartKey"] == {
                "scope": SERVED_SCOPE,
                "doc_id": f"cursor-{index}",
            }

    def test_truncation_note_echoes_no_query_or_scope(
        self, scoped_env, recording_table: _RecordingTable
    ) -> None:
        # The note is static: it must not become a new echo channel for the
        # model-supplied query or for the injected scope.
        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert "refund" not in result["note"]
        assert SERVED_SCOPE not in result["note"]


class TestSearchPaginationBoundaries:
    """The boundaries a moto-backed partition cannot cheaply produce.

    Each of these distinguishes a correct implementation from a plausible broken
    one, and none is reachable through the moto fixtures above: those exhaust the
    partition in a single page, so no test there pairs a full result set WITH an
    unread page, and none exercises a page that carries a cursor but no items.
    """

    @staticmethod
    def _install(
        monkeypatch: pytest.MonkeyPatch, pages: list[dict[str, Any]]
    ) -> _RecordingTable:
        table = _RecordingTable(pages=pages)
        monkeypatch.setattr(
            search_module, "documents_table_from_event", lambda _event: table
        )
        return table

    def test_cap_reached_with_an_unread_page_is_truncated(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exactly _MAX_RESULTS matches on one page, and a cursor pointing at more.
        # No further match was SEEN, so the per-match check cannot fire -- only the
        # page-boundary check can, and if it were dropped this would report a capped
        # set as complete.
        table = self._install(
            monkeypatch,
            [
                {
                    "Items": [
                        {
                            "scope": SERVED_SCOPE,
                            "doc_id": f"DOC-{index:04d}",
                            "title": f"Refund case {index:04d}",
                        }
                        for index in range(search_module._MAX_RESULTS)
                    ]
                }
            ],
        )

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert len(result["results"]) == search_module._MAX_RESULTS
        assert result["truncated"] is True
        # It stopped as soon as the cap was full: it did not spend a further read.
        assert len(table.calls) == 1

    def test_empty_items_with_a_cursor_keeps_paging_to_the_page_cap(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A page can legitimately return zero items and still carry a cursor (a
        # Limit consumed by non-matching items). The walk must continue -- and must
        # still stop at the page cap rather than following the cursor forever.
        table = self._install(monkeypatch, [{"Items": []}])

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert len(table.calls) == search_module._MAX_PAGE_READS
        assert result["results"] == []
        assert result["truncated"] is True

    def test_absent_items_key_with_a_cursor_is_handled(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The handler reads Items via .get(), so a response that omits the key
        # entirely must not raise -- it would surface as the generic error and look
        # like a credential failure.
        table = self._install(monkeypatch, [{}])

        result = search_documents_handler(
            {"query": "refund", "context": _context()}, None
        )

        assert len(table.calls) == search_module._MAX_PAGE_READS
        assert set(result) == {"results", "truncated", "note"}
        assert result["results"] == []
        assert result["truncated"] is True


# ---------------------------------------------------------------------------
# reply -- per-entry body bound
# ---------------------------------------------------------------------------


def _conversation(documents_table: Any, doc_id: str) -> list[str]:
    """Return the stored ``conversation`` list for a served-scope document."""
    item = documents_table.get_item(
        Key={"scope": SERVED_SCOPE, "doc_id": doc_id}
    ).get("Item", {})
    return item.get("conversation", [])


class TestReplyBodyBound:
    """``body`` is byte-length-bounded before the write, at the exact boundary."""

    def test_body_at_the_limit_is_accepted(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])
        body = "x" * reply_module._MAX_BODY_BYTES

        result = reply_handler(
            {"doc_id": "PAY-001", "body": body, "context": _context()}, None
        )

        assert result == {"success": True}
        assert _conversation(documents_table, "PAY-001") == [body]

    def test_body_over_the_limit_is_rejected_and_writes_nothing(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])
        body = "x" * (reply_module._MAX_BODY_BYTES + 1)

        result = reply_handler(
            {"doc_id": "PAY-001", "body": body, "context": _context()}, None
        )

        assert result == {"error": reply_module._BODY_TOO_LONG_ERROR}
        # The bound is enforced BEFORE the write: no conversation was created, so
        # the rejected call consumed no write capacity.
        assert _conversation(documents_table, "PAY-001") == []

    def test_bound_is_measured_in_utf8_bytes_not_characters(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        # DynamoDB sizes a String by its UTF-8 byte length, and the product
        # guarantee (_MAX_BODY_BYTES * _MAX_CONVERSATION_LEN < 400 KB) only holds if
        # the bound counts the same thing. A character-count check would accept this
        # body -- 1001 code points, comfortably under the limit as characters -- and
        # store 4004 bytes, letting a full conversation reach four times its
        # intended ceiling.
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])
        body = "\U0001f600" * (reply_module._MAX_BODY_BYTES // 4 + 1)
        assert len(body) < reply_module._MAX_BODY_BYTES
        assert len(body.encode("utf-8")) > reply_module._MAX_BODY_BYTES

        result = reply_handler(
            {"doc_id": "PAY-001", "body": body, "context": _context()}, None
        )

        assert result == {"error": reply_module._BODY_TOO_LONG_ERROR}
        assert _conversation(documents_table, "PAY-001") == []

    def test_the_two_bounds_multiply_to_under_the_item_limit(self) -> None:
        # The guarantee itself, not an implementation detail: a list bound alone
        # would still let a document be driven past DynamoDB's 400 KB item limit one
        # legal reply at a time, after which every further write to it hard-fails.
        # 320 KB leaves headroom for the document's own title and body, the keys,
        # and DynamoDB's per-element list overhead.
        worst_case_bytes = (
            reply_module._MAX_BODY_BYTES * reply_module._MAX_CONVERSATION_LEN
        )
        assert worst_case_bytes <= 320 * 1024, (
            f"{reply_module._MAX_BODY_BYTES} bytes x "
            f"{reply_module._MAX_CONVERSATION_LEN} entries = {worst_case_bytes} "
            "bytes, which leaves too little headroom under the 400 KB item limit"
        )

    def test_rejection_echoes_no_argument_or_credential(
        self, scoped_env, put_documents
    ) -> None:
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])
        body = "sentinel-body-text" + "x" * reply_module._MAX_BODY_BYTES

        result = reply_handler(
            {"doc_id": "PAY-001", "body": body, "context": _context()}, None
        )

        message = result["error"]
        # The new specific message must hold the same line as the generic one: it
        # names a static limit only, never the arguments, scope or credentials.
        assert set(result) == {"error"}
        assert "sentinel-body-text" not in message
        assert "PAY-001" not in message
        assert SERVED_SCOPE not in message
        for secret in (_ACCESS_KEY_ID, _SECRET_ACCESS_KEY, _SESSION_TOKEN):
            assert secret not in message


# ---------------------------------------------------------------------------
# reply -- conversation length bound
# ---------------------------------------------------------------------------


class TestReplyConversationBound:
    """The conversation list length is bounded server-side by the update's condition."""

    def test_first_reply_on_a_seeded_document_is_accepted(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        # ``conversation`` is absent on seed, so the condition's
        # attribute_not_exists branch is what admits the very first reply. Without
        # it the bound would reject every document's opening reply.
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])

        result = reply_handler(
            {"doc_id": "PAY-001", "body": "first", "context": _context()}, None
        )

        assert result == {"success": True}
        assert _conversation(documents_table, "PAY-001") == ["first"]

    def test_reply_that_exactly_reaches_the_cap_is_accepted(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        # Boundary: a list one short of the cap still accepts an append.
        item = make_document(SERVED_SCOPE, "PAY-001")
        item["conversation"] = [
            f"entry-{index}" for index in range(reply_module._MAX_CONVERSATION_LEN - 1)
        ]
        put_documents([item])

        result = reply_handler(
            {"doc_id": "PAY-001", "body": "last", "context": _context()}, None
        )

        assert result == {"success": True}
        stored = _conversation(documents_table, "PAY-001")
        assert len(stored) == reply_module._MAX_CONVERSATION_LEN
        assert stored[-1] == "last"

    def test_full_conversation_rejects_further_replies_without_growing(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        item = make_document(SERVED_SCOPE, "PAY-001")
        item["conversation"] = [
            f"entry-{index}" for index in range(reply_module._MAX_CONVERSATION_LEN)
        ]
        put_documents([item])

        result = reply_handler(
            {"doc_id": "PAY-001", "body": "one too many", "context": _context()}, None
        )

        # Distinct from the generic error so the model stops retrying an append
        # that can never succeed. The message names neither cause of a condition
        # failure -- see tests/test_reply_target_must_exist.py, which asserts the
        # same string for the absent-document cause.
        assert result == {"error": reply_module._APPEND_REFUSED_ERROR}
        # The condition is what refused the write, so the item did not grow.
        stored = _conversation(documents_table, "PAY-001")
        assert len(stored) == reply_module._MAX_CONVERSATION_LEN
        assert "one too many" not in stored

    def test_other_client_errors_stay_generic(
        self, scoped_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only ConditionalCheckFailedException maps to the conversation-full
        # message; every other ClientError must keep returning the generic error,
        # so a throttle or an access denial never leaks a reason to the model.
        class _FailingTable:
            def update_item(self, **_kwargs: Any) -> None:
                raise ClientError(
                    {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
                    "UpdateItem",
                )

        monkeypatch.setattr(
            reply_module, "documents_table_from_event", lambda _event: _FailingTable()
        )

        result = reply_handler(
            {"doc_id": "PAY-001", "body": "hello", "context": _context()}, None
        )

        assert result == {"error": reply_module._GENERIC_ERROR}


# ---------------------------------------------------------------------------
# Published limits must not drift from the enforced ones
# ---------------------------------------------------------------------------
# Moved to tests/test_published_limits.py. Those assertions are about the
# model-facing CONTRACT in cdk/gateway_resources.py rather than about handler
# behaviour, and this module was over the repo's 400-line limit.

