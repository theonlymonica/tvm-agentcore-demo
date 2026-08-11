"""``reply`` must not CREATE documents -- only append to ones that exist.

The gap these tests close: the write's partition key comes from the interceptor and
is trustworthy, but ``doc_id`` comes from the model, out of text it has just read.
``UpdateItem`` is an upsert -- it "adds a new item to the table if it does not
already exist" -- so before this change an in-scope injection could do more than
append to existing documents: it could materialise new ones with invented ids
inside the served partition. The fix is one clause,
``attribute_exists(doc_id)``, in the ``ConditionExpression`` that was already on
that call.

Why the tests must be TABLE-BACKED. The kwarg-discarding stub tables used in
``tests/test_handler_transport_errors.py`` never evaluate an expression, so a
condition that parsed into something harmless would pass every assertion made
against them. Everything here runs against moto so the real semantics decide the
outcome, and no assertion inspects the expression string: each one fails if the
condition stops DOING its job, not if its text changes.

The anti-vacuity argument, spelled out, because "no item was created" is exactly
the kind of assertion that also passes when nothing is being tested:

  * ``TestReplyRefusesToCreate`` shows the handler returns the refusal message and
    creates nothing. The message is only reachable via
    ``ConditionalCheckFailedException``, so moto must have evaluated the condition
    and rejected the write.
  * ``TestTheConditionIsWhatRefuses`` is the positive control. It issues the SAME
    ``UpdateItem`` against the SAME moto table, once with the condition this change
    replaced and once with no condition at all, and both create the item. So moto
    does implement the upsert, and the absence above is attributable to the added
    clause rather than to moto ignoring conditions or to some unrelated failure.

What is deliberately NOT covered: writes to other documents that DO exist in the
caller's own partition. Those remain possible, are outside the claim, and are a
known residual risk. Nothing here asserts a cross-scope property either
-- moto does not evaluate the ``LeadingKeys`` session policy, so a test using these
credentials could not tell an enforced partition boundary from an unenforced one.
``TestForeignPartitionIdIsNotMaterialised`` is about the SERVED partition only: the
shape it rules out is an invented-but-real-elsewhere id being created locally.
"""

from __future__ import annotations

from typing import Any

import reply.handler as reply_module
from reply.handler import handler as reply_handler
from tests.conftest import SERVED_SCOPE, make_document

# Matches the credential shapes the other tool tests use, so a fixture never
# contradicts the wire contract. moto accepts any credential values.
_ACCESS_KEY_ID = "ASIAIOSFODNN7EXAMPLE"
_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEXAMPLETOKEN"

# The id no seeded document uses -- the "invented by the model" case.
_INVENTED_ID = "PAY-999-INVENTED"

# The UpdateExpression the handler issues. The controls reuse it verbatim so they
# differ from the handler in the CONDITION only.
_UPDATE_EXPRESSION = (
    "SET conversation = list_append(if_not_exists(conversation, :empty), :entry)"
)

# The condition as it stood BEFORE this change: the conversation cap alone. On an
# absent item its attribute_not_exists branch is true, which is precisely how an
# invented doc_id used to materialise a new document.
_CAP_ONLY_CONDITION = (
    "attribute_not_exists(conversation) OR size(conversation) < :max_entries"
)


def _context(scope: str = SERVED_SCOPE) -> dict[str, Any]:
    """Return a well-formed injected ``context`` object for ``scope``.

    Args:
        scope: The served scope to advertise as authoritative.

    Returns:
        The ``context`` object shape the REQUEST interceptor injects.
    """
    return {
        "served_scope": scope,
        "tenant_credentials": {
            "access_key_id": _ACCESS_KEY_ID,
            "secret_access_key": _SECRET_ACCESS_KEY,
            "session_token": _SESSION_TOKEN,
        },
    }


def _item(table: Any, doc_id: str, *, scope: str = SERVED_SCOPE) -> dict[str, Any]:
    """Return the stored item for a composite key, or {} when absent.

    Args:
        table: The moto-backed Documents table.
        doc_id: The sort key to read.
        scope: The partition key to read.

    Returns:
        The stored item, or an empty dict when no such item exists.
    """
    return table.get_item(Key={"scope": scope, "doc_id": doc_id}).get("Item", {})


def _reply(doc_id: str, body: str = "appended by the model") -> dict[str, Any]:
    """Invoke the reply handler for a served-scope document.

    Args:
        doc_id: The model-supplied identifier to write to.
        body: The reply body.

    Returns:
        The handler's response dict.
    """
    return reply_handler(
        {"doc_id": doc_id, "body": body, "context": _context()}, None
    )


class TestReplyRefusesToCreate:
    """A ``doc_id`` naming no existing document is refused, not materialised."""

    def test_invented_doc_id_creates_no_item(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        # A real seeded document exists in the partition, so the write is being
        # refused on the TARGET's absence rather than on an empty partition.
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])

        result = _reply(_INVENTED_ID)

        assert result == {"error": reply_module._APPEND_REFUSED_ERROR}
        # The assertion the whole change exists for.
        assert _item(documents_table, _INVENTED_ID) == {}

    def test_refusal_is_reached_through_the_condition_not_an_input_check(
        self, scoped_env, put_documents
    ) -> None:
        # _APPEND_REFUSED_ERROR is returned from ONE place: the
        # ConditionalCheckFailedException branch. Distinguishing it from the generic
        # error is what proves the request reached DynamoDB and was rejected there,
        # rather than being screened out by a local validity check -- which is the
        # difference between a property of the data and a branch in the handler.
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])

        result = _reply(_INVENTED_ID)

        assert result["error"] != reply_module._GENERIC_ERROR
        assert result["error"] != reply_module._BODY_TOO_LONG_ERROR

    def test_repeated_attempts_never_accumulate_an_item(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        # An injected model that ignores the refusal and retries must not get there
        # by persistence: the condition is stateless, so attempt N behaves like
        # attempt 1.
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])

        for _ in range(5):
            assert _reply(_INVENTED_ID) == {
                "error": reply_module._APPEND_REFUSED_ERROR
            }

        assert _item(documents_table, _INVENTED_ID) == {}


class TestTheConditionIsWhatRefuses:
    """Positive control: the same write without the clause DOES create the item.

    Without these, ``TestReplyRefusesToCreate`` would pass just as happily against a
    moto that silently ignored ``ConditionExpression``, or against any unrelated
    failure that also happened to leave the table untouched.
    """

    def test_the_previous_cap_only_condition_would_have_created_it(
        self, scoped_env, documents_table
    ) -> None:
        documents_table.update_item(
            Key={"scope": SERVED_SCOPE, "doc_id": _INVENTED_ID},
            UpdateExpression=_UPDATE_EXPRESSION,
            ConditionExpression=_CAP_ONLY_CONDITION,
            ExpressionAttributeValues={
                ":empty": [],
                ":entry": ["appended by the model"],
                ":max_entries": reply_module._MAX_CONVERSATION_LEN,
            },
        )

        # The bug, reproduced under real DynamoDB semantics: the cap condition
        # admits the write, and UpdateItem's upsert creates a document that never
        # existed -- with only the model-supplied id and the reply on it.
        created = _item(documents_table, _INVENTED_ID)
        assert created.get("conversation") == ["appended by the model"]
        assert "body" not in created

    def test_an_unconditioned_update_would_have_created_it(
        self, scoped_env, documents_table
    ) -> None:
        documents_table.update_item(
            Key={"scope": SERVED_SCOPE, "doc_id": _INVENTED_ID},
            UpdateExpression=_UPDATE_EXPRESSION,
            ExpressionAttributeValues={
                ":empty": [],
                ":entry": ["appended by the model"],
            },
        )

        assert _item(documents_table, _INVENTED_ID).get("conversation") == [
            "appended by the model"
        ]


class TestExistingDocumentsAreUnaffected:
    """The clause must not be satisfiable only by over-tightening the write."""

    def test_existing_document_still_accepts_a_reply(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        item = make_document(SERVED_SCOPE, "PAY-001")
        item["conversation"] = ["earlier"]
        put_documents([item])

        assert _reply("PAY-001", "later") == {"success": True}
        assert _item(documents_table, "PAY-001")["conversation"] == [
            "earlier",
            "later",
        ]

    def test_seeded_document_without_a_conversation_accepts_its_first_reply(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        # ``conversation`` is absent on seed. A fix keyed on that attribute instead
        # of on the sort key -- attribute_exists(conversation) -- would also make
        # the invented-id test pass, while rejecting every real document's opening
        # reply. This is the test that separates the two.
        put_documents([make_document(SERVED_SCOPE, "PAY-001")])

        assert _reply("PAY-001", "first") == {"success": True}
        assert _item(documents_table, "PAY-001")["conversation"] == ["first"]

    def test_the_cap_still_rejects_on_an_existing_document(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        # Composition check: the new clause is ANDed in front of the cap's OR pair.
        # A missing pair of parentheses is NOT what this catches (both forms
        # evaluate identically -- see the note); what it catches is the cap clause
        # being dropped or weakened while the existence clause was added.
        item = make_document(SERVED_SCOPE, "PAY-001")
        item["conversation"] = [
            f"entry-{index}" for index in range(reply_module._MAX_CONVERSATION_LEN)
        ]
        put_documents([item])

        result = _reply("PAY-001", "one too many")

        assert result == {"error": reply_module._APPEND_REFUSED_ERROR}
        stored = _item(documents_table, "PAY-001")["conversation"]
        assert len(stored) == reply_module._MAX_CONVERSATION_LEN
        assert "one too many" not in stored


class TestForeignPartitionIdIsNotMaterialised:
    """An id that exists elsewhere is still absent HERE, so it cannot be created."""

    def test_id_from_a_foreign_partition_creates_nothing_in_the_served_one(
        self, scoped_env, documents_table, put_documents
    ) -> None:
        # The realistic injection shape: the payload names a plausible id rather
        # than a random one. Under the served scope that id addresses a DIFFERENT
        # item -- one that does not exist -- so the condition refuses it.
        put_documents([
            make_document(SERVED_SCOPE, "PAY-001"),
            make_document("billing-internal", "BIL-002"),
        ])

        result = _reply("BIL-002")

        assert result == {"error": reply_module._APPEND_REFUSED_ERROR}
        assert _item(documents_table, "BIL-002") == {}
        # And the real document in the other partition was not touched either.
        foreign = _item(documents_table, "BIL-002", scope="billing-internal")
        assert "conversation" not in foreign


class TestRefusalMessageNamesNeitherCause:
    """The one message must stay true of both causes of a condition failure."""

    def test_the_message_does_not_name_the_conversation_cap(self) -> None:
        # It used to. Once the existence clause landed, a message naming the cap
        # asserted something specific and false about a document that does not
        # exist -- and that string is returned to the model, so it enters the
        # model's context and can reach what the operator reads. The cap is
        # published in the tool description instead
        # (tests/test_published_limits.py).
        message = reply_module._APPEND_REFUSED_ERROR
        assert str(reply_module._MAX_CONVERSATION_LEN) not in message
        assert "conversation" not in message.lower()

    def test_the_message_is_terminal_rather_than_transient(self) -> None:
        # What stopped the model retrying was never that the text named the cap; it
        # was that the failure is permanent. Keeping it distinct from the generic
        # error is what carries that, under either cause.
        assert reply_module._APPEND_REFUSED_ERROR != reply_module._GENERIC_ERROR
        assert "no reply can be appended" in reply_module._APPEND_REFUSED_ERROR
