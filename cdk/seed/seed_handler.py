"""
Seed custom resource handler (CDK Provider on_event_handler).

This Lambda handles CloudFormation custom resource lifecycle events
(Create/Update/Delete) for seeding the Documents_Table with the generated
document list. It receives the document items as a property in the event
and writes them with boto3 Table.batch_writer (25-item batches, implicit
UnprocessedItems retries).

When used with CDK's custom_resources.Provider, this handler returns a dict
with PhysicalResourceId (the Provider framework handles the cfnresponse
callback automatically).

Deterministic ids make re-seeding idempotent: PutItem on the same key
overwrites rather than duplicates.

Functions:
    handler: CDK Provider on_event_handler.

Citation:
    - DynamoDB BatchWriteItem (25-item batches):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithItems.html
    - boto3 Table.batch_writer (implicit batching and retries):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/programming-with-python.html
    - CDK custom_resources.Provider on_event_handler contract:
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.custom_resources/Provider.html
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PHYSICAL_RESOURCE_ID = "documents-seed"


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle CloudFormation custom resource lifecycle events.

    On Create and Update, writes the document items to the Documents_Table
    using Table.batch_writer which handles 25-item batching and implicit
    UnprocessedItems retries.

    On Delete, does nothing (seed data is removed when the table is deleted
    via RemovalPolicy.DESTROY).

    The document list is passed via the event's ResourceProperties.Documents
    field as a JSON-encoded string (CloudFormation custom resource properties
    must be strings).

    Args:
        event: CloudFormation custom resource event with RequestType and
            ResourceProperties.
        context: Lambda context object.

    Returns:
        Dict with PhysicalResourceId (required by CDK Provider framework).

    Raises:
        ValueError: If required properties are missing.
    """
    request_type = event.get("RequestType", "")
    logger.info("Seed handler invoked: RequestType=%s", request_type)

    if request_type in ("Create", "Update"):
        _seed_documents(event)

    # On Delete, do nothing. Table removal handles cleanup.
    return {"PhysicalResourceId": PHYSICAL_RESOURCE_ID}


def _seed_documents(event: Dict[str, Any]) -> None:
    """Write document items to the Documents_Table via batch_writer.

    Uses boto3 Table.batch_writer context manager which:
    - Buffers writes and splits into 25-item batches automatically
    - Retries UnprocessedItems implicitly

    Deterministic ids mean re-seeding overwrites the same keys (idempotent).

    Args:
        event: The CloudFormation event containing ResourceProperties.

    Raises:
        ValueError: If table name or documents are missing from the event.
    """
    properties = event.get("ResourceProperties", {})
    table_name = properties.get("TableName")
    documents_json = properties.get("Documents")

    if not table_name:
        raise ValueError("TableName not provided in ResourceProperties")
    if not documents_json:
        raise ValueError("Documents not provided in ResourceProperties")

    documents = json.loads(documents_json)
    logger.info(
        "Seeding %d documents to table '%s'", len(documents), table_name
    )

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    with table.batch_writer() as batch:
        for doc in documents:
            batch.put_item(Item=doc)

    logger.info("Seed write complete: %d items written", len(documents))
