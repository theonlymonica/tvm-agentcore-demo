"""
Data layer resources for the multi-tenant data isolation demo.

This module provisions:
- DocumentsTable (DynamoDB, COMPOSITE key: PK scope, SK doc_id):
  stores seed documents partitioned by the scope that owns them. The
  scope-as-partition-key shape is what makes dynamodb:LeadingKeys
  fine-grained access control possible: a per-request STS session policy
  can confine reads/writes to a single served partition.
- Seed custom resource: writes generated documents via BatchWriteItem
  using boto3 Table.batch_writer (25-item batches, implicit retries).

The DocumentsTable items carry scope, doc_id, title, body
attributes. The `conversation` List attribute is absent/empty on seed
(DynamoDB is schemaless, so it need not be declared in the table
definition); the reply tool's UpdateItem list_append initializes it on
first use.

Functions:
    create_data_resources: Provision all data-layer resources and return refs.

Citation:
    - aws_cdk.aws_dynamodb.Table / TableProps (partition_key + sort_key as
      aws_cdk.aws_dynamodb.Attribute; billing_mode; removal_policy; table_name):
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_dynamodb/TableProps.html
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_dynamodb/Table.html
    - aws_cdk.aws_dynamodb.Attribute / AttributeType.STRING:
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_dynamodb/Attribute.html
    - aws_cdk.custom_resources.Provider:
      https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.custom_resources/Provider.html
    - DynamoDB BatchWriteItem (25-item limit, UnprocessedItems):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithItems.html
    - boto3 Table.batch_writer (implicit batching/retries):
      https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/programming-with-python.html
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import aws_cdk as cdk
import aws_cdk.aws_dynamodb as dynamodb
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as lambda_
import aws_cdk.custom_resources as cr
from constructs import Construct

from asset_packaging import python_lambda_code
from documents_roles import create_documents_roles
from observability import lambda_log_group
from seed.documents_seed_generator import generate_documents


@dataclass
class DataResources:
    """Container for the data-layer resource references.

    Attributes:
        documents_table: The DocumentsTable DynamoDB table (composite key:
            PK scope, SK doc_id).
        documents_table_arn: The DocumentsTable ARN. Exposed so downstream
            wiring (env vars for the scoped-credentials helper and the
            DocumentsAccessRole / DocumentsWriteRole identity policies) can
            reference the table by ARN without re-deriving it.
        documents_access_role: The DocumentsAccessRole construct (READ:
            GetItem/Query). Exposed so the downstream Lambda wiring can attach
            the read-tool exec roles as trust principals and add sts:AssumeRole
            grants. See documents_roles.py.
        documents_access_role_arn: The DocumentsAccessRole ARN (for the
            DOCUMENTS_ACCESS_ROLE_ARN env var / read-tool sts:AssumeRole grant).
        documents_write_role: The DocumentsWriteRole construct (WRITE:
            UpdateItem only). Exposed so the downstream Lambda wiring can attach
            the reply exec role as the sole trust principal and add its
            sts:AssumeRole grant.
        documents_write_role_arn: The DocumentsWriteRole ARN (for the
            DOCUMENTS_WRITE_ROLE_ARN env var / reply sts:AssumeRole grant).
    """

    documents_table: dynamodb.Table
    documents_table_arn: str
    documents_access_role: iam.Role
    documents_access_role_arn: str
    documents_write_role: iam.Role
    documents_write_role_arn: str


def create_data_resources(scope: Construct) -> DataResources:
    """Provision the data-layer resources for the demo.

    Creates:
    1. DocumentsTable — composite key PK scope (S) + SK doc_id (S),
       PAY_PER_REQUEST, DESTROY. Scope-partitioned so dynamodb:LeadingKeys
       fine-grained access control can confine access to one partition.
    2. Seed custom resource — calls generate_documents() at synth time,
       passes items to a Lambda that writes them with batch_writer.
    3. DocumentsAccessRole (read) + DocumentsWriteRole (write) — scoped
       DynamoDB roles the tool Lambdas assume with per-request LeadingKeys
       session policies (see documents_roles.py). Trust principals are wired
       later, when the tool Lambdas exist.

    Args:
        scope: The CDK Stack or Construct to attach resources to.

    Returns:
        DataResources with references to all provisioned resources, including
        the DocumentsTable ARN and the scoped read/write role constructs + ARNs
        for downstream IAM/env wiring.
    """
    documents_table = _create_documents_table(scope)
    _create_seed_custom_resource(scope, documents_table)

    # Scoped DynamoDB roles. Created here so the read/write role constructs +
    # ARNs are available on DataResources for the downstream Lambda wiring;
    # their trust policies are intentionally left as a temporary
    # AccountPrincipal placeholder for that wiring to narrow (see
    # documents_roles.py).
    documents_roles = create_documents_roles(scope, documents_table.table_arn)

    return DataResources(
        documents_table=documents_table,
        documents_table_arn=documents_table.table_arn,
        documents_access_role=documents_roles.documents_access_role,
        documents_access_role_arn=documents_roles.documents_access_role_arn,
        documents_write_role=documents_roles.documents_write_role,
        documents_write_role_arn=documents_roles.documents_write_role_arn,
    )


def _create_documents_table(scope: Construct) -> dynamodb.Table:
    """Create the DocumentsTable DynamoDB table with a composite key.

    Schema:
        - partition_key: scope (S) — the scope that owns the document
          (e.g. payments-core, billing-internal). Chosen as PK so
          dynamodb:LeadingKeys can gate access per partition. `scope` is a
          DynamoDB reserved word, so any expression naming it must use a
          Key={...} map, a Key("scope") builder, or a #scope alias.
        - sort_key: doc_id (S) — the document identifier (e.g. PAY-001).
        - table_name (physical): toxic-flow-documents — follows the
          toxic-flow-* naming convention. (The CDK construct id / logical id
          remains "DocumentsTable".)
        - billing_mode: PAY_PER_REQUEST
        - removal_policy: DESTROY
        - encryption: AWS_MANAGED
        - point-in-time recovery: enabled

    Additional attributes (title, body, conversation) are not declared in the
    table definition because DynamoDB is schemaless; they exist on the items
    written by the seed and the reply tool.

    Durability / protection settings
    --------------------------------
    Two of the three candidate settings are applied here; the third is
    deliberately rejected because it contradicts a tested requirement.

    * ``encryption=AWS_MANAGED`` — moves the table off the AWS-OWNED key (the
      default, invisible to the account) onto the account's ``aws/dynamodb``
      managed key, so key usage is attributable in CloudTrail. A
      CUSTOMER-MANAGED key was considered and NOT taken: it bills monthly, and
      KMS enforces a 7-30 day pending-deletion window, so every ``cdk destroy``
      of this deliberately ephemeral stack would leave a billing artefact behind.
      Nothing here needs a custom key policy, grants, or cross-account access —
      the three things a CMK actually buys.
    * ``point_in_time_recovery_specification`` — enabled. Cheap on a
      PAY_PER_REQUEST table holding a few dozen fabricated documents, and it
      makes the reply tool's ``UpdateItem`` writes recoverable, which matters
      when a test or injection run corrupts the seeded state.
    * ``deletion_protection`` — deliberately NOT enabled. It is INCOMPATIBLE
      with this table's ``RemovalPolicy.DESTROY``: DynamoDB refuses to delete a
      protected table, so stack teardown would fail. Deleting the table on
      teardown is a requirement pinned by
      ``tests/test_synth_config.py::test_deletion_policy_is_delete``, and
      ``tests/test_synth_operational_posture.py`` asserts the absence of
      deletion protection so a future change cannot introduce the deadlock.

    Args:
        scope: The CDK Stack or Construct.

    Returns:
        The DocumentsTable construct.
    """
    table = dynamodb.Table(
        scope,
        "DocumentsTable",
        table_name="toxic-flow-documents",
        partition_key=dynamodb.Attribute(
            name="scope", type=dynamodb.AttributeType.STRING
        ),
        sort_key=dynamodb.Attribute(
            name="doc_id", type=dynamodb.AttributeType.STRING
        ),
        billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
        encryption=dynamodb.TableEncryption.AWS_MANAGED,
        point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
            point_in_time_recovery_enabled=True,
        ),
        removal_policy=cdk.RemovalPolicy.DESTROY,
    )
    return table


def _create_seed_custom_resource(
    scope: Construct,
    documents_table: dynamodb.Table,
) -> None:
    """Create the seed custom resource that populates DocumentsTable.

    Approach:
    1. Call generate_documents() at CDK synth time to produce the item list
    2. Pass the items as a JSON string in the custom resource properties
    3. A Lambda handler receives the items and writes them with
       Table.batch_writer (handles 25-item batching and retries)

    NOTE: Under the composite-key schema the seed items MUST carry both
    scope (PK) and doc_id (SK). Producing that shape is owned by
    the seed generator; this custom resource passes through whatever
    generate_documents() returns.

    Deterministic ids make re-seeding idempotent: PutItem on the same
    composite key overwrites rather than duplicates.

    Args:
        scope: The CDK Stack or Construct.
        documents_table: The DocumentsTable to seed.
    """
    # Generate documents at synth time (runs the post-generation verification)
    documents = generate_documents()
    documents_json = json.dumps(documents)

    # Create the seed handler Lambda
    seed_handler_dir = os.path.join(os.path.dirname(__file__), "seed")

    seed_fn = lambda_.Function(
        scope,
        "DocumentsSeedFunction",
        function_name="toxic-flow-documents-seed",
        runtime=lambda_.Runtime.PYTHON_3_14,
        handler="seed_handler.handler",
        code=python_lambda_code(seed_handler_dir),
        timeout=cdk.Duration.minutes(2),
        # Bounded, stack-owned log group.
        log_group=lambda_log_group(
            scope,
            "DocumentsSeedFunctionLogGroup",
            function_name="toxic-flow-documents-seed",
        ),
        environment={
            "DOCUMENTS_TABLE_NAME": documents_table.table_name,
        },
    )

    # Grant the seed Lambda EXACTLY the two write actions it performs.
    #
    # This grant was narrowed from `documents_table.grant_write_data(seed_fn)`,
    # which expands to BatchWriteItem + PutItem + UpdateItem + the item-removal
    # action + DescribeTable. The seeder only ever writes: seed_handler.py drives
    # `Table.batch_writer`, so on the wire it issues BatchWriteItem alone, and
    # PutItem is kept for the documented single-item path. UpdateItem and
    # item removal were conferred and never used, so a compromised or buggy
    # seeder could have mutated or destroyed seeded documents rather than only
    # (re)writing them.
    #
    # Deterministic ids keep re-seeding idempotent: a batch put on the same
    # composite key overwrites rather than duplicates, so no update action is
    # needed for re-seeds.
    #
    # DescribeTable is also dropped: boto3's batch_writer only needs it when
    # `overwrite_by_pkeys` is passed, which the seeder does not do.
    seed_fn.add_to_role_policy(
        iam.PolicyStatement(
            actions=[
                "dynamodb:BatchWriteItem",
                "dynamodb:PutItem",
            ],
            resources=[documents_table.table_arn],
        )
    )

    # Create the Provider framework
    provider = cr.Provider(
        scope,
        "DocumentsSeedProvider",
        on_event_handler=seed_fn,
    )

    # Create the custom resource that triggers the seed
    cdk.CustomResource(
        scope,
        "DocumentsSeed",
        service_token=provider.service_token,
        properties={
            "TableName": documents_table.table_name,
            "Documents": documents_json,
            # Change this value to force re-seeding on update
            "SeedVersion": "1",
        },
    )
