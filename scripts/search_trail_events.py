"""Search the audit trail's gzipped CloudTrail objects.

WHY THIS EXISTS.

CloudTrail writes data events as gzipped JSON into S3, one object per batch, under a
date-partitioned prefix. That makes them awkward to search by hand in three separate
ways, and each one has bitten this project:

- They are COMPRESSED, so `aws s3 cp` + a text search needs a decompression step and
  a temporary file per object.
- Each object holds a `Records` ARRAY, so a match is a needle inside one element of
  one array inside one of many files — grep tells you the file matched and nothing
  more useful than that.
- The CloudTrail console does NOT show them. Event history covers management events
  only, so the events this project cares about are invisible there. That is not a
  permissions problem and no amount of clicking finds them.

Athena is the documented way to query this at scale, but the CloudTrail SerDe needs a
schema that matches the records exactly, and getting it wrong fails with
`HIVE_BAD_DATA` rather than with anything actionable (see notes/audit-queries.sql).
This script is the direct route: stream the objects, decompress in memory, filter, and
print. No table, no schema, no temp files.

Usage:
    # every DynamoDB data event mentioning a document, today
    python scripts/search_trail_events.py --contains PAY-001

    # who touched the table over the last three days, one line each
    python scripts/search_trail_events.py --days 3 --source dynamodb --summary

    # the agent invocations, full JSON
    python scripts/search_trail_events.py --source bedrock-agentcore --raw

The bucket is read from the CloudFormation stack, so there is no hardcoded name and
no account id in this file.

Environment:
    AWS_PROFILE / AWS_REGION are used as usual by boto3. STACK_NAME overrides the
    stack the bucket is discovered from (default ScopedCredentialsStack).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import boto3

#: Stack whose outputs name the trail bucket.
_DEFAULT_STACK = "ScopedCredentialsStack"

#: Logical id of the trail bucket inside that stack. Matched by prefix because CDK
#: appends a hash to the logical id.
_BUCKET_LOGICAL_PREFIX = "AuditTrailBucket"

#: Fields printed by --summary. Chosen to answer "who did what to which item",
#: which is the question this trail exists for.
_SUMMARY_FIELDS = (
    "eventTime",
    "eventName",
    "readOnly",
    "sourceIdentity",
    "roleSession",
    "table",
    "key",
    "caller",
)


def _resolve_bucket(session: boto3.Session, stack_name: str) -> str:
    """Find the trail bucket's physical name from the stack.

    Args:
        session: The boto3 session to use.
        stack_name: CloudFormation stack that owns the bucket.

    Returns:
        The bucket name.

    Raises:
        SystemExit: If the bucket cannot be found, with a message rather than a
            traceback — a wrong profile is the usual cause and a stack trace hides it.
    """
    cfn = session.client("cloudformation")
    try:
        resources = cfn.describe_stack_resources(StackName=stack_name)[
            "StackResources"
        ]
    except Exception as exc:  # noqa: BLE001 - message beats traceback here
        sys.exit(f"could not read stack {stack_name}: {exc}")

    for resource in resources:
        if resource["LogicalResourceId"].startswith(_BUCKET_LOGICAL_PREFIX):
            return resource["PhysicalResourceId"]
    sys.exit(
        f"no resource starting with {_BUCKET_LOGICAL_PREFIX} in stack {stack_name}"
    )


def _date_prefixes(region: str, account: str, days: int) -> Iterator[str]:
    """Yield one S3 prefix per day, newest first.

    Args:
        region: Region segment of the trail's key layout.
        account: Account id segment.
        days: How many days back to cover, including today.

    Yields:
        Key prefixes like ``AWSLogs/<account>/CloudTrail/<region>/2026/09/09/``.
    """
    today = datetime.now(timezone.utc).date()
    for offset in range(days):
        day = today - timedelta(days=offset)
        yield (
            f"AWSLogs/{account}/CloudTrail/{region}/"
            f"{day.year:04d}/{day.month:02d}/{day.day:02d}/"
        )


def _iter_records(
    session: boto3.Session,
    bucket: str,
    prefixes: Iterator[str],
) -> Iterator[dict[str, Any]]:
    """Stream every CloudTrail record under the given prefixes.

    Objects are decompressed in memory — nothing is written to disk, so there is no
    scratch directory to clean up and no half-downloaded state if this is interrupted.

    Args:
        session: The boto3 session to use.
        bucket: Trail bucket name.
        prefixes: Key prefixes to walk.

    Yields:
        One CloudTrail record dict at a time.
    """
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for prefix in prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json.gz"):
                    continue
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                try:
                    payload = json.loads(gzip.decompress(body))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"skipping unreadable {key}: {exc}", file=sys.stderr)
                    continue
                yield from payload.get("Records", [])


def _summarize(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce one record to the fields that answer the audit question.

    Args:
        record: A CloudTrail record.

    Returns:
        A flat dict with the summary fields; missing values become None.
    """
    identity = record.get("userIdentity") or {}
    context = identity.get("sessionContext") or {}
    params = record.get("requestParameters") or {}
    arn = identity.get("arn") or ""

    return {
        "eventTime": record.get("eventTime"),
        "eventName": record.get("eventName"),
        "readOnly": record.get("readOnly"),
        # The Cognito sub — the human. Present on events made with a vended session.
        "sourceIdentity": context.get("sourceIdentity"),
        # Last ARN segment: the STS RoleSessionName, which is the join key to the
        # interceptor's own audit record.
        "roleSession": arn.rsplit("/", 1)[-1] if "/" in arn else None,
        "table": params.get("tableName"),
        "key": params.get("key"),
        # For AgentCore events there is no vended session; the caller is the IAM
        # principal that invoked the agent.
        "caller": arn or None,
    }


def _matches(record: dict[str, Any], contains: str | None, source: str | None) -> bool:
    """Decide whether a record passes the filters.

    Args:
        record: A CloudTrail record.
        contains: Case-insensitive substring searched across the whole record.
        source: Substring matched against ``eventSource``.

    Returns:
        True when the record should be printed.
    """
    if source and source.lower() not in (record.get("eventSource") or "").lower():
        return False
    if contains and contains.lower() not in json.dumps(record).lower():
        return False
    return True


def main() -> None:
    """Parse arguments, stream matching records, print them."""
    parser = argparse.ArgumentParser(
        description="Search the audit trail's gzipped CloudTrail objects in S3.",
    )
    parser.add_argument(
        "--contains",
        help="case-insensitive substring to search for anywhere in the record "
        "(e.g. a doc_id, a Cognito sub, a role session name)",
    )
    parser.add_argument(
        "--source",
        help="filter on eventSource, e.g. dynamodb or bedrock-agentcore",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="how many days back to search, including today (default 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="stop after this many matches (default 50)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="print the whole record instead of the audit summary",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print one compact line per match instead of indented JSON",
    )
    args = parser.parse_args()

    session = boto3.Session()
    region = session.region_name
    if not region:
        sys.exit("no region configured; set AWS_REGION or use a profile that has one")

    account = session.client("sts").get_caller_identity()["Account"]
    bucket = _resolve_bucket(session, os.environ.get("STACK_NAME", _DEFAULT_STACK))

    print(
        f"searching s3://{bucket} — {args.days} day(s), region {region}",
        file=sys.stderr,
    )

    found = 0
    for record in _iter_records(
        session, bucket, _date_prefixes(region, account, args.days)
    ):
        if not _matches(record, args.contains, args.source):
            continue

        if args.raw:
            print(json.dumps(record, indent=2, sort_keys=True))
        elif args.summary:
            summary = _summarize(record)
            print(
                "  ".join(
                    f"{field}={summary[field]}"
                    for field in _SUMMARY_FIELDS
                    if summary[field] is not None
                )
            )
        else:
            print(json.dumps(_summarize(record), indent=2, sort_keys=True))

        found += 1
        if found >= args.limit:
            print(f"-- stopped at --limit {args.limit}", file=sys.stderr)
            break

    if not found:
        print(
            "no matching records. Note the trail delivers to S3 in batches and can "
            "lag several minutes behind a live call, and the CloudTrail console's "
            "Event history does NOT show these events at all — it covers management "
            "events only.",
            file=sys.stderr,
        )
    else:
        print(f"-- {found} match(es)", file=sys.stderr)


if __name__ == "__main__":
    main()
