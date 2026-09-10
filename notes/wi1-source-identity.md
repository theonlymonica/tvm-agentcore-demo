# Source identity and the session join key — closing the attribution gap

_Decision record for work item 1 of the auditability track. Written in the house
style of `notes/`: what was claimed, what was actually true, what changed, which
decisions are arguable, and — explicitly — what is still not covered. Identifiers
below are placeholders per `notes/README.md`; the raw unredacted output stays in the
gitignored `evidence/` directory._

## The control and what it is for

This project mints short-lived, partition-scoped DynamoDB credentials per request.
That is a security win and an **attribution loss**, and the second half is what this
work item is about.

The vended credential is a role session on one of two shared roles
(`DocumentsAccessRole`, `DocumentsWriteRole`). Every request produced the *same*
session shape, so the record of a table access looked identical no matter which human
was on the other end. An auditor reading a `dynamodb:UpdateItem` could see that
`DocumentsWriteRole` modified a document and could see which partition — and then
stopped. There was nothing in the record tying the access back to a person, or to the
conversation it happened in.

The chain we need to be able to walk is:

> human → Gateway request → model's reasoning → requested tool → granted scope →
> actual DynamoDB call

Work item 1 covers the two ends of it: **who**, and **what was actually touched**.

## WHY — the credential was anonymous by construction

**Claim under test:** "an action on the documents table can be traced back to the
human who caused it."

**What would falsify it:** any record of a table access from which the person cannot
be recovered without hand-correlating timestamps.

**Exercised?** Yes, live, before changing anything. Three `AssumeRole` events on the
Documents roles, all carrying `roleSessionName: scope-payments-core` and **no**
`sourceIdentity`. The session name was the *scope*, which is the one thing already
visible in the session policy and the session tag — so it added nothing — and it was
identical across users and across requests. Nothing in the event, and nothing in the
subsequent data-plane call, distinguished one human from another.

Baseline transcript: `evidence/wi1/assumerole-baseline-raw.txt` (local).

## What changed

Two STS parameters are now set on the single `AssumeRole` call site
(`interceptor/scoped_credentials.py`):

| Parameter | Value | Purpose |
|---|---|---|
| `SourceIdentity` | the Cognito `sub` from the verified JWT | WHO — immutable, survives role chaining |
| `RoleSessionName` | `gw-` + the Gateway `Mcp-Session-Id` | WHICH REQUEST — the join key |

`sub` was chosen over username or email because it is the only Cognito claim that is
immutable and non-reassignable. A username can be recycled to a different person; an
audit trail keyed on a recycled identifier silently attributes one person's actions to
another.

`SourceIdentity` is the right field rather than a session tag because **the value
cannot be changed once set and persists into chained sessions**
([AWS global condition context keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html)).
A tag can be re-set by whoever holds the credential; the attribution must not be
editable by the code being audited.

Fail-closed: `interceptor/jwt_claims.py` now resolves scope and subject in one
verification (`verified_identity_from_authorization`), and a token without a usable
`sub` yields no credential at all. There is deliberately no "unknown" fallback — an
unattributable vend is refused rather than recorded as anonymous.

The Gateway session id is hashed into nothing and truncated into nothing: it is used
verbatim behind a `gw-` prefix, so the value in CloudTrail is greppable against the
interceptor's own log line without a translation step.

## Finding: `sts:SetSourceIdentity` must be UNCONDITIONED in the trust policy

This is not documented, and it cost a deploy cycle to establish.

The obvious shape — add `sts:SetSourceIdentity` to the existing trust statement
alongside `sts:AssumeRole` and `sts:TagSession`, under the same
`StringLike aws:RequestTag/scope: "*"` condition — makes **every vend fail**:

```
is not authorized to perform: sts:SetSourceIdentity
on resource: arn:aws:iam::123456789012:role/DocumentsAccessRole
```

Both sides granted the action; verified by reading the live policies after the deploy,
not the source. Ruled out IAM propagation delay with a third attempt more than twelve
minutes later. The documentation states only that the trust policy "must have the
`sts:SetSourceIdentity` permission" and that `AssumeRole*` fails without it
([Monitor and control actions taken with assumed roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_monitor.html));
it says nothing about the action being incompatible with a request-tag condition.

**Remedy:** `sts:SetSourceIdentity` lives in its own trust statement with no
condition. Evidence: `evidence/wi1/assumerole-accessdenied-setsourceidentity.txt`.

**Why that is not a weakening.** `sts:AssumeRole` remains tag-conditioned, so no
session can be created without the scope tag. `sts:SetSourceIdentity` on its own
grants no ability to assume anything — it only permits stamping an identity onto a
session that some other, still-conditioned, grant authorised. The trust policy now has
exactly two statements, and `tests/test_scope_tag_abac.py` plus
`tests/test_source_identity_join.py` assert both halves: that no statement grants
`sts:AssumeRole` without the condition, and that the unconditioned statement grants
`sts:SetSourceIdentity` and nothing else.

## Finding: the documentation contradicts itself on the length limit

| Source | Stated limit |
|---|---|
| [STS API Reference, `AssumeRole`](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html) | "Minimum length of 2. Maximum length of **64**", pattern `[\w+=,.@-]*` |
| [IAM User Guide, Monitor and control actions taken with assumed roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_monitor.html) | "must be between 2 and **256** characters long" |

We enforce **64**, the stricter of the two, because the API is what rejects the call —
a 200-character source identity accepted by the User Guide's rule would fail at the
STS boundary, and failing there means no credential and a broken request. A Cognito
`sub` is a 36-character UUID and `gw-` + session id is 39, so both fit with room to
spare; the constants exist to fail loudly if the input shape ever changes.

The reserved `aws:` prefix is rejected explicitly rather than left to STS, so the
error surfaces in our own code with our own message.

## The other end of the chain: DynamoDB data events

`SourceIdentity` lands in the `AssumeRole` event, which is a **management** event —
free, and readable in CloudTrail Event history for 90 days with no trail. But the
table access performed *with* that credential is a **data-plane** event, and CloudTrail
records data events only if a trail explicitly selects them
([Logging DynamoDB operations by using AWS CloudTrail](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/logging-using-cloudtrail.html)).
Without a trail the chain stopped at the moment the credential was minted.

`cdk/audit_trail.py` adds that trail. What the data events contribute, verified
against live output rather than assumed:

- `userIdentity.sessionContext.sourceIdentity` — the `sub`, **on the data access
  itself**, not merely on the `AssumeRole`. This matters for retention: the data event
  survives in S3 long after the 90-day Event history window closes, so the "who"
  outlives the event that established it.
- The assumed-role ARN ends in the `RoleSessionName`, giving the join key.
- `requestParameters.key` — **the item keys are recorded.** Stated plainly because it
  is a privacy consequence, not a feature: the partition key here is the TEAM NAME and
  the sort key is the document id, so both are written into the trail. The document
  body is not.
- `readOnly` distinguishes a read from a write, and `resources` names the table ARN.

### Cost, and a correction to an earlier claim in this repo

Data events are **$0.10 per 100,000 events delivered to S3**
([CloudTrail pricing](https://aws.amazon.com/cloudtrail/pricing/)), plus S3 storage.
One tool call produces three data events, so the per-call cost is three millionths of
a dollar; roughly 33,000 calls buy ten cents. Charging is per event delivered, so an
idle trail costs nothing beyond storage already written — measured: 38 minutes with no
activity produced zero new objects.

An earlier version of `cdk/audit_trail.py` justified excluding management events as
avoiding a duplicate charge. **That was wrong** and has been corrected in the module:
the first copy of management events delivered to S3 is free. They are excluded because
they are account-wide and constant — every deploy, every console action, every
service-linked role — and would bury the handful of relevant events under noise while
growing the bucket whether or not anyone is testing.

### Two implementation notes worth keeping

`cloudtrail.DataResourceType` has **no `DYNAMO_DB_TABLE` member** in the pinned
`aws-cdk-lib` (2.263.0); calling it raises `AttributeError`. The member exists in later
releases. The selector is therefore declared through the L1 escape hatch rather than
bumping the whole dependency for one property — `AWS::DynamoDB::Table` is a documented
*basic* event-selector resource type, so no advanced selector is needed
([DataResource](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-cloudtrail-trail-dataresource.md)).

The L2 `Trail` refuses `management_events=ReadWriteType.NONE` unless a selector was
added through the L2 API, and it decides that during synth-time validation — before
the L1 override exists. So the prop is left at its default while the override, which
carries `IncludeManagementEvents: false`, determines what actually ships.
`tests/test_audit_trail_scope.py` asserts the shipped shape on the synthesized
template rather than trusting that reasoning.

## Controls NOT weakened

- `sts:AssumeRole` is still conditioned on the scope request tag.
- The per-request inline session policy, the 900-second expiry, and the interceptor's
  JWT validation are untouched.
- Tool execution roles still hold no `sts:AssumeRole` and no DynamoDB permission.
- No credential value is written to any log, span attribute, or test fixture; the
  fail-closed test asserts the subject itself is not disclosed in the error path.
- Gateway application-log delivery remains disabled.

## Verified

Live, after the change. One invocation, three `AssumeRole` events and — from the
trail — three DynamoDB data events, all carrying the same two values:

```
AssumeRole (management event, Event history)
  roleSessionName : gw-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
  sourceIdentity  : 11111111-2222-3333-4444-555555555555
  tags            : [{scope: payments-core}]          <- unchanged

GetItem    readOnly=true   key={scope: payments-core, doc_id: PAY-001}
Query      readOnly=true   key={scope: payments-core}
UpdateItem readOnly=false  key={scope: payments-core, doc_id: PAY-001}
  userIdentity.arn ends in  DocumentsAccessRole/gw-aaaaaaaa-…-eeeeeeeeeeee
                            DocumentsWriteRole/gw-aaaaaaaa-…-eeeeeeeeeeee
  sessionContext.sourceIdentity : 11111111-2222-3333-4444-555555555555
  resources : arn:aws:dynamodb:us-east-1:123456789012:table/scoped-credentials-documents
```

- The `sourceIdentity` equals the demo user's `sub`, confirmed independently through
  `cognito-idp admin-get-user` rather than inferred from our own log.
- The `roleSessionName` suffix equals the `Mcp-Session-Id` the interceptor logged for
  that same request.
- The read path used the read role and the write path the write role, so the split is
  visible in the record.
- Suite: 688 passing.

Transcripts (local, unredacted): `evidence/wi1/assumerole-after-raw.txt`,
`evidence/wi1/dynamodb-data-events.json`.

## What was NOT verified

- **Role chaining.** `SourceIdentity` is documented to persist across chained
  sessions; this system never chains, so the persistence property is taken from the
  documentation and not exercised here.
- **A second human.** The join was proven with one Cognito user. Two concurrent users
  producing two distinct `sourceIdentity` values in interleaved requests has not been
  run; `tests/test_source_identity_join.py` covers the derivation in isolation, which
  is not the same claim.
- **Behaviour when the Gateway omits `Mcp-Session-Id`.** The code raises rather than
  inventing a session name, and the unit test covers it, but no live request has been
  observed without that header.
- **The digest cadence when idle.** 32 digest files were written once at trail
  creation — one per region, because the CDK trail is multi-region — and none in the
  following 38 minutes. Whether a digest appears at each hour boundary with zero
  activity was not observed. Each is ~380 bytes either way.
- **Nothing here was checked against a real auditor's requirements.** The claim is
  that the records are joinable, not that they satisfy any particular framework.

## Open issue: the three retention clocks do not agree

The chain is only as long as its shortest link, and today the links expire at
different times:

| Link | Where | Retention |
|---|---|---|
| `AssumeRole` — who + join key | CloudTrail Event history | 90 days, fixed, not extendable |
| DynamoDB data event — what was touched | trail's S3 bucket | indefinite (no lifecycle rule) |
| interceptor record — tool, arguments, scope (work item 2) | `/aws/lambda/scoped-credentials-session-guard` | **30 days** |

After 30 days an auditor can still establish *who* touched *what*, because work item 1
put both values inside the data event. What is lost is *which tool was requested, with
which arguments, and which scope was granted* — the work item 2 record. Raising the
interceptor log group's retention is a policy decision with a storage cost and has
**not** been taken; it should be settled before work item 2 starts writing records
worth keeping, not after.
