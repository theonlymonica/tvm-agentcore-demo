# WI1 — SourceIdentity and a join key in the STS session

## The claim

Every DynamoDB action performed with vended credentials is traceable in CloudTrail
to the human who signed in and to the Gateway request that caused it.

Before this change it was traceable to neither. `RoleSessionName` was
`f"scope-{served_scope}"`, so every action by every user in a scope appeared under
the identical session name `scope-payments-core`, and no `SourceIdentity` was set at
all. The containment controls were intact; the attribution did not exist.

## What changed

| File | Change |
|---|---|
| `interceptor/jwt_claims.py` | New `verified_identity_from_authorization` returns `VerifiedIdentity(served_scope, subject)` from ONE verification. `served_scope_from_authorization` is now a wrapper over it. New `_subject_from_claims` reads `sub`. |
| `interceptor/scoped_credentials.py` | `vend_scoped_credentials` takes required keyword-only `subject` and `gateway_identifier`, and passes `SourceIdentity` plus a derived `RoleSessionName`. New `build_role_session_name`, `_source_identity_or_raise`, `SourceIdentityError`, `RoleSessionNameError`. |
| `interceptor/handler.py` | Resolves the identity, fails closed when it is unavailable, and passes both values into `_vend_for_tool` (signature changed accordingly). |
| `cdk/lambda_iam.py` | `sts:SetSourceIdentity` added to the single conditioned trust statement of both scoped roles, and to the interceptor's identity-side statement. |

## Decisions and their grounds

**`sub`, not username or email.** AWS documents `sub` as "A unique identifier
(UUID), or subject, for the authenticated user. The username might not be unique in
your user pool. The sub claim is the best way to identify a given user"
([access token claims](https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-the-access-token.html)).
A username or email can be reassigned to a different person; an audit anchor cannot.

**`SourceIdentity` for the person, `RoleSessionName` for the request.** These are
not redundant. `SourceIdentity` is immutable once set, is present in the request
context of every action the session takes, and persists across role chaining
([aws:SourceIdentity](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html)).
`RoleSessionName` is caller-settable per assume — which is exactly why it is the
right field for the request-grained value — and it appears in the assumed-role ARN,
so it lands in the `sessionContext` of every DynamoDB data event.

**The Gateway identifier is `Mcp-Session-Id`.** The
[documented REQUEST interceptor payload](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-types.html)
carries `interceptorInputVersion`, `mcp.rawGatewayRequest.body`, and
`mcp.gatewayRequest.{path, httpMethod, headers, body}`. **There is no request id in
it.** The only identifier available is the `Mcp-Session-Id` header, which the code
already extracted. An AWS blog post shows an `mcp.requestContext` field whose
contents the devguide does not specify — see *Not verified* below.

Consequence, stated plainly rather than glossed: `Mcp-Session-Id` is
**session-grained, not request-grained**. Every tool call inside one MCP session
shares it (`agent/agent_core.py`: "Within this block, all tool calls share one
Mcp-Session-Id"). So the join key identifies the *conversation*, not the individual
tool call. Distinguishing two calls within one conversation needs a further
discriminator, and the only per-call value available is the JSON-RPC `id` in the
**body** — which this work item forbids as a source. WI2's interceptor record is
where per-call granularity belongs, since it can carry both the session identifier
and the call's own arguments.

**Derivation rule for `RoleSessionName`.** Constraints are length 2–64 and pattern
`[\w+=,.@-]*`
([API_AssumeRole](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html)).
The derivation is deterministic and total:

- fits and matches → `gw-<identifier>` (verbatim, so an operator can eyeball it)
- otherwise → `gwh-<sha256(identifier)[:32]>`

Two different prefixes so the forms are never ambiguous. SHA-256 is used as
deterministic shortening, not as a secret — the identifier is not confidential.

**`re.ASCII` on the pattern.** Python's `\w` matches Unicode word characters by
default. Without `re.ASCII` a subject containing a Cyrillic letter would pass the
local check and then be rejected by STS, converting a clean fail-closed refusal into
an opaque runtime `AccessDenied`. IAM's "upper- and lower-case alphanumeric
characters" means ASCII.

**Fail closed on a missing `sub`.** `verified_identity_from_authorization` returns
`None` — the same value an unresolvable scope produces — so the handler's existing
short-circuit applies and no read occurs. There is no substitution from the request
body, no username fallback, and no synthesized placeholder. A placeholder would be
worse than no attribution: the trail would look complete while naming nobody.

**Fail closed on a missing Gateway identifier.** Also refused, rather than falling
back to the old scope-derived name. A session that cannot be joined to the request
that caused it defeats the purpose of naming it. Risk accepted knowingly: if the
Gateway ever omits the header on a `tools/call`, scoped tools stop working rather
than silently losing their audit trail. The documented payload example includes the
header, and `passRequestHeaders=true` is set.

**Required keyword-only parameters, no defaults.** An unattributed or unjoinable
vend must be impossible to *express*, not merely discouraged. This is why 7 test
files needed updating — that churn is the feature working.

## Documentation contradiction (recorded, not resolved in our favour)

Two AWS pages disagree on the maximum length of `SourceIdentity`:

- [API_AssumeRole](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html):
  "Length Constraints: Minimum length of 2. **Maximum length of 64**."
- [Monitor and control actions taken with assumed roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_monitor.html):
  "The value of source identity must be between 2 and **256** characters long."

This module enforces the **stricter** bound (64), reasoning that the API reference
describes the validator the request actually meets, and that a value accepted under
the stricter rule is accepted under both. The choice is immaterial for the value
passed — a Cognito `sub` is a 36-character UUID — but it is enforced explicitly so a
future non-UUID subject cannot silently depend on the looser reading.

## Controls NOT weakened

Verified by the suite, not by inspection:

- The `scope` session tag is still passed, exactly one, unchanged.
- The inline session policy is still passed, with all three condition operators.
- `DurationSeconds` is still 900; the policy TTL is still 60 seconds.
- All three trust actions share ONE statement, still conditioned on
  `aws:RequestTag/scope` — splitting `sts:AssumeRole` into its own unconditioned
  statement to make room would have re-opened the untagged-assume hole.
- The identity-side grant stays pinned to the two Documents role ARNs, not `"*"`.
- `TransitiveTagKeys` is still unset.
- The returned credentials dict is still exactly three keys — no audit field was
  smuggled into the wire contract.
- The handler still mutates nothing in the inbound event.

## Verified

- 637 tests pass (594 pre-change baseline + 43 new). Output:
  `evidence/wi1/pytest-after-change.txt`.
- The synthesized template carries the change on both roles and on the identity
  statement: `evidence/wi1/synth-iam-after.json`. Trust `Action` on both
  `DocumentsAccessRole` and `DocumentsWriteRole` is
  `['sts:AssumeRole', 'sts:TagSession', 'sts:SetSourceIdentity']` with
  `Condition {'StringLike': {'aws:RequestTag/scope': '*'}}`; the
  `TagScopedDocumentsSessions` identity statement is
  `['sts:TagSession', 'sts:SetSourceIdentity']`.
- Account for all work: `761018895255`, profile `monica-doctor`, region
  `us-east-1` (`aws sts get-caller-identity`).

## NOT verified

- **The live CloudTrail chain.** Nothing here has been deployed yet. The
  `sourceIdentity` / `roleSessionName` fields in a real `AssumeRole` event, and in
  the `sessionContext` of a real DynamoDB data event, are still to be produced.
- **`mcp.requestContext`.** Its existence and contents are **not documented** in the
  devguide. Whether it carries a per-request id — which would give a
  request-grained join key instead of a session-grained one — must be settled by
  logging the event's key names in the account.
- **Whether `Mcp-Session-Id` is present on every `tools/call`** in this deployment.
  The documented payload example includes it; the fail-closed choice above depends
  on it.
- **CloudTrail data events for the document table.** Not enabled, and no trail
  exists in the account at all (`describe-trails` → `[]`,
  `list-event-data-stores` → `[]`). Whether item keys appear in the event, and the
  cost, are still to be established.
- Whether a Cognito `sub` in this pool is always a UUID. Cognito documents it as a
  UUID and the constraint is asserted against the STS pattern, but only test-user
  tokens have been examined so far — none yet, in fact, since nothing is deployed.
