# The audit map

_Per question: where the answer is stored, which field holds it, and which key joins
that record to its neighbours. Identifiers are placeholders per `notes/README.md`._

This is a map, not an archive. Almost nothing here is written by us: the point of the
three work items was to make records AWS already writes **joinable**, by putting the
missing identifiers inside them. The one record we author is row 3, because no AWS
service observes what it contains.

Read it with `notes/audit-queries.sql`, which is the runnable form of rows 1–5.

## The six questions

| # | Question | Source | Field | Join key |
|---|---|---|---|---|
| 1 | Which human was signed in? | CloudTrail data event (trail's S3 bucket) | `userIdentity.sessionContext.sourceIdentity` = the Cognito `sub` | `role_session_name`, the last ARN segment |
| 2 | Which tool did the agent request? | Interceptor log (`/aws/lambda/scoped-credentials-session-guard`) | `tool_requested` (raw, e.g. `Reply___reply`) and `tool_classified` | `role_session_name` |
| 3 | With which original arguments? | Same record | `arguments` (model-supplied, `context` excluded) | `role_session_name` |
| 4 | Which scope did the interceptor grant? | Same record | `granted_scope` | `role_session_name` |
| 5 | Which API did the credentials call, on which table? | CloudTrail data event | `eventName`, `requestParameters.tableName`, `requestParameters.key`, `readOnly`, `resources[].ARN` | `role_session_name` |
| 6 | Which reasoning led to the request? | Bedrock model invocation log (`/aws/bedrock/scoped-credentials-modelinvocations`) | `input` / `output` (the tool the model chose is in `output…toolUse.name`) | **none — see the gap below** |

Rows 1–5 join on one key: `gw-<Mcp-Session-Id>`, present as the STS `RoleSessionName`
in every CloudTrail record and recorded verbatim as `role_session_name` in the
interceptor's. It ties the three tool calls of one user turn together and ties them to
the person.

## Two supporting records

| Question | Source | Field |
|---|---|---|
| Which application invoked the agent? | CloudTrail data event, `InvokeAgentRuntime` | `userIdentity.arn` (the IAM principal), `responseElements.runtimeSessionId` |
| Was a credential ever minted for this request? | CloudTrail management event, `AssumeRole` (Event history, 90 days, free) | `sourceIdentity`, `roleSessionName`, `requestParameters.tags` |

The `AssumeRole` event is redundant with row 1 by design: work item 1 put the same two
values into the data event, so the "who" survives after Event history's 90-day window
closes. Note that `InvokeAgentRuntime` identifies the calling **application**, not the
end user — the Cognito identity travels inside the JWT in the request payload, which
CloudTrail does not record, and should not.

## Credentials appear nowhere

Not by scrubbing, which can rot, but structurally:

- The interceptor's record is emitted **before** the vend, so no credential exists in
  scope when it is built. The `context` key — the RETIRED request-body credential key,
  which no tool declares or reads — is excluded, and every builder parameter
  is keyword-only so no unnamed value can be passed in.
- CloudTrail redacts by itself: `InvokeAgentRuntime` shows
  `response: HIDDEN_DUE_TO_SECURITY_REASONS`, and `AssumeRole` never returns the keys.
- The credentials travel as propagated request headers and the request body is forwarded
  unchanged, so the body carries only the model-supplied arguments. Gateway
  application-log delivery stays disabled as well, so no enriched body or enriched
  arguments can reach a log through it.

## The gap: row 6 has no join key

Verified, not assumed. The Bedrock model invocation record contains
`accountId, identity, inferenceRegion, input, modelId, operation, output, region,
requestId, schemaType, schemaVersion, timestamp` — no `sub`, no `Mcp-Session-Id`, and
no trace field. Its `identity.arn` names the runtime's own session
(`.../BedrockAgentCore-<uuid>`), which was measured NOT to equal the
`runtimeSessionId` the caller supplies.

Four candidate trusted join points were checked and none reaches it:

1. **The interceptor's input payload.** Documented to carry only `path`,
   `httpMethod`, `headers`, `body` — no caller identity. It cannot see the runtime
   session even in principle.
2. **CloudTrail.** `InvokeAgentRuntime` is now captured (row above) but carries the
   application identity and the runtime session, neither of which appears in the
   reasoning record.
3. **The forwarded trace id.** The Gateway does forward `X-Amzn-Trace-Id`, and it is
   recorded as `trace_id`. But each tool call arrives under a FRESH trace — three
   different roots for three calls in one turn — so it does not continue the runtime's
   trace, and the reasoning record has no trace field to match it against anyway.
4. **AgentCore spans** (`aws/spans`). One outer `AgentCore.Runtime.Invoke` span, on a
   third trace id again, with no child spans for the LLM calls or tool invocations.
   Those require instrumenting the agent framework, which puts the evidence back in
   the hands of the component this architecture assumes may be hostile.

**What this means in practice.** With one user at a time, row 6 joins by time and by
tool name: the model's chosen tool (`output…toolUse.name`, e.g.
`ReadDocument___read_document`) equals `tool_requested` in the interceptor's record
seconds later, and the ordering is unambiguous. With two users acting concurrently it
is not, and no stored record resolves it — you could tell who read a document, but not
which of the two reasonings led to it.

The honest conclusion: the chain **person → request → granted scope → table access**
is complete and key-joined. The step **reasoning → request** is correlated, not
proven. That is a platform limitation today, not a design choice, and it is stated
here rather than papered over with a timestamp match presented as a join.

## What was NOT verified

- **The Athena path to rows 1, 5 and the supporting `InvokeAgentRuntime` row.** The
  table creates but every SELECT fails inside the CloudTrail SerDe; see the status
  note at the top of `notes/audit-queries.sql`. The FACTS were verified by reading the
  trail's objects directly — it is the SQL route that is unproven, not the evidence.
- **Rows 2–4 were verified by a live Logs Insights query** returning `subject`,
  `granted_scope`, `tool_requested`, `tool_classified`, `gateway_session_id`,
  `role_session_name` and `trace_id` from a `doc_id` alone, across two user turns
  correctly separated by session. That query ran; the Athena ones did not.
- **Concurrency.** Every measurement here used one user. The two-user case is exactly
  where row 6's weakness bites, and it has not been exercised.
- **Nothing here was checked against a real auditor's requirements.** The claim is
  that the records are joinable, not that they satisfy any particular framework.
