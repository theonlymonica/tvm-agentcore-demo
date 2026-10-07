-- The auditor's queries.
--
-- The point of these is that they RUN. A map saying "the answer is in CloudTrail"
-- is a promise; a query that starts from a document id and comes back with a person
-- is the thing that keeps it. Two engines are needed because the records live in two
-- places, and neither can see the other's:
--
--   * CloudWatch Logs Insights  -> the interceptor's audit record (work item 2):
--                                  which tool, which original arguments, which scope.
--   * Athena over the trail's S3 -> the CloudTrail data events (work item 1):
--                                  who (sourceIdentity), what was touched, and the
--                                  AgentCore invocation.
--
-- Identifiers below are PLACEHOLDERS. Substitute before running:
--   <TRAIL_BUCKET>   the audit trail's bucket name (stack output)
--   <ACCOUNT_ID>     the AWS account id
--   <REGION>         the region the stack is deployed in, e.g. us-east-1
--   <ATHENA_DB>      an Athena database, e.g. default
--
-- Cost note: Athena bills per byte SCANNED. Every query below constrains the
-- partition (`timestamp`) — without that, a query reads the whole trail prefix and
-- the bill grows with retention rather than with the question being asked.


-- ---------------------------------------------------------------------------
-- 1. One-time: the table over the trail's objects.
-- ---------------------------------------------------------------------------
-- STATUS: this DDL is NOT verified. Written from the documented schema, it created
-- successfully, but every SELECT against it failed inside the CloudTrail SerDe:
--
--   HIVE_BAD_DATA: Error Parsing a column in the table:
--       Current entry must be closed before a null can be written
--   HIVE_BAD_DATA: Error Parsing a column in the table: null
--
-- Three schema variants were tried (with and without `sourceidentity` inside
-- `sessioncontext`, with and without `onbehalfof` / `ec2roledelivery` /
-- `webidfederationdata`). The SerDe is strict about the nested struct shape and the
-- records in this trail carry fields the published examples do not, so the exact
-- struct is a guessing game — and a guessed schema that parses today breaks the day
-- CloudTrail adds a field.
--
-- Do NOT hand-tune this. Two paths that avoid the guess entirely:
--
--   a) Let CloudTrail generate it. The console's Event history page has a
--      "Create Athena table" action which emits a CREATE TABLE statement matching
--      the records in YOUR account. The documentation calls this "the easiest way"
--      for exactly this reason:
--      https://aws.amazon.com/blogs/mt/use-amazon-athena-and-aws-cloudtrail-to-estimate-billing-for-aws-config-rule-evaluations/
--   b) Use CloudTrail Lake (an event data store), which is queryable with SQL with no
--      schema to get wrong. It is a separate paid resource, so price it first:
--      https://aws.amazon.com/cloudtrail/pricing/
--
-- The facts these queries are meant to return WERE verified, by reading the trail's
-- gzipped objects directly. What is unverified is this SQL path to them, not the
-- evidence.
--
-- Partition projection rather than ALTER TABLE ADD PARTITION: CloudTrail writes a
-- new prefix every day, so a manually partitioned table silently stops covering
-- "today" the moment nobody remembers to add it — and a query that returns no rows
-- because a partition is missing looks exactly like a query that returns no rows
-- because nothing happened. That distinction is the whole value of an audit trail.

CREATE EXTERNAL TABLE IF NOT EXISTS <ATHENA_DB>.scoped_credentials_audit_trail (
    eventversion STRING,
    useridentity STRUCT<
        type: STRING,
        principalid: STRING,
        arn: STRING,
        accountid: STRING,
        invokedby: STRING,
        accesskeyid: STRING,
        username: STRING,
        sessioncontext: STRUCT<
            attributes: STRUCT<
                mfaauthenticated: STRING,
                creationdate: STRING>,
            sessionissuer: STRUCT<
                type: STRING,
                principalid: STRING,
                arn: STRING,
                accountid: STRING,
                username: STRING>,
            sourceidentity: STRING>>,
    eventtime STRING,
    eventsource STRING,
    eventname STRING,
    awsregion STRING,
    sourceipaddress STRING,
    useragent STRING,
    errorcode STRING,
    errormessage STRING,
    requestparameters STRING,
    responseelements STRING,
    additionaleventdata STRING,
    requestid STRING,
    eventid STRING,
    resources ARRAY<STRUCT<
        arn: STRING,
        accountid: STRING,
        type: STRING>>,
    eventtype STRING,
    apiversion STRING,
    readonly STRING,
    recipientaccountid STRING,
    serviceeventdetails STRING,
    sharedeventid STRING,
    vpcendpointid STRING,
    eventcategory STRING
)
PARTITIONED BY (`timestamp` STRING)
ROW FORMAT SERDE 'com.amazon.emr.hive.serde.CloudTrailSerde'
STORED AS INPUTFORMAT 'com.amazon.emr.cloudtrail.CloudTrailInputFormat'
OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 's3://<TRAIL_BUCKET>/AWSLogs/<ACCOUNT_ID>/CloudTrail/<REGION>/'
TBLPROPERTIES (
    'projection.enabled' = 'true',
    'projection.timestamp.type' = 'date',
    'projection.timestamp.format' = 'yyyy/MM/dd',
    'projection.timestamp.range' = '2026/01/01,NOW',
    'projection.timestamp.interval' = '1',
    'projection.timestamp.interval.unit' = 'DAYS',
    'storage.location.template' =
        's3://<TRAIL_BUCKET>/AWSLogs/<ACCOUNT_ID>/CloudTrail/<REGION>/${timestamp}'
);


-- ---------------------------------------------------------------------------
-- 2. "Who touched this document, and what did they do to it?"
-- ---------------------------------------------------------------------------
-- The auditor's entry point. Starts from a document id and returns the person.
--
-- `sourceidentity` is the Cognito `sub` — the human. It is read from the DATA event,
-- not from the AssumeRole event, and that matters for retention: the data event
-- outlives the 90-day Event history window, so the "who" survives the event that
-- established it.
--
-- The role session name is recovered from the assumed-role ARN's last segment; it is
-- the join key back to the interceptor's record (query 4).

SELECT
    eventtime,
    eventname,
    readonly,
    useridentity.sessioncontext.sourceidentity      AS human_subject,
    element_at(split(useridentity.arn, '/'), -1)    AS role_session_name,
    useridentity.sessioncontext.sessionissuer.username AS assumed_role,
    json_extract_scalar(requestparameters, '$.key.doc_id') AS doc_id,
    json_extract_scalar(requestparameters, '$.key.scope')  AS partition_scope
FROM <ATHENA_DB>.scoped_credentials_audit_trail
WHERE eventsource = 'dynamodb.amazonaws.com'
  AND `timestamp` >= '2026/09/01'
  AND json_extract_scalar(requestparameters, '$.key.doc_id') = 'PAY-001'
ORDER BY eventtime DESC;


-- ---------------------------------------------------------------------------
-- 3. "Which application invoked the agent, and under which runtime session?"
-- ---------------------------------------------------------------------------
-- The AgentCore half. `useridentity.arn` here is the IAM principal that called
-- InvokeAgentRuntime — the APPLICATION, not the end user. The end user travels
-- inside the JWT in the request payload, which CloudTrail does not record (correctly:
-- recording it would put a bearer token in a log). Do not read this as the human.

SELECT
    eventtime,
    eventname,
    useridentity.type                               AS caller_type,
    useridentity.arn                                AS calling_principal,
    json_extract_scalar(responseelements, '$.runtimeSessionId') AS runtime_session_id,
    sourceipaddress
FROM <ATHENA_DB>.scoped_credentials_audit_trail
WHERE eventsource = 'bedrock-agentcore.amazonaws.com'
  AND eventcategory = 'Data'
  AND `timestamp` >= '2026/09/01'
ORDER BY eventtime DESC;


-- ---------------------------------------------------------------------------
-- 4. CloudWatch Logs Insights — "what was requested, and what was granted?"
-- ---------------------------------------------------------------------------
-- Not SQL. Run against the interceptor's log group:
--   /aws/lambda/scoped-credentials-session-guard
--
-- Starts from the same doc_id as query 2 and returns the three facts CloudTrail
-- cannot see, because they are decided before any AWS API is called. Join to query 2
-- on `role_session_name`.
--
--   fields @timestamp, subject, granted_scope, tool_requested, tool_classified,
--          gateway_session_id, role_session_name, trace_id,
--          arguments.doc_id as doc_id
--   | filter @message like /TOOL_CALL_AUDIT/
--   | filter arguments.doc_id = "PAY-001"
--   | sort @timestamp desc
--   | limit 20
--
-- To go the other way — from a person to everything they did — swap the second
-- filter for:
--
--   | filter subject = "<COGNITO_SUB>"
--
-- A refused call appears here with no matching row in query 2: the record is written
-- BEFORE the credential is minted, so an attempt that failed to vend is still
-- audited. That asymmetry is intentional and is the reason the two queries are read
-- together rather than joined blindly.
