// SCOPE OF THIS POLICY — read this before adding a `when` clause.
//
// This is a per-TOOL allow-list, NOT the tenant boundary. `principal` is
// deliberately unconstrained: every authenticated caller the policy engine sees
// may invoke this tool. Deleting this file would change which *tools* are
// callable — never which DynamoDB partition a caller can reach.
//
// The tenant boundary is enforced at request time, elsewhere:
//   interceptor/handler.py            derives served_scope from the validated
//                                     JWT and fails closed when none can be.
//   interceptor/scoped_credentials.py pins dynamodb:LeadingKeys to that scope
//                                     in the vended session's inline policy.
//   cdk/documents_roles.py            requires LeadingKeys to match
//                                     ${aws:PrincipalTag/scope} (session-tag
//                                     ABAC) in the scoped roles themselves.
//
// For this tool specifically: this is the write-class tool, and the in-scope
// write amplification it permits is a known residual — a `when` clause here
// would not close it, because the writes stay inside the caller's own partition.
//
// Constraining the principal here would express that same boundary a second
// time, with no added enforcement and two places to keep in sync. The Cedar
// policies are not the tenant boundary; the scoped credential is.
//
// The action string below is the gateway's composite tool name
// ({targetName}___{toolName}) and MUST match a registered gateway target. That
// is checked ONCE, at synth, by tests/test_cedar_policy_actions.py. The Policy
// Engine does NOT check it at deploy time: FAIL_ON_ANY_FINDINGS would also
// reject the deliberately unconstrained principal above, so no deploy carrying
// these policies could succeed — see the comment on validation_mode in
// cdk/policy_resources.py.

permit(
  principal,
  action == AgentCore::Action::"Reply___reply",
  resource == AgentCore::Gateway::"<gateway-arn>"
);
