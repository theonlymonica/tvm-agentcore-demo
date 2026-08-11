"""
AgentCore Gateway resources.

This module provisions:
- AgentCore Gateway with MCP protocol and CUSTOM_JWT inbound authorization
  (the served_scope is derived from a signature-validated Cognito JWT claim)
- pre-stage-scope REQUEST interceptor wiring (pass_request_headers=True)
- post-stage RESPONSE interceptor wiring — strips credential-shaped material
  from tool replies; wired only when a response_interceptor_fn is supplied
- PolicyEngine attachment via AwsCustomResource (UpdateGateway API)

Inbound auth:
  The gateway inbound authorizer is CUSTOM_JWT, wired to the MANAGED Cognito
  user pool + app client created in cdk/auth_resources.py. Validation is via
  ``allowedClients`` (NOT ``allowedAudience``): the Cognito access token carries
  ``aud=null`` and ``client_id`` matching the app client — confirmed live. The
  discovery URL and app client id arrive as an ``AuthResources`` argument: they
  are deploy-time CloudFormation tokens, so they cannot be imported as literal
  constants.

  The RESPONSE interceptor is a credential-shaped-material scrubber
  (response_interceptor/handler.py) that only strips credential fields a tool
  might echo back and never blocks — it is not an enforcement gate. Driven by
  the single response_interceptor_fn, it is declared at BOTH points —
  _build_interceptor_configs (creation) here and attach_policy_engine's
  UpdateGateway payload (gateway_wiring.py) — so they can never diverge and the
  UpdateGateway call can never silently drop it. Wiring is conditional: when
  response_interceptor_fn is None only the REQUEST interceptor is declared.

The LambdaInterceptor auto-grants the gateway role lambda:InvokeFunction
on the interceptor Lambda.

Policy engine attachment:
  The stable Gateway L2 does not accept policy_engine_configuration as a
  constructor parameter. Setting it via the L1 escape hatch causes a race
  condition: CloudFormation creates the Gateway (with PolicyEngineConfiguration)
  in parallel with the SupportGatewayServiceRoleDefaultPolicy (the inline IAM
  policy containing GetPolicyEngine permission). The Gateway creation calls
  GetPolicyEngine before the IAM permission exists → "Access denied".

  The fix: create the Gateway WITHOUT policy_engine_configuration, then use an
  AwsCustomResource to call the UpdateGateway API AFTER both the Gateway and its
  IAM role policy are fully created. This guarantees the IAM permission exists
  before the service attempts GetPolicyEngine.

Citation:
  - Gateway L2 (stable):
    https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_bedrockagentcore/README.html
  - CDK GatewayAuthorizer.using_custom_jwt (Inbound authorization — discovery_url,
    allowed_audience, allowed_clients, allowed_scopes, custom_claims):
    https://constructs.dev/packages/aws-cdk-lib/v/2.261.0?submodule=aws_bedrockagentcore&lang=python
  - CDK LambdaInterceptor.for_request / for_response + interceptor_configurations
    prop (one interceptor of each kind; auto-grants lambda:InvokeFunction):
    https://constructs.dev/packages/aws-cdk-lib/v/2.261.0/api/LambdaInterceptor?lang=python&submodule=aws_bedrockagentcore
  - interceptorConfigurations entry shape (interceptor.lambda.arn +
    interceptionPoints ["REQUEST"|"RESPONSE"]):
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-interceptors-configuration.html
  - CUSTOM_JWT customJWTAuthorizer service shape (discoveryUrl, allowedAudience,
    allowedClients, allowedScopes, customClaims):
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-create-api.html
  - UpdateGateway with Policy Engine (service docs):
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/update-gateway-with-policy.html
  - AwsCustomResource (CDK custom_resources):
    https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.custom_resources/AwsCustomResource.html
  - Gateway Tool Naming (service docs — triple underscore ___):
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html
"""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_bedrockagentcore as agentcore
import aws_cdk.aws_bedrock_agentcore_alpha as agentcore_alpha
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as lambda_
from constructs import Construct

# The managed Cognito identity references. The pool/client ids are deploy-time
# tokens, so they are threaded in as an argument instead of being imported as
# literal constants.
from auth_resources import AuthResources
# Post-creation AwsCustomResource wiring, extracted to keep this file within
# the code-modularity line limit.
from gateway_wiring import attach_policy_engine, enable_gateway_tracing


def create_gateway(
    scope: Construct,
    interceptor_fn: lambda_.Function,
    policy_engine: agentcore_alpha.PolicyEngine,
    auth: AuthResources,
    response_interceptor_fn: lambda_.Function | None = None,
    tool_fns: dict[str, lambda_.Function] | None = None,
) -> agentcore.Gateway:
    """Create the AgentCore Gateway with interceptor and policy engine.

    The policy engine is attached via an AwsCustomResource that calls the
    UpdateGateway API after the gateway and its IAM role policy are both
    fully created, preventing the "Access denied while calling
    GetPolicyEngine" deployment race condition.

    Args:
        scope: The CDK Stack or Construct to attach resources to.
        interceptor_fn: The pre-stage-scope REQUEST interceptor Lambda function.
        policy_engine: The PolicyEngine to attach to this gateway.
        auth: The managed Cognito identity references (``create_auth_resources``).
            Supplies the CUSTOM_JWT ``discovery_url`` and ``app_client_id``; both
            are deploy-time tokens resolved by CloudFormation.
        response_interceptor_fn: The post-stage RESPONSE interceptor Lambda that
            strips credential-shaped material from tool replies. When provided,
            it is wired as the single RESPONSE interceptor at BOTH declaration
            points — _build_interceptor_configs (creation) here AND
            attach_policy_engine's UpdateGateway payload — so the two
            declarations can never diverge and UpdateGateway can never silently
            drop it. When None, only the REQUEST interceptor is wired (the
            existing test_synth_config.py call site calls create_gateway without
            it and must keep working).
        tool_fns: Optional dict mapping tool names to their Lambda fns.

    Returns:
        The provisioned Gateway construct.
    """
    gateway = agentcore.Gateway(
        scope,
        "SupportGateway",
        gateway_name="support-tools-gateway",
        protocol_configuration=agentcore.McpProtocolConfiguration(
            instructions=(
                "Customer support tools"
            ),
            supported_versions=[
                agentcore.MCPProtocolVersion.MCP_2025_03_26,
                agentcore.MCPProtocolVersion.MCP_2025_06_18,
            ],
        ),
        authorizer_configuration=(
            # CUSTOM_JWT inbound auth: served_scope is derived from a
            # signature-validated Cognito JWT claim. Validate via
            # allowed_clients ONLY — the Cognito access token has aud=null, so
            # allowed_audience does not apply. At least one of allowed_clients /
            # allowed_audience is required by the construct. Both values are
            # tokens for the MANAGED pool/client, so the authorizer always
            # points at the pool this stack owns.
            agentcore.GatewayAuthorizer.using_custom_jwt(
                discovery_url=auth.discovery_url,
                allowed_clients=[auth.app_client_id],
            )
        ),
        interceptor_configurations=_build_interceptor_configs(
            interceptor_fn, response_interceptor_fn
        ),
    )

    # Add 2025-11-25 to supported MCP versions via L1 escape hatch.
    # The CDK enum only has MCP_2025_03_26 and MCP_2025_06_18, but the
    # gateway service supports 2025-11-25 (confirmed in docs) and the
    # mcp SDK 1.8+ (required by strands-agents) uses it by default.
    cfn_gw = gateway.node.default_child
    cfn_gw.add_property_override(
        "ProtocolConfiguration.Mcp.SupportedVersions",
        ["2025-03-26", "2025-06-18", "2025-11-25"],
    )
    cfn_gw.add_property_override(
        "ProtocolConfiguration.Mcp.SessionConfiguration",
        {"SessionTimeoutInSeconds": 3600},
    )

    # -- Omit ProtocolType from the template (immutable-field fix) --
    # The Gateway L2 synthesizes ``ProtocolType: MCP`` whenever an
    # McpProtocolConfiguration is supplied, but the LIVE gateway
    # (support-tools-gateway-EXAMPLE1) was created with protocolType=null
    # (GetGateway confirms null + a full protocolConfiguration.mcp). ProtocolType
    # is optional on both CreateGateway and UpdateGateway ("If you omit this
    # field, the gateway can have both MCP and HTTP targets") and the service
    # rejects any change to it on an existing gateway ("Protocol type cannot be
    # updated for an existing gateway").
    #
    # That template-vs-gateway divergence stays dormant until ANY other Gateway
    # property changes; adding the RESPONSE interceptor to
    # InterceptorConfigurations forces CloudFormation to issue an in-place
    # UpdateGateway that re-sends ProtocolType=MCP and is rejected as immutable.
    #
    # Deleting ProtocolType from the template makes the desired state (omitted →
    # null) match the deployed gateway, so the UpdateGateway carries no
    # protocolType change (null → null, a no-op) and the RESPONSE-interceptor
    # update goes through. ProtocolType is a mutable property ("Update requires:
    # No interruption" in the CFN resource schema), so removing it is an in-place
    # update, NOT a replacement — the gateway is never recreated/destroyed.
    #
    # Citations:
    #   - CreateGateway (protocolType optional; omit => MCP+HTTP targets):
    #     https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-create-api.html
    #   - UpdateGateway (protocolType Required: No; Valid Values: MCP):
    #     https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_UpdateGateway.html
    #   - CFN AWS::BedrockAgentCore::Gateway (ProtocolType Required: No,
    #     Update requires: No interruption):
    #     https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-bedrockagentcore-gateway.html
    #   - CfnResource.add_property_deletion_override (deletes a property):
    #     https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk/CfnResource.html
    cfn_gw.add_property_deletion_override("ProtocolType")

    # -- Grant the gateway role permissions to call the policy engine --
    # These IAM statements allow the gateway to call GetPolicyEngine and
    # AuthorizeAction at runtime.
    get_policy_stmt = iam.PolicyStatement(
        actions=["bedrock-agentcore:GetPolicyEngine"],
        resources=[policy_engine.policy_engine_arn],
    )
    auth_action_stmt = iam.PolicyStatement(
        actions=[
            "bedrock-agentcore:AuthorizeAction",
            "bedrock-agentcore:PartiallyAuthorizeActions",
        ],
        # Scope to the two concrete ARNs the AWS docs require instead of "*":
        # the policy engine ARN and the gateway ARN.
        resources=[
            policy_engine.policy_engine_arn,
            gateway.gateway_arn,
        ],
    )

    gateway.role.add_to_principal_policy(get_policy_stmt)
    gateway.role.add_to_principal_policy(auth_action_stmt)

    # -- Grant the workload-access-token mint (REQUIRED whenever a policy engine
    # -- is attached) --
    #
    # With a Policy Engine attached, the gateway propagates the caller's session
    # identity by minting a Workload Access Token before it invokes a target. On
    # the IAM inbound flow that mint calls ``GetWorkloadAccessToken``, so the
    # gateway execution role needs it IN ADDITION to the three policy
    # permissions above. Without it EVERY ``tools/call`` fails at the token-mint
    # step and the caller sees only a generic "An internal error occurred" —
    # Cedar still evaluates to ALLOW, the target Lambda is never invoked, and
    # nothing appears in the tool's CloudWatch logs, which makes this
    # exceptionally hard to diagnose from the outside. Observed live on
    # 2026-08-06:
    #
    #   Failed to get workload identity token - client error: User:
    #   ...assumed-role/...SupportGatewayServiceRole.../gateway-session-... is
    #   not authorized to perform: bedrock-agentcore:GetWorkloadAccessToken on
    #   resource: ...workload-identity-directory/default/workload-identity/
    #   support-tools-gateway-...
    #
    # The gateway's own workload identity is created for it at deploy time and
    # named after the gateway id, hence the ``<gatewayId>*`` suffix pattern. The
    # directory root is granted alongside the child, per the AWS example.
    #
    # Citation:
    #   - AgentCore Gateway and Policy IAM permissions — "IAM permissions for
    #     temporal policies" gives this exact statement (sid
    #     PolicySessionWorkloadIdentity) and states that without it tool
    #     invocations fail at the token-mint step:
    #     https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-permissions.html
    stack = cdk.Stack.of(gateway)
    directory_arn = (
        f"arn:aws:bedrock-agentcore:{stack.region}:{stack.account}"
        ":workload-identity-directory/default"
    )
    workload_identity_stmt = iam.PolicyStatement(
        sid="PolicySessionWorkloadIdentity",
        actions=["bedrock-agentcore:GetWorkloadAccessToken"],
        resources=[
            directory_arn,
            f"{directory_arn}/workload-identity/{gateway.gateway_id}*",
        ],
    )
    gateway.role.add_to_principal_policy(workload_identity_stmt)

    # -- Attach policy engine via UpdateGateway custom resource --
    # We do NOT set policy_engine_configuration on the Gateway at creation
    # time. CloudFormation creates the Gateway resource and the IAM
    # DefaultPolicy (containing GetPolicyEngine permission) in parallel.
    # If policy_engine_configuration is set at creation, the service calls
    # GetPolicyEngine before the IAM permission exists → "Access denied".
    #
    # Instead, this AwsCustomResource calls UpdateGateway to attach the
    # policy engine AFTER both the Gateway and IAM policy are fully
    # created, eliminating the race condition.
    #
    # Citation:
    #   https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/update-gateway-with-policy.html
    attach_policy_engine(
        scope, gateway, policy_engine, interceptor_fn, auth, response_interceptor_fn
    )

    # -- Lambda tool targets --
    # Three targets whose names form the first half of the Cedar action name:
    #   ReadDocument___read_document
    #   SearchDocuments___search_documents
    #   Reply___reply
    # Target names must match the Cedar action prefixes character-for-character.
    # Citation:
    #   https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html
    #   https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_bedrockagentcore/GatewayTarget.html
    if tool_fns:
        _create_tool_targets(scope, gateway, tool_fns)

    # Export the gateway ID for operational use
    cdk.CfnOutput(
        scope,
        "SupportGatewayGatewayId",
        value=gateway.gateway_id,
        description="Gateway identifier for operational use",
    )

    # Enable gateway TRACES delivery so Cedar decision spans reach aws/spans.
    enable_gateway_tracing(scope, gateway)

    return gateway


def _build_interceptor_configs(
    request_fn: lambda_.Function,
    response_fn: lambda_.Function | None = None,
) -> list:
    """Build the interceptor configurations list for the gateway.

    The REQUEST interceptor is always wired with ``pass_request_headers=True``.
    The RESPONSE interceptor (the credential-shaped-material scrubber) is wired
    ONLY when ``response_fn`` is provided — one of its two declaration points
    (the other is ``attach_policy_engine``'s UpdateGateway payload); both are
    driven by the single ``response_interceptor_fn`` so they cannot diverge.
    Keeping it conditional preserves the ``response_fn is None`` path used by the
    synth/config test. A gateway takes at most one interceptor of each kind.
    Citation:
    https://constructs.dev/packages/aws-cdk-lib/v/2.261.0/api/LambdaInterceptor?lang=python&submodule=aws_bedrockagentcore

    Args:
        request_fn: The pre-stage-scope REQUEST interceptor Lambda.
        response_fn: The RESPONSE interceptor Lambda, or None for REQUEST only.

    Returns:
        A REQUEST LambdaInterceptor, plus a RESPONSE one when ``response_fn`` is
        provided.
    """
    configs = [
        agentcore.LambdaInterceptor.for_request(
            request_fn, pass_request_headers=True
        )
    ]
    if response_fn is not None:
        configs.append(agentcore.LambdaInterceptor.for_response(response_fn))
    return configs


def _create_tool_targets(
    scope: Construct,
    gateway: agentcore.Gateway,
    tool_fns: dict[str, lambda_.Function],
) -> None:
    """Create the three Lambda tool targets on the gateway.

    Target names are load-bearing: they form the first half of the Cedar
    action name via the ``{targetName}___{toolName}`` pattern. The Cedar
    policies use action names ``ReadDocument___read_document``,
    ``SearchDocuments___search_documents``, and ``Reply___reply``, so the
    target names here MUST be exactly ``ReadDocument``, ``SearchDocuments``,
    and ``Reply``.

    No ``session_id`` property is injected into the tool schemas — the guard
    keys on the gateway ``Mcp-Session-Id`` header instead.

    Citation:
      - Gateway tool naming (triple underscore):
        https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html
      - GatewayTarget.for_lambda:
        https://constructs.dev/packages/aws-cdk-lib/v/2.261.0/api/GatewayTarget?lang=python&submodule=aws_bedrockagentcore
      - ToolSchema.from_inline / ToolDefinition / SchemaDefinition:
        https://constructs.dev/packages/aws-cdk-lib/v/2.261.0/api/ToolSchema?lang=python&submodule=aws_bedrockagentcore

    Args:
        scope: The CDK Stack or Construct to attach resources to.
        gateway: The Gateway to add targets to.
        tool_fns: Dict mapping tool names to their Lambda functions.
    """
    # -- ReadDocument target --
    agentcore.GatewayTarget.for_lambda(
        scope,
        "ReadDocumentTarget",
        gateway_target_name="ReadDocument",
        gateway=gateway,
        lambda_function=tool_fns["read_document"],
        tool_schema=agentcore.ToolSchema.from_inline([
            agentcore.ToolDefinition(
                name="read_document",
                description="Read a document by its id",
                input_schema=agentcore.SchemaDefinition(
                    type=agentcore.SchemaDefinitionType.OBJECT,
                    properties={
                        "doc_id": agentcore.SchemaDefinition(
                            type=agentcore.SchemaDefinitionType.STRING,
                            description="The unique identifier of the document to read",
                        ),
                        # No served_scope and no context property. The served
                        # scope and the vended tenant_credentials are injected
                        # by the REQUEST interceptor at arguments["context"]
                        # AFTER the model writes the call, so they travel only
                        # forward and are never advertised to the model in the
                        # published schema.
                    },
                    required=["doc_id"],
                ),
            ),
        ]),
    )

    # -- SearchDocuments target --
    agentcore.GatewayTarget.for_lambda(
        scope,
        "SearchDocumentsTarget",
        gateway_target_name="SearchDocuments",
        gateway=gateway,
        lambda_function=tool_fns["search_documents"],
        tool_schema=agentcore.ToolSchema.from_inline([
            agentcore.ToolDefinition(
                name="search_documents",
                description=(
                    "Search documents by title keyword. Returns at most 25 matches "
                    "and reports `truncated: true` when more may exist — narrow the "
                    "query rather than repeating it"
                ),
                input_schema=agentcore.SchemaDefinition(
                    type=agentcore.SchemaDefinitionType.OBJECT,
                    properties={
                        "query": agentcore.SchemaDefinition(
                            type=agentcore.SchemaDefinitionType.STRING,
                            description="The search query to match against document titles",
                        ),
                        # No served_scope and no context property. The served
                        # scope and the vended tenant_credentials are injected
                        # by the REQUEST interceptor at arguments["context"]
                        # AFTER the model writes the call, so they travel only
                        # forward and are never advertised to the model in the
                        # published schema.
                    },
                    required=["query"],
                ),
            ),
        ]),
    )

    # -- Reply target --
    agentcore.GatewayTarget.for_lambda(
        scope,
        "ReplyTarget",
        gateway_target_name="Reply",
        gateway=gateway,
        lambda_function=tool_fns["reply"],
        tool_schema=agentcore.ToolSchema.from_inline([
            agentcore.ToolDefinition(
                name="reply",
                description=(
                    "Post a reply on an existing document. The reply is appended "
                    "to that document's conversation, which holds at most 50 "
                    "entries — a document already at that limit accepts no "
                    "further replies. Replying cannot create a document: an id "
                    "naming no existing document is refused, not materialised"
                ),
                input_schema=agentcore.SchemaDefinition(
                    type=agentcore.SchemaDefinitionType.OBJECT,
                    properties={
                        "doc_id": agentcore.SchemaDefinition(
                            type=agentcore.SchemaDefinitionType.STRING,
                            description="The document to post the reply on",
                        ),
                        "body": agentcore.SchemaDefinition(
                            type=agentcore.SchemaDefinitionType.STRING,
                            description=(
                                "The reply body text (maximum 4000 bytes, UTF-8)"
                            ),
                        ),
                        # No served_scope and no context property. reply
                        # builds the composite key {scope, doc_id}; the scope
                        # and the vended tenant_credentials arrive via the
                        # REQUEST interceptor at arguments["context"] AFTER the
                        # model writes the call, so they travel only forward and
                        # are never advertised to the model in the published
                        # schema.
                    },
                    required=["doc_id", "body"],
                ),
            ),
        ]),
    )





