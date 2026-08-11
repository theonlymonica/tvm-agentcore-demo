"""
CDK stack that provisions the multi-tenant data isolation infrastructure.

Resources:
- Managed Cognito identity layer: user pool + SRP-only public app client +
  one Cognito group per scope (via the auth_resources module)
- DocumentsTable (composite key served_scope + document_id), scoped DynamoDB
  roles, and the seed custom resource (all via the data_resources /
  documents_roles modules)
- Three tool Lambdas (read_document, search_documents, reply)
- REQUEST interceptor Lambda (decodes the validated JWT claim)
- RESPONSE interceptor Lambda (zip; strips credential-shaped material
  from tool replies)
- AgentCore Gateway (CUSTOM_JWT inbound auth) with MCP sessions + REQUEST and
  RESPONSE interceptors
- AgentCore Policy Engine + Cedar policies (delegated to policy_resources)
- AgentCore Runtime (delegated to runtime_resources module)

IAM model — the served scope is enforced before a tool ever runs:
- read_document / search_documents: NO direct DynamoDB grant. Each exec role is
  granted sts:AssumeRole on DocumentsAccessRole and assumes it at request time
  with a LeadingKeys session policy scoped to the served partition.
- reply: NO direct table-wide dynamodb:UpdateItem grant. Its exec role is granted
  sts:AssumeRole on DocumentsWriteRole and assumes it with a scoped write session
  policy.
- REQUEST interceptor: NO data-plane permissions. The interceptor only decodes
  the gateway-validated JWT claim to derive served_scope; it reads no DynamoDB
  and no SSM.
- RESPONSE interceptor: a zip-packaged Lambda that strips credential-shaped
  material from tool replies before the Gateway returns them. It holds NO
  data-plane permission and NO sts:AssumeRole — its execution role is the
  default Lambda logging role only.
The sts:AssumeRole grants and the scoped-role trust narrowing are wired in
cdk/lambda_iam.py. The RESPONSE interceptor is a post-stage egress scrubber
(defense in depth); the primary enforcement remains the structural scope
confinement applied before the tool stage.

Documentation references:
- aws_cdk.aws_lambda.Function (runtime PYTHON_3_14, environment, role):
  https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_lambda/Function.html
- IAM sts:AssumeRole (assume-role grant + trust policy):
  https://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import aws_cdk as cdk
import aws_cdk.aws_ecr_assets as ecr_assets
import aws_cdk.aws_lambda as lambda_
from constructs import Construct

from asset_packaging import ASSET_EXCLUDE, python_lambda_code
from auth_resources import SCOPE_GROUPS, create_auth_resources
from data_resources import DataResources, create_data_resources
from gateway_resources import create_gateway
from lambda_iam import wire_lambda_iam
from observability import TOOL_RESERVED_CONCURRENCY, lambda_log_group
from policy_resources import add_cedar_policies, create_policy_engine
from runtime_resources import create_runtime

# Add the repo root to sys.path so we can import shared.config_loader
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from shared.config_loader import load_config


class ToxicFlowStack(cdk.Stack):
    """Main CDK stack for the multi-tenant data isolation demo.

    Provisions the data layer (DocumentsTable, scoped roles, seed), the three
    tool Lambdas plus the REQUEST interceptor with least-privilege IAM
    (scoped-role assumption, no direct data-plane grants), and the AgentCore
    Gateway/Policy/Runtime.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        config = load_config()

        # ---------------------------------------------------------------
        # Identity layer: MANAGED Cognito user pool + app client + scope
        # groups. Created before the Lambdas because the REQUEST
        # interceptor's JWT-verification env (issuer + JWKS URL) is derived
        # from the pool, and before the gateway because the CUSTOM_JWT
        # authorizer points at the app client.
        # ---------------------------------------------------------------
        self.auth = create_auth_resources(self)

        # ---------------------------------------------------------------
        # Data layer: DocumentsTable, scoped roles, seed
        # ---------------------------------------------------------------
        data = create_data_resources(self)

        # ---------------------------------------------------------------
        # Lambda functions (three tools + REQUEST interceptor)
        # ---------------------------------------------------------------
        self._create_lambda_functions(data)

        # ---------------------------------------------------------------
        # IAM: scoped-role assumption grants + scoped-role trust narrowing.
        # Hardening: the INTERCEPTOR assumes the scoped roles and vends
        # credentials to the tools. The tool exec roles get NO sts:AssumeRole
        # and NO DynamoDB permission.
        # ---------------------------------------------------------------
        wire_lambda_iam(
            data=data,
            interceptor_fn=self.interceptor_fn,
        )

        # ---------------------------------------------------------------
        # AgentCore Policy Engine
        # ---------------------------------------------------------------
        self.policy_engine = create_policy_engine(scope=self)

        # ---------------------------------------------------------------
        # AgentCore Gateway + targets + REQUEST interceptor + policy
        # ---------------------------------------------------------------
        self.gateway = create_gateway(
            scope=self,
            interceptor_fn=self.interceptor_fn,
            policy_engine=self.policy_engine,
            # Managed Cognito pool/client: supplies the CUSTOM_JWT
            # discovery URL + allowed client id as deploy-time tokens.
            auth=self.auth,
            # The zip-packaged RESPONSE interceptor, wired as the gateway's
            # RESPONSE interceptor.
            response_interceptor_fn=self.response_interceptor_fn,
            tool_fns={
                "read_document": self.read_document_fn,
                "search_documents": self.search_documents_fn,
                "reply": self.reply_fn,
            },
        )

        # ---------------------------------------------------------------
        # Cedar policies
        # ---------------------------------------------------------------
        add_cedar_policies(
            scope=self,
            policy_engine=self.policy_engine,
            gateway=self.gateway,
        )

        # ---------------------------------------------------------------
        # AgentCore Runtime
        # ---------------------------------------------------------------
        self.runtime = create_runtime(
            scope=self,
            gateway=self.gateway,
            config=config,
        )

    # -------------------------------------------------------------------
    # Lambda function factory
    # -------------------------------------------------------------------

    def _create_lambda_functions(self, data: DataResources) -> None:
        """Create the three tool Lambdas, the REQUEST interceptor, and the
        RESPONSE interceptor Lambdas.

        Bundling:
        - Tool Lambdas: bundled from the entire tools/ directory so that the
          common/ package (scoped_credentials) is available on the Python path.
          Handler paths are read_document.handler.handler, etc.
        - Interceptor Lambda: bundled from the repo root so the interceptor/
          package resolves; REQUEST handler interceptor.handler.handler.
        - RESPONSE interceptor Lambda: a plain zip bundled from the repo-root
          response_interceptor/ directory (handler handler.handler); distinct
          from the container-image REQUEST interceptor.

        Environment:
        - read/search: DOCUMENTS_TABLE_NAME (consumed by
          tools/common/scoped_credentials.py).
        - reply: DOCUMENTS_TABLE_NAME.
        - interceptor: scoped-role ARNs + table ARN (for the AssumeRole vend)
          plus the JWT-verification inputs.

        Args:
            data: The DataResources containing table and scoped-role references.
        """
        _root = os.path.join(os.path.dirname(__file__), "..")
        tools_dir = os.path.join(_root, "tools")
        interceptor_dir = os.path.join(_root, "interceptor")
        response_interceptor_dir = os.path.join(_root, "response_interceptor")

        table_arn = data.documents_table_arn
        table_name = data.documents_table.table_name

        # -- read_document Lambda (uses interceptor-vended creds) --
        # No role ARN / table ARN env: the tool no longer assumes a role. It only
        # needs the table NAME to bind the boto3 Table to the vended credentials.
        self.read_document_fn = lambda_.Function(
            self,
            "ReadDocumentFunction",
            function_name="toxic-flow-read-document",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler="read_document.handler.handler",
            code=python_lambda_code(tools_dir),
            timeout=cdk.Duration.seconds(10),
            # Operational posture: a bounded, stack-owned log group and a
            # per-tool concurrency cap so one caller cannot exhaust account
            # concurrency. See cdk/observability.py.
            log_group=lambda_log_group(
                self,
                "ReadDocumentFunctionLogGroup",
                function_name="toxic-flow-read-document",
            ),
            reserved_concurrent_executions=TOOL_RESERVED_CONCURRENCY,
            environment={
                "DOCUMENTS_TABLE_NAME": table_name,
            },
        )

        # -- search_documents Lambda (uses interceptor-vended creds) --
        self.search_documents_fn = lambda_.Function(
            self,
            "SearchDocumentsFunction",
            function_name="toxic-flow-search-documents",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler="search_documents.handler.handler",
            code=python_lambda_code(tools_dir),
            timeout=cdk.Duration.seconds(10),
            log_group=lambda_log_group(
                self,
                "SearchDocumentsFunctionLogGroup",
                function_name="toxic-flow-search-documents",
            ),
            reserved_concurrent_executions=TOOL_RESERVED_CONCURRENCY,
            environment={
                "DOCUMENTS_TABLE_NAME": table_name,
            },
        )

        # -- reply Lambda (uses interceptor-vended WRITE creds) --
        self.reply_fn = lambda_.Function(
            self,
            "ReplyFunction",
            function_name="toxic-flow-reply",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler="reply.handler.handler",
            code=python_lambda_code(tools_dir),
            timeout=cdk.Duration.seconds(10),
            log_group=lambda_log_group(
                self,
                "ReplyFunctionLogGroup",
                function_name="toxic-flow-reply",
            ),
            reserved_concurrent_executions=TOOL_RESERVED_CONCURRENCY,
            environment={
                "DOCUMENTS_TABLE_NAME": table_name,
            },
        )

        # Interceptor environment: scoped-role ARNs + table ARN (for the
        # AssumeRole vend) plus the JWT-verification inputs.
        #
        # COGNITO_ALLOWED_CLIENT_IDS pins the app client IN the interceptor. The
        # gateway CUSTOM_JWT authorizer already validates the access token's
        # `client_id` against its own `allowedClients` (cdk/gateway_resources.py),
        # but the interceptor's in-Lambda verification is defense in depth and must
        # not depend on that upstream config staying correct — so it re-checks the
        # same claim. Both consumers read the SAME managed app client
        # (`self.auth.app_client_id`, a deploy-time token — never a hardcoded id),
        # which is what keeps them from drifting apart. jwt_claims fails CLOSED if
        # this var is unset, so it is required, not optional.
        interceptor_env = {
            "DOCUMENTS_ACCESS_ROLE_ARN": data.documents_access_role_arn,
            "DOCUMENTS_WRITE_ROLE_ARN": data.documents_write_role_arn,
            "DOCUMENTS_TABLE_ARN": table_arn,
            # Derived from the MANAGED pool, so the interceptor always
            # verifies tokens against the pool this stack owns.
            "COGNITO_JWKS_URL": self.auth.jwks_url,
            "COGNITO_ISSUER": self.auth.issuer,
            "COGNITO_ALLOWED_CLIENT_IDS": self.auth.app_client_id,
            # The known-scope set, from the same constant that creates the
            # Cognito groups — so changing SCOPE_GROUPS is enough and the
            # interceptor's module default never has to be edited in parallel.
            "KNOWN_SCOPE_GROUPS": ",".join(SCOPE_GROUPS),
        }

        # -- REQUEST interceptor Lambda (CONTAINER IMAGE) --
        # The interceptor VERIFIES the JWT
        # signature (RS256 against the Cognito pool JWKS), issuer, and expiry in
        # `interceptor/jwt_claims.py` BEFORE deriving served_scope — it no longer
        # trusts the gateway CUSTOM_JWT check alone (defense in depth). That
        # verification needs `cryptography` (via PyJWT[crypto]), a native binary a
        # zip asset cannot bundle, so the interceptor is packaged as a
        # container-image Lambda built on `public.ecr.aws/lambda/python:3.14`
        # (see interceptor/Dockerfile). COGNITO_JWKS_URL / COGNITO_ISSUER supply
        # the verification inputs (jwt_claims fails closed if either is unset).
        #
        # The vending path: the interceptor derives served_scope, then assumes
        # DocumentsAccessRole
        # (read) or DocumentsWriteRole (reply) with a LeadingKeys session policy
        # (DurationSeconds=900) and injects the temporary credentials into
        # params.arguments as UNDECLARED fields. It therefore keeps the scoped-role
        # ARNs + table ARN env. The sts:AssumeRole grant + scoped-role trust are
        # wired in cdk/lambda_iam.py.
        #
        # PACKAGING NOTES:
        # - Construct id "SessionGuardFunction" is UNCHANGED so the auto-created
        #   execution role's logical id (SessionGuardFunctionServiceRole...) is
        #   preserved — the scoped-role trust in lambda_iam.py names that role ARN,
        #   so keeping the id keeps the trust valid without a second deploy.
        # - `function_name` is deliberately OMITTED (auto-generated). Switching
        #   PackageType Zip->Image requires CloudFormation REPLACEMENT of the
        #   function; a fixed function name would collide during the
        #   create-new-then-delete-old replacement.
        # - DockerImageFunction takes NO `runtime`/`handler` — the image CMD
        #   (`interceptor.handler.handler`) is the handler. Built for ARM64 to
        #   match the local machine + the agent container; the interceptor/
        #   .dockerignore keeps the build context minimal.
        self.interceptor_fn = lambda_.DockerImageFunction(
            self,
            "SessionGuardFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                interceptor_dir,
                platform=ecr_assets.Platform.LINUX_ARM64,
                # Alongside interceptor/.dockerignore, which only reaches
                # root-level caches. See cdk/asset_packaging.py.
                exclude=ASSET_EXCLUDE,
            ),
            architecture=lambda_.Architecture.ARM_64,
            # Headroom for the cold-start JWKS fetch (~once per container) +
            # RS256 verify + the sts:AssumeRole vend on the first request.
            timeout=cdk.Duration.seconds(15),
            memory_size=256,
            # Bounded, stack-owned log group. `function_name`
            # is deliberately auto-generated here (see the PACKAGING NOTES
            # above), so the group carries the stable `toxic-flow-session-guard`
            # label instead of the physical function name — an explicit
            # LoggingConfig group need not match /aws/lambda/<physical-name>.
            # No reserved concurrency: this interceptor is on EVERY gateway
            # request, so capping it would throttle all tenants at once
            # (cdk/observability.py records the reasoning).
            log_group=lambda_log_group(
                self,
                "SessionGuardFunctionLogGroup",
                function_name="toxic-flow-session-guard",
            ),
            environment=interceptor_env,
        )

        # -- RESPONSE interceptor Lambda (ZIP PACKAGE) --
        # Distinct from the container-image REQUEST interceptor above: this is a
        # plain zip Lambda bundled straight from the repo-root
        # `response_interceptor/` directory (handler `handler.handler` ->
        # response_interceptor/handler.py, which uses the pure
        # credential_scrubber.py). It strips credential-shaped material from tool
        # replies before the Gateway returns them. The scrubber is pure stdlib
        # (copy + re), so nothing forces a container image: the RESPONSE
        # interceptor stays a separate zip Lambda rather than being folded into
        # the container-image interceptor/.
        #
        # LEAST PRIVILEGE: created with NO `role=`
        # override, so CDK assigns the DEFAULT Lambda execution role (CloudWatch
        # Logs only). It is deliberately NOT passed to wire_lambda_iam and receives
        # NO DynamoDB permission and NO sts:AssumeRole grant anywhere in the stack
        # — it never touches the data plane, it only rewrites the reply body in
        # memory. `function_name` follows the frozen `toxic-flow-*` convention and
        # the runtime is PYTHON_3_14 like every other Lambda here.
        self.response_interceptor_fn = lambda_.Function(
            self,
            "ResponseInterceptorFunction",
            function_name="toxic-flow-response-interceptor",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler="handler.handler",
            code=python_lambda_code(response_interceptor_dir),
            timeout=cdk.Duration.seconds(10),
            # Bounded, stack-owned log group. Like the
            # REQUEST interceptor this sits on every gateway response, so it
            # takes no reserved-concurrency cap.
            log_group=lambda_log_group(
                self,
                "ResponseInterceptorFunctionLogGroup",
                function_name="toxic-flow-response-interceptor",
            ),
        )
