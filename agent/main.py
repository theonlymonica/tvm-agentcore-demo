"""
Support agent entrypoint for AgentCore Runtime.

Implements the AgentCore Runtime HTTP service contract:
  POST /invocations — agent interaction endpoint
  GET  /ping        — health check endpoint
  Port: 8080, Host: 0.0.0.0, Platform: ARM64

The agent:
  - Initializes a Strands Agent with model_id from config.json
  - Obtains session_id from the X-Amzn-Bedrock-AgentCore-Runtime-Session-Id header
  - Reads the user's Cognito access token from the invocation payload ("user_jwt")
    and forwards it to the AgentCore Gateway as the Authorization: Bearer header
    (the gateway inbound auth is CUSTOM_JWT)
  - Exposes the three gateway tools to the model
  - Lets the model choose tool order at runtime (no hardcoded sequence)
  - On tool-call failure: continues and records the failure

References:
  - AgentCore Runtime HTTP contract:
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html
  - Runtime sessions:
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html
  - Strands MCP tools:
    https://strandsagents.com/docs/user-guide/concepts/tools/mcp-tools/
  - Gateway inbound auth (CUSTOM_JWT):
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-inbound-auth.html
  - Gateway sessions:
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-sessions.html
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from agent_core import create_agent_and_invoke
from request_limits import BodyTooLarge, MalformedBody, parse_bounded_json

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("support_agent")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Support Agent", version="1.0.0")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class InvocationRequest(BaseModel):
    """Request body for the /invocations endpoint."""

    input: Dict[str, Any]


class InvocationResponse(BaseModel):
    """Response body for the /invocations endpoint."""

    output: Dict[str, Any]


# ---------------------------------------------------------------------------
# POST /invocations
# ---------------------------------------------------------------------------


@app.post("/invocations", response_model=InvocationResponse)
async def invoke_agent(request: Request) -> InvocationResponse:
    """Handle agent invocation requests from AgentCore Runtime.

    Extracts the session_id from the runtime session header, then invokes
    the Strands agent with the user's prompt, connecting to the AgentCore
    Gateway as an MCP client.
    """
    # Read the JSON body, bounded. The endpoint takes the raw Request (the
    # runtime delivers the payload directly and two payload shapes are
    # tolerated), so the InvocationRequest model does not apply and neither
    # uvicorn nor this app supplied any size cap: `await request.json()` would
    # buffer a body of any size. parse_bounded_json rejects a declared oversize
    # body, then caps the bytes actually read -- a chunked body carries no
    # Content-Length, so the streamed cap is the load-bearing check. The read and
    # parse live in request_limits so they are testable without fastapi; all this
    # endpoint owns is the mapping onto status codes.
    try:
        body = await parse_bounded_json(
            request.headers.get("content-length"), request.stream()
        )
    except BodyTooLarge as exc:
        logger.warning("Rejected an oversized invocation body.")
        raise HTTPException(status_code=413, detail=str(exc))
    except MalformedBody as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # AgentCore /invocations delivers the payload body directly, e.g.
    # {"prompt": "..."} (see runtime HTTP protocol contract). Read prompt
    # from the top level; tolerate a legacy {"input": {"prompt": ...}} shape.
    prompt = body.get("prompt", "")
    if not prompt and isinstance(body.get("input"), dict):
        prompt = body["input"].get("prompt", "")

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="No prompt found in input. Provide a 'prompt' key.",
        )

    # NOTE: there is deliberately NO per-invocation model override. The model is
    # fixed by bedrock_model_id (config.json / BEDROCK_MODEL_ID), which is also
    # what the Runtime role's Bedrock grant is pinned to, so a caller cannot
    # select an arbitrary or more expensive model. Any "model_id" key in the
    # request body is ignored.

    # The user's Cognito ACCESS token, forwarded by the operator console under
    # the contract field name "user_jwt".
    # Read from the top level, tolerating a legacy {"input": {"user_jwt": ...}}
    # shape. agent_core attaches it as the gateway "Authorization: Bearer <token>"
    # header so the CUSTOM_JWT gateway can validate it and the REQUEST interceptor
    # can derive served_scope from its claims.
    #
    # Fail-closed: if user_jwt is absent, no Authorization header is sent and the
    # CUSTOM_JWT gateway rejects the call with 401. We do NOT invent a fallback
    # token. SECURITY: never log the token value — only whether one was provided.
    user_jwt = body.get("user_jwt")
    if not user_jwt and isinstance(body.get("input"), dict):
        user_jwt = body["input"].get("user_jwt")

    # Obtain session_id from the AgentCore Runtime session header.
    # For HTTP protocol agents the header is:
    # X-Amzn-Bedrock-AgentCore-Runtime-Session-Id
    # Reference: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html
    session_id = request.headers.get("x-amzn-bedrock-agentcore-runtime-session-id")

    if not session_id:
        # Halt and return a session-unavailable error; never generate
        # a substitute identifier.
        logger.error("Session ID unavailable — no session header present.")
        raise HTTPException(
            status_code=400,
            detail=(
                "Session identifier unavailable. The AgentCore Runtime did not "
                "provide a session ID (X-Amzn-Bedrock-AgentCore-Runtime-Session-Id). "
                "Cannot proceed without a valid session."
            ),
        )

    # Log only whether a token was provided — never the token value itself.
    logger.info(
        "Invocation received. session_id=%s user_jwt_present=%s",
        session_id,
        bool(user_jwt),
    )

    try:
        # Offloaded to a worker thread, NOT called inline. create_agent_and_invoke
        # is fully synchronous and drives the Strands agent to completion — up to
        # MAX_AGENT_TURNS model turns, potentially minutes — so calling it directly
        # from this coroutine would hold the event loop for the whole run. Nothing
        # else on this worker could be served meanwhile, including GET /ping: the
        # container would be unable to report liveness precisely while it is busy
        # doing the work it exists for. Keeping the event loop free is what keeps
        # /ping answerable while an invocation is in flight.
        #
        # Safe to move off the main thread: agent_core holds no module-level
        # mutable state (only a logger, prompt constants and two scalars) and
        # builds its agent per call, so there is nothing shared to race on.
        result = await asyncio.to_thread(
            create_agent_and_invoke,
            session_id=session_id,
            prompt=prompt,
            user_jwt=user_jwt,
        )
    except Exception as exc:
        logger.exception("Agent invocation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {exc}",
        )

    return InvocationResponse(
        output={
            "message": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# GET /ping
# ---------------------------------------------------------------------------


@app.get("/ping")
async def ping() -> Dict[str, str]:
    """Health check endpoint required by AgentCore Runtime service contract.

    Reference:
      https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html
    """
    return {"status": "Healthy"}


# ---------------------------------------------------------------------------
# Direct run (local testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
