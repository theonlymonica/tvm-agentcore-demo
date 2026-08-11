# Dockerfile for the support agent — targets linux/arm64 as required by
# AgentCore Runtime (a wrong-architecture package fails deploy).
# Reference: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/getting-started-custom.html
#
# The agent must expose:
#   POST /invocations  — agent interaction endpoint
#   GET  /ping         — health check endpoint
# and listen on port 8080.
#
# Configuration is injected via environment variables at deploy time:
#   BEDROCK_MODEL_ID — e.g. us.anthropic.claude-sonnet-4-6
#   AWS_REGION       — e.g. us-east-1
#   GATEWAY_URL      — the AgentCore Gateway MCP endpoint URL
# Optional:
#   MAX_REQUEST_BODY_BYTES — /invocations body cap; defaults to 256 KiB
#                            (see request_limits.py)

# Digest-pinned so a rebuild resolves the same base layers every time. The
# digest is the multi-arch OCI index for `python:3.14-slim-bookworm`, so
# --platform=linux/arm64 still selects the arm64/v8 manifest from within it.
#
# Rolling the pin is Dependabot's job, not a manual chore: .github/dependabot.yml
# declares a `docker` ecosystem for this directory, and a digest pin is what
# makes that target useful — it can propose a reviewable bump, which a floating
# tag gives it no way to express. Never drop this back to a bare tag.
FROM --platform=linux/arm64 python:3.14-slim-bookworm@sha256:3299f3062a4f3da96c403bfeec3f042768c0bf8a72eee6dd5d90e3a1368ff334

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source files. Every .py in agent/ must be listed here -- a module
# that main.py imports but the image does not carry fails at container start.
# tests/test_agent_request_limits.py asserts this list stays complete.
COPY main.py ./
COPY agent_core.py ./
COPY request_limits.py ./

# Runtime-only env, set AFTER the pip install so build-time bytecode
# precompilation of the installed packages is untouched. Bytecode caching for
# the app modules is pointless anyway: /app is root-owned and the runtime user
# below cannot write to it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Unprivileged runtime account. Deliberately NOT changing ownership of /app:
# the source stays root-owned and world-readable, so the agent process can
# execute its own code but cannot rewrite it. It does get a real home
# directory, because dependencies resolved from lower-bound-only requirements
# (strands-agents, mcp, boto3, ...) may write a cache or config under $HOME on
# first use, and a non-existent HOME would surface as a FileNotFoundError at
# import or first tool call. AgentCore Runtime's container contract (ARM64,
# 0.0.0.0:8080, /invocations + /ping) says nothing about the user, and 8080 is
# above 1024, so an unprivileged account can still bind it.
# Not --system: these are explicit ids well above the system range, which is
# what container hardening guides call for, and pairing --system with a uid
# above SYS_UID_MAX makes useradd warn on every build.
RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 \
       --create-home --home-dir /home/agent \
       --shell /usr/sbin/nologin agent
USER 10001:10001

# Expose the port expected by AgentCore Runtime
EXPOSE 8080

# No HEALTHCHECK — but the blocker is gone, so this is now a choice rather than
# a constraint. /invocations used to run the agent synchronously on the event
# loop, so GET /ping could not be served while an invocation was in flight and a
# probe would have reported a busy container as unhealthy. main.py
# now offloads the run with asyncio.to_thread, so /ping stays answerable
# throughout. A HEALTHCHECK is therefore safe to add whenever it earns its keep;
# it is left out here because AgentCore Runtime polls /ping over its own service
# contract and ignores Docker health status, so it would only serve plain
# `docker run` / Compose / ECS deployments of this image.

# Entrypoint: run the Strands agent server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
