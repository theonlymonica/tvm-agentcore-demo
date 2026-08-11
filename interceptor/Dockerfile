# Scope-injecting REQUEST interceptor — container-image Lambda. Built on the
# official AWS Lambda Python 3.14 base image
# (Amazon Linux 2023) so the native `cryptography` dependency (via PyJWT[crypto],
# used for JWKS signature verification in jwt_claims) matches the Lambda runtime
# and architecture. Build context is this `interceptor/` directory.
#
# Digest-pinned so a rebuild resolves the same base layers every time — the
# digest is the multi-arch manifest list, so the CDK asset's LINUX_ARM64
# platform still resolves the arm64/v8 manifest from within it. This pins the
# BASE ONLY: the pip install below still floats (PyJWT patch level and the
# transitively resolved `cryptography` version), so this image is not
# reproducible end to end.
#
# No USER directive: the Lambda execution environment already runs the handler
# as an unprivileged sandbox account regardless of the image's user, and the
# base image's bootstrap expects the default.
#
# Rolling the pin is Dependabot's job (.github/dependabot.yml declares a
# `docker` ecosystem for this directory), and a digest pin is what gives it a
# bump it can actually express. Never drop this back to a bare tag.
FROM public.ecr.aws/lambda/python:3.14@sha256:b3a2430425b2214e67eb56bc918ae35b7868596a27eea56be48976df2503d8af

# JWT signature verification dependency, pinned in requirements.txt so the
# version lives where a reviewer looks for it and where Dependabot can bump it
# (neither ecosystem reads a version out of a RUN line). PyJWT[crypto] pulls in
# `cryptography`, installed as a prebuilt abi3 wheel — no compilation needed on
# this base image.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the interceptor package into the task root so the handler resolves as
# `interceptor.handler.handler` (matching the previous zip packaging).
COPY . ${LAMBDA_TASK_ROOT}/interceptor/

CMD ["interceptor.handler.handler"]
