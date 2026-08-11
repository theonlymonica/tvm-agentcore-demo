"""
CDK app entry point.

This is the top-level CDK application that instantiates the main stack.

The stack must be pinned to a specific account and region so that
CloudFormation validates the template against that region's type registry.
Without an explicit env, environment-agnostic stacks may fail validation
for newer resource types like AWS::BedrockAgentCore::*.
"""

import os
import sys
from pathlib import Path

import aws_cdk as cdk

# Add repo root to sys.path so shared.config_loader is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from shared.config_loader import load_config
from scoped_credentials_stack import ScopedCredentialsStack

config = load_config()

app = cdk.App()
ScopedCredentialsStack(
    app,
    "ScopedCredentialsStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=config.aws_region,
    ),
)
app.synth()
