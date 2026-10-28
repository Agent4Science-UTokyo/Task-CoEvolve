"""Vanilla Terminus-2 baseline with no KIRA-specific modifications."""

import litellm_azure_cancel_fix  # noqa: F401  (side-effect import: works around a litellm Azure timeout-swallow bug)
from harbor.agents.terminus_2 import Terminus2

AgentHarness = Terminus2
