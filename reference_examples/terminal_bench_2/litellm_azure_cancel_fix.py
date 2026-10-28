"""Workaround for litellm's Azure provider swallowing asyncio.CancelledError.

Details and repro steps: scripts/test/azure_cancel_swallow_bug.md

`litellm.llms.azure.azure.AzureChatCompletion.acompletion()` has:

    except asyncio.CancelledError as e:
        logging_obj.post_call(...)
        raise AzureOpenAIError(status_code=500, message=str(e))

which converts harbor's agent timeout (a single `cancel()` via `asyncio.wait_for`)
into a plain `Exception`, swallowing it. This makes baseline_kira.py's tenacity
`@retry` mistake it for a retryable transient error, permanently defeating the
timeout (reproduced against the real API: scripts/test/repro_cancel_swallow.py).

Since `AzureOpenAIError` is raised via implicit chaining rather than
`raise ... from e`, the real `CancelledError` survives in `__context__`. This
module wraps `AzureChatCompletion.acompletion` so that when it catches an
`AzureOpenAIError`, it checks `__cause__`/`__context__` for a hidden
`CancelledError` and re-raises that instead, restoring correct cancellation
propagation (staying a `BaseException` so it slips past tenacity and reaches
harbor). Applied as an import-time side effect — just importing this module
is enough. Once litellm fixes the underlying bug this module becomes a no-op
(re-check with the repro script after upgrading litellm).

Lives at the repo root (like anthropic_caching.py), not under agents/: files
under agents/ other than BASELINE_FILES are dropped by meta_harness.py's
snapshot_code() and are subject to the candidate archive/restore dance, so a
shared infra fix has to sit outside that directory.
"""

import asyncio

from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.common_utils import AzureOpenAIError

_original_acompletion = AzureChatCompletion.acompletion


async def _cancel_safe_acompletion(self, *args, **kwargs):
    try:
        return await _original_acompletion(self, *args, **kwargs)
    except AzureOpenAIError as e:
        cancelled = (
            e.__cause__ if isinstance(e.__cause__, asyncio.CancelledError) else e.__context__
        )
        if isinstance(cancelled, asyncio.CancelledError):
            raise cancelled
        raise


AzureChatCompletion.acompletion = _cancel_safe_acompletion
