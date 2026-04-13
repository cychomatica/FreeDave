"""
Utilities for reproducible / deterministic generation across baseline and FreeDave.

Usage
-----
Call ``setup_deterministic_env(seed)`` once at process start (before model
loading, if possible).  Then pass ``deterministic=True`` when constructing
``DLMGeneration`` so that every forward pass is routed through the same SDPA
kernel regardless of whether an explicit attention mask is present.

Why this matters
~~~~~~~~~~~~~~~~
Baseline decoding often passes ``attention_mask=None`` to the model, allowing
PyTorch to select Flash Attention.  FreeDave constructs an explicit 4-D tree
attention mask, which disables Flash Attention and triggers the *math* or
*memory-efficient* SDPA backend.  Different SDPA implementations produce
slightly different floating-point results, which cascade through argmax /
confidence comparisons and cause token-level divergence even at temperature 0.

Forcing a single SDPA backend for both paths eliminates this divergence source.
"""

import os
import logging
import torch
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

_DETERMINISTIC_ENV_CONFIGURED = False


def setup_deterministic_env(seed: int = 42) -> None:
    """One-shot global configuration for deterministic CUDA execution.

    Sets random seeds and enables PyTorch deterministic algorithms so that
    baseline and FreeDave follow identical floating-point code-paths.

    Must be called **before** model loading / first CUDA kernel launch for
    ``CUBLAS_WORKSPACE_CONFIG`` to take effect.
    """
    global _DETERMINISTIC_ENV_CONFIGURED
    if _DETERMINISTIC_ENV_CONFIGURED:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    _DETERMINISTIC_ENV_CONFIGURED = True
    logger.info(
        "Deterministic environment configured (seed=%d, cudnn.deterministic=True, "
        "deterministic_algorithms=True, CUBLAS_WORKSPACE_CONFIG=%s)",
        seed,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
    )


# ---------------------------------------------------------------------------
# SDPA kernel context manager
# ---------------------------------------------------------------------------

def _has_sdpa_kernel_ctx() -> bool:
    """Check whether the running PyTorch exposes an SDPA-backend selector."""
    try:
        # PyTorch >= 2.2
        from torch.nn.attention import sdpa_kernel, SDPBackend  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        # PyTorch 2.0 – 2.1
        torch.backends.cuda.sdp_kernel  # noqa: B018
        return True
    except AttributeError:
        return False


@contextmanager
def deterministic_sdpa(backend: Optional[str] = None):
    """Context manager that forces a single SDPA backend.

    Parameters
    ----------
    backend : str or None
        ``"math"`` (default / safest – identical across mask-present and
        mask-absent calls), ``"mem_efficient"``, or ``"flash"``.
        ``None`` selects ``"math"``.

    The guard is a no-op when the running PyTorch version does not support
    backend selection (< 2.0).
    """
    if backend is None:
        backend = "math"

    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        _BACKEND_MAP = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        }
        if backend not in _BACKEND_MAP:
            raise ValueError(
                f"Unknown SDPA backend {backend!r}; choose from {list(_BACKEND_MAP)}"
            )
        with sdpa_kernel(_BACKEND_MAP[backend]):
            yield
        return
    except ImportError:
        pass

    try:
        enable = {
            "math": dict(enable_flash=False, enable_math=True, enable_mem_efficient=False),
            "flash": dict(enable_flash=True, enable_math=False, enable_mem_efficient=False),
            "mem_efficient": dict(enable_flash=False, enable_math=False, enable_mem_efficient=True),
        }
        if backend not in enable:
            raise ValueError(
                f"Unknown SDPA backend {backend!r}; choose from {list(enable)}"
            )
        with torch.backends.cuda.sdp_kernel(**enable[backend]):
            yield
        return
    except (AttributeError, RuntimeError):
        pass

    yield
