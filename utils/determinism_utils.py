import torch
import os
import random
import numpy as np
from contextlib import contextmanager
from functools import wraps
from termcolor import cprint
import sys

def setup_determinism(seed: int = 42):
    """
    Sets global settings for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Disable non-deterministic SDPA backends
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # Disable liger_kernel imports
    sys.modules['liger_kernel'] = None
    sys.modules['liger_kernel.ops'] = None
    sys.modules['liger_kernel.ops.swiglu'] = None

    # Control modeling/sdar flags via environment variables
    # These must be set BEFORE importing modeling.sdar
    os.environ["SDAR_USE_FLASH_ATTN"] = "0"
    os.environ["SDAR_USE_LIGER_KERNEL"] = "0"

    os.environ["SDAR_USE_EAGER_ATTN"] = "1"

    cprint(f"Determinism enabled with seed: {seed}", color='yellow')

@contextmanager
def deterministic(enabled: bool = True, seed: int = 42):
    """
    Context manager to enable deterministic behavior for a block of code.
    Note: Some settings (like CUDA flags) are global and will persist.
    
    IMPORTANT: This context manager should be entered BEFORE importing modeling.sdar
    to ensure the environment variables take effect.
    """

    if not enabled:
        yield
        return

    # Store current RNG states
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    
    # Store current global flags
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_use_deterministic = torch.are_deterministic_algorithms_enabled()

    # Store SDPA backend states
    prev_flash_sdp = torch.backends.cuda.flash_sdp_enabled()
    prev_mem_efficient_sdp = torch.backends.cuda.mem_efficient_sdp_enabled()
    prev_math_sdp = torch.backends.cuda.math_sdp_enabled()
    
    # Store SDAR environment variables
    prev_sdar_flash_attn = os.environ.get("SDAR_USE_FLASH_ATTN")
    prev_sdar_liger_kernel = os.environ.get("SDAR_USE_LIGER_KERNEL")
    prev_sdar_eager_attn = os.environ.get("SDAR_USE_EAGER_ATTN")
    
    try:
        setup_determinism(seed)
        yield
    finally:
        # Restore RNG states
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        
        # Restore global flags
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.use_deterministic_algorithms(prev_use_deterministic)
        
        # Restore SDPA backend states
        torch.backends.cuda.enable_flash_sdp(prev_flash_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(prev_mem_efficient_sdp)
        torch.backends.cuda.enable_math_sdp(prev_math_sdp)
        
        # Restore SDAR environment variables
        if prev_sdar_flash_attn is not None:
            os.environ["SDAR_USE_FLASH_ATTN"] = prev_sdar_flash_attn
        elif "SDAR_USE_FLASH_ATTN" in os.environ:
            del os.environ["SDAR_USE_FLASH_ATTN"]
        if prev_sdar_liger_kernel is not None:
            os.environ["SDAR_USE_LIGER_KERNEL"] = prev_sdar_liger_kernel
        elif "SDAR_USE_LIGER_KERNEL" in os.environ:
            del os.environ["SDAR_USE_LIGER_KERNEL"]
        if prev_sdar_eager_attn is not None:
            os.environ["SDAR_USE_EAGER_ATTN"] = prev_sdar_eager_attn
        elif "SDAR_USE_EAGER_ATTN" in os.environ:
            del os.environ["SDAR_USE_EAGER_ATTN"]

def deterministic_run(seed: int = 42):
    """
    Decorator to enable deterministic behavior for a function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with deterministic(seed):
                return func(*args, **kwargs)
        return wrapper
    return decorator
