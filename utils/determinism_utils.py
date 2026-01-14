import torch
import os
import random
import numpy as np
from contextlib import contextmanager
from functools import wraps
from termcolor import cprint

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
    cprint(f"Determinism enabled with seed: {seed}", color='yellow')

@contextmanager
def deterministic(enabled: bool = True, seed: int = 42):
    """
    Context manager to enable deterministic behavior for a block of code.
    Note: Some settings (like CUDA flags) are global and will persist.
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
