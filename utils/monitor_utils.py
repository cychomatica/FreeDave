'''
Methods to monitor model forward passes
'''

import torch
import time
from contextlib import contextmanager

class ForwardMonitor:
    '''Uses PyTorch hooks to count forward passes and track wall-clock time.'''
    
    def __init__(self, model):
        self.model = model
        self.nfe = 0
        self.elapsed = 0.0
        self.hook_handle = None
        self._start_time = None
        
    def _forward_hook(self, module, input, output):
        self.nfe += 1
        return output
        
    def start(self):
        self.nfe = 0
        self._start_time = time.time()
        self.hook_handle = self.model.register_forward_hook(self._forward_hook)
        
    def stop(self):
        self.elapsed = time.time() - self._start_time if self._start_time else 0.0
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
        
    def reset(self):
        self.nfe = 0
        self.elapsed = 0.0
        self._start_time = None

    def avg_forward_time(self):
        return self.elapsed / self.nfe if self.nfe > 0 else 0.0

    def get_nfe(self):
        return self.nfe

    def get_elapsed_time(self):
        return self.elapsed
        
    @contextmanager
    def count(self):
        '''Context manager that tracks both NFE and wall-clock time.'''
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def __str__(self):
        return f"time: {self.elapsed:.4f}s | nfe: {self.nfe} | avg forward: {self.avg_forward_time()*1000:.2f}ms"

class CudaTimer:
    '''CUDA-aware timer that properly synchronizes GPU operations'''
    
    def __init__(self, name='Operation'):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        
    @property
    def elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


@contextmanager
def cuda_timer(name='Operation'):
    '''Context manager for timing CUDA operations
    
    Usage:
        with cuda_timer("Forward pass") as timer:
            # your GPU operations
            output = model(input)
        print(f"Time: {timer.elapsed_time:.4f}s")
    '''
    timer = CudaTimer(name)
    torch.cuda.synchronize()
    start_time = time.time()
    
    try:
        yield timer
    finally:
        torch.cuda.synchronize()
        end_time = time.time()
        timer.start_time = start_time
        timer.end_time = end_time