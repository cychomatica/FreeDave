'''
Methods to monitor model forward passes
'''

import torch
import time
from contextlib import contextmanager

class ForwardPassCounter:
    '''A simple counter for tracking forward passes.'''
    
    def __init__(self):
        self.count = 0
        
    def reset(self):
        self.count = 0
        
    def increment(self):
        self.count += 1
        
    def __str__(self):
        return f"Forward passes: {self.count}"

class ForwardHookCounter:
    '''Uses PyTorch hooks to count forward passes without modifying the model.'''
    
    def __init__(self, model):
        self.model = model
        self.counter = ForwardPassCounter()
        self.hook_handle = None
        
    def _forward_hook(self, module, input, output):
        '''Hook function that gets called on every forward pass.'''
        self.counter.increment()
        return output
        
    def start_counting(self):
        '''Start monitoring forward passes.'''
        self.counter.reset()
        self.hook_handle = self.model.register_forward_hook(self._forward_hook)
        
    def stop_counting(self):
        '''Stop monitoring and return count.'''
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
        return self.counter.count
        
    def reset_count(self):
        self.counter.reset()
        
    @contextmanager
    def count_context(self):
        '''Context manager for counting forward passes.'''
        self.start_counting()
        try:
            yield self.counter
        finally:
            self.stop_counting()

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