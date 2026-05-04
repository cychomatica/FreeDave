'''
Methods to monitor model forward passes
'''

import torch
import time
from contextlib import contextmanager
import functools
from typing import Optional
import contextvars
from typing import List as _List, Optional as _Optional

class ForwardMonitor:
    '''Uses PyTorch hooks to count forward passes and track wall-clock time.'''

    _active_stack_var: contextvars.ContextVar[_List["ForwardMonitor"]] = contextvars.ContextVar(
        "forward_monitor_active_stack", default=[]
    )
    
    def __init__(self, model):
        self.model = model
        self.nfe = 0
        self.elapsed = 0.0
        self.hook_handle = None
        self._start_time = None
        
    def _forward_hook(self, module, input, output):
        self.nfe += 1
        return output

    @classmethod
    def get_active(cls) -> _Optional["ForwardMonitor"]:
        stack = cls._active_stack_var.get()
        return stack[-1] if stack else None

    def _push_active(self):
        stack = list(self._active_stack_var.get())
        stack.append(self)
        self._active_stack_var.set(stack)

    def _pop_active(self):
        stack = list(self._active_stack_var.get())
        if stack and stack[-1] is self:
            stack.pop()
            self._active_stack_var.set(stack)
        
    def start(self):
        self.nfe = 0
        self._start_time = time.time()
        self._push_active()
        self.hook_handle = self.model.register_forward_hook(self._forward_hook)
        
    def stop(self):
        self.elapsed = time.time() - self._start_time if self._start_time else 0.0
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
        self._pop_active()
        
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

def debug_only(message: Optional[str] = None):
    def decorator(func):
        called = False
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal called
            if not called:
                print(f"Calling {func.__name__}. Make sure {func.__name__} is for debugging only." if message is None else message)
                called = True
            return func(*args, **kwargs)
        return wrapper
    return decorator