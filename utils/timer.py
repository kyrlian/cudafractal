from functools import wraps
from time import time

def timing_wrapper(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        # print(f'Function: {func.__name__} | Args: {args} | Took: {end_time - start_time:.4f} sec')
        print(f'Function: {func.__name__} Took: {end_time - start_time:.4f}s')
        return result
    return wrap
