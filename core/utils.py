import functools
import logging
import time
import tracemalloc


def time_benchmark(func):
    """Decorator that logs function execution time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logging.info(f"[Time] {func.__name__}: {duration:.6f}s")
        return result

    return wrapper


def memory_benchmark(func):
    """Decorator that logs peak memory usage."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        tracemalloc.clear_traces()
        result = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logging.info(f"[Memory] {func.__name__}: {peak / 1024 / 1024:.6f} MB")
        return result

    return wrapper
