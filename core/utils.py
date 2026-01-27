import time
import tracemalloc
import functools
import logging


def time_benchmark(func):
    """
    A decorator that prints the time a function takes to execute.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.info(f"[Time Benchmark] Function '{func.__name__}' took: {duration:.6f} seconds.")
        return result

    return wrapper


def memory_benchmark(func):
    """
    A decorator that prints the peak memory usage of a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        tracemalloc.clear_traces()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024
        logging.info(f"[Memory Benchmark] Function '{func.__name__}' peaked at: {peak_mb:.6f} MB")
        return result

    return wrapper
