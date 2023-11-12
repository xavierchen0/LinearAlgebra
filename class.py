import time

# create a decorator to time functions
def timer(func):
    """
    Create a decorator to time functions
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Time taken to run {func.__name__}: {end - start}")
        return value
    return wrapper

class LinearAlgebra():
    """
    Initialise the LinearAlgebra class to perform linear algebra operations
    """