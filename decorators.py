import functools
import time
import numpy as np

def timer(func):
    """
    A decorator to time functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Time taken to run {func.__name__}: {end - start}")
        return value
    return wrapper

def validate_matrix(func):
    '''
    A decorator to check if the input matrix is valid.
    '''
    @functools.wraps(func)
    def wrapper_check_input(*args, **kwargs):
        matrix = args[0]
        matrix1 = kwargs.get('matrix')
        def check(mat):
            # check if matrix is a numpy array
            if not isinstance(mat, np.ndarray):
                raise TypeError("Matrix must be a numpy array")
            else:
                # check if data type of the numpy arrray is float64 or int64
                if mat.dtype != np.float64:
                    raise TypeError("Matrix must be of type float64")
                else:
                    # check if matrix is a 1darray or 2darray
                    if mat.ndim <= 2:
                        raise ValueError("Matrix must be a 1darray or 2darray")
                    else:
                        # check if matrix is not an empty numpy array
                        if mat.size == 0:
                            raise ValueError("Matrix must not be an empty numpy array")

        if matrix1 is not None:
            matrix = matrix1
            check(matrix)
        else:
            check(matrix)

    return wrapper_check_input