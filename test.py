import pytest
from decorators import validate_matrix
import numpy as np

def test_validate_matrix_errors():
    '''
    Test if respective errors are raised.
    '''
    @validate_matrix
    def test_func(matrix):
        return matrix

    # test if function raises TypeError if matrix is not a numpy array
    with pytest.raises(TypeError):
        test_func([1, 2, 3])

    # test if function raises TypeError if matrix is not of type float64
    with pytest.raises(TypeError):
        test_func(np.array([1, 2, 3], dtype=np.int64))

    # test if function raises ValueError if matrix is a 0darray
    with pytest.raises(ValueError):
        test_func(np.array(1, dtype=np.float64))

    # test if function raises ValueError if matrix is a 3darray
    with pytest.raises(ValueError):
        test_func(np.ones((1,1,1)))

    # test if function raises ValueError if matrix is an empty numpy array
    with pytest.raises(ValueError):
        test_func(np.array([[]]))

def test_validate_matrix_args():
    '''
    Test if arguments are passed correctly.
    '''
    @validate_matrix
    def test_func(matrix):
        return matrix
    
    # test if function returns the matrix if matrix is given as a positional argument
    assert test_func(np.ones((1,1), dtype=np.float64)) == np.ones((1,1), dtype=np.float64)

    # test if function returns the matrix if matrix is given as a keyword argument
    assert test_func(matrix=np.ones((1,1), dtype=np.float64)) == np.ones((1,1), dtype=np.float64)