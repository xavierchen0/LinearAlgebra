import pytest
from decorators import validate_matrix
import numpy as np
import linalg_func as lg

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

def test_linalg_find_nonzero_index():
    '''
    Test if function returns the correct list of indices of the first non-zero element in each row.
    '''
    # test if function returns the correct list of indices of the first non-zero element in each row for a 1darray with only zeros
    assert lg.LinearAlgebra.find_nonzero_index(np.zeros((1), dtype=np.float64)) == []

    # test if function returns the correct list of indices of the first non-zero element in each row for a 2darray with only zeros
    assert lg.LinearAlgebra.find_nonzero_index(np.zeros((5,5), dtype=np.float64)) == []

    # test if function returns the correct list of indices of the first non-zero element in each row for 1darray
    assert lg.LinearAlgebra.find_nonzero_index(np.array([0, 0, 1, 2, 3], dtype=np.float64)) == [2]

    # test if function returns the correct list of indices of the first non-zero element in each row for 2darray
    assert lg.LinearAlgebra.find_nonzero_index(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 3]], dtype=np.float64)) == [2, 3]

def test_linalg_find_zero_rows():
    '''
    Test if function returns the correct list of indices of the zero rows.
    '''
    # test if function returns the correct list of indices of the zero rows for a 1darray with only zeros
    assert lg.LinearAlgebra.find_zero_rows(np.zeros((5), dtype=np.float64)) == [0]

    # test if function returns the correct list of indices of the zero rows for a 2darray with only zeros
    assert lg.LinearAlgebra.find_zero_rows(np.zeros((5,5), dtype=np.float64)) == [x for x in range(5)]

    # test if function returns an empty list a 1darray with no zeros
    assert lg.LinearAlgebra.find_zero_rows(np.array([0, 0, 1, 2, 3], dtype=np.float64)) == []

    # test if function returns the correct list of indices of the zero rows for a 2darray
    assert lg.LinearAlgebra.find_zero_rows(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 3]], dtype=np.float64)) == []

    # test if function returns the correct list of indices of the zero rows for a 2darray
    assert lg.LinearAlgebra.find_zero_rows(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.float64)) == [1,3]