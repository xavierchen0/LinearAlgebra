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
    assert lg.LinearAlgebra.find_nonzero_index(np.zeros((5,5), dtype=np.float64)) == [-1] * 5

    # test if function returns the correct list of indices of the first non-zero element in each row for 1darray
    assert lg.LinearAlgebra.find_nonzero_index(np.array([0, 0, 1, 2, 3], dtype=np.float64)) == [2]

    # test if function returns the correct list of indices of the first non-zero element in each row for 2darray
    assert lg.LinearAlgebra.find_nonzero_index(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 3]], dtype=np.float64)) == [2, 3]

    # test if function returns the correct list of indices of the first non-zero element in each row for 2darray
    assert lg.LinearAlgebra.find_nonzero_index(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 0, 0],[0, 0, 0, 2, 3]], dtype=np.float64)) == [2, -1, 3]

def test_linalg_find_zero_rows():
    '''
    Test if function returns the correct list of indices of the zero rows.
    '''
    # test if function returns the correct list of indices of the zero rows for a 1darray with only zeros
    assert lg.LinearAlgebra.find_zero_rows(np.zeros((5), dtype=np.float64)) == [0]

    # test if function returns the correct list of indices of the zero rows for a 2darray with only zeros
    assert lg.LinearAlgebra.find_zero_rows(np.zeros((5,5), dtype=np.float64)) == [x for x in range(5)]

    # test if function returns an empty list for a 1darray with no zeros
    assert lg.LinearAlgebra.find_zero_rows(np.array([0, 0, 1, 2, 3], dtype=np.float64)) == []

    # test if function returns the correct list of indices of the zero rows for a 2darray that do not contain zero rows
    assert lg.LinearAlgebra.find_zero_rows(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 3]], dtype=np.float64)) == []

    # test if function returns the correct list of indices of the zero rows for a 2darray that cotains zero rows
    assert lg.LinearAlgebra.find_zero_rows(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.float64)) == [1,3]

def test_linalg_move_zerorows_bottom():
    '''
    Test if function moves all zero rows to the bottom of the matrix.
    '''
    # test if function returns the same matrix for a 1darray with only zeros
    assert np.allclose(lg.LinearAlgebra.move_zerorows_bottom(np.zeros((5), dtype=np.float64)), np.zeros((5), dtype=np.float64), rtol=1e-05, atol=1e-08)

    # test if function returns the same matrix for a 2darray with only zeros
    assert np.allclose(lg.LinearAlgebra.move_zerorows_bottom(np.zeros((5), dtype=np.float64)), np.zeros((5,5), dtype=np.float64), rtol=1e-05, atol=1e-08)

    # test if function returns the same matrix for a 1darray with no zeros
    assert np.allclose(lg.LinearAlgebra.move_zerorows_bottom(np.array([0, 0, 1, 2, 3], dtype=np.float64)), np.array([0, 0, 1, 2, 3], dtype=np.float64), rtol=1e-05, atol=1e-08)

    # test if function moves all zero rows to the bottom of the matrix for a 2darray
    assert np.allclose(lg.LinearAlgebra.move_zerorows_bottom(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.float64)), np.array([[0, 0, 1, 2, 3], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float64), rtol=1e-05, atol=1e-08)

    # test if function returns the original matrix for a 2darray that do not contain zero rows
    assert np.allclose(lg.LinearAlgebra.move_zerorows_bottom(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 3]], dtype=np.float64)), np.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 3]], dtype=np.float64), rtol=1e-05, atol=1e-08)

def test_linalg_is_row_echelon():
    '''
    Test if function checks if a given matrix is in row echelon form.
    '''
    # test if function returns True for a 1darray with only zeros
    assert lg.LinearAlgebra.is_row_echelon(np.zeros((5), dtype=np.float64)) == True

     # test if function returns True for a 2darray with only zeros
    assert lg.LinearAlgebra.is_row_echelon(np.zeros((5,5), dtype=np.float64)) == True

    # test if function returns True for a 1darray that is in row echelon form
    assert lg.LinearAlgebra.is_row_echelon(np.array([0, 0, 1, 2, 3], dtype=np.float64)) == True

    # test if function returns True for a 2darray that is in row echelon form
    assert lg.LinearAlgebra.is_row_echelon(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 3]], dtype=np.float64)) == True
    
    # test if function returns True for a 2darray that is in not row echelon form with zero rows in the middle
    assert lg.LinearAlgebra.is_row_echelon(np.array([[0, 0, 1, 2, 3], [0, 0, 0, 0, 0],[0, 0, 0, 2, 3]], dtype=np.float64)) == False

    # test if function returns True for a 2darray that is in not row echelon form
    assert lg.LinearAlgebra.is_row_echelon(np.array([[0, 0, 1, 2, 3], [0, 1, 2, 3, 0],[0, 0, 0, 0, 3]], dtype=np.float64)) == False

def test_find_zero_in_col():
    '''
    Test if function finds the index of the first zero element in a given column.
    '''
    # test for a matrix with a zero in the first column
    matrix1 = np.array([[0, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    assert lg.LinearAlgebra.is_zero_inrowofcol(matrix1, 0) == True

    # test for a atrix with a zero in the second column
    matrix2 = np.array([[1, 0, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    assert lg.LinearAlgebra.is_zero_inrowofcol(matrix2, 1) == True

    # test for a atrix with a zero in the first column
    matrix3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    assert lg.LinearAlgebra.is_zero_inrowofcol(matrix3, 0) == False

    # test for a matrix with a zero in the last position of the third column
    matrix4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=np.float64)
    assert lg.LinearAlgebra.is_zero_inrowofcol(matrix4, 2) == False

    # # test for a matrix with all zeros
    matrix5 = np.zeros((3, 3))
    assert lg.LinearAlgebra.is_zero_inrowofcol(matrix5, 0) == True

def test_gauss_elim():
# Test Case 1:
    matrix1 = np.array([[0, 0, -2, 0 , 7 , 12], [2, 4, -5, 6, -5, -1], [2, 4, -10, 6, 12, 28]], dtype=np.float64)
    assert np.array_equal(lg.LinearAlgebra.gauss_elim(matrix1), np.array([[1, 2, -5, 3, 6, 14], [0, 0, 1, 0, -3.4, -5.8], [0, 0, 0, 0, 1, 2]]))