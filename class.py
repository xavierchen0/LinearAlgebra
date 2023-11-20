import time
import functools
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

def check_input(func):
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

class LinearAlgebra():
    """
    Initialise the LinearAlgebra class to perform linear algebra operations
    """
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        pass

    @check_input
    def test(self, matrix):
        return matrix

    @timer
    def gauss_elim(self, data: np.ndarray) -> np.ndarray:
        '''
        Inputs a numpy array (maximum 2darray) and performs Gaussian elimination on it.

        Returns a numpy array of the matrix in row echelon form.
        '''
        # check if data is a numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        
        # check if data is a 2darray
        if data.ndim != 2:
            raise ValueError("Input data must be a 2darray")
        
        # check if data type of the numpy arrray is float64 or int64
        if data.dtype != np.float64:
            raise TypeError("Input data must be of type float64")
        
        # check if data is not an empty numpy array
        if data.size == 0:
            raise ValueError("Input data must not be an empty numpy array")

        # function to find the index of the first non-zero element in a row
        def find_first_nonzero_row(row:np.ndarray) -> int or np.nan:
            '''
            Finds the index of the first non-zero element in a row.
            '''
            # iterate through the row to find the index of the first non-zero element
            for i in range(row.size):
                if row[i] != 0:
                    return i
            return np.nan
        
        # function to find the index of the first non-zero element in a col
        def find_first_zero_col(col:np.ndarray) -> int or False:
            '''
            Finds the index of the first non-zero element in a column.
            '''
            # iterate through the col to find the index of the first zero element starting from the top
            for i in range(col.size):
                if col[i] == 0:
                    return i
            return False
                
        # function to check if matrix is in row echelon form
        def is_row_echelon_form(data:np.ndarray) -> bool:
            '''
            Check if the matrix is in row echelon form.
            '''
            # iterate through the rows of the matrix to check if the first non-zero element of the lower row 
            # is to the right of the first non-zero element of the row above it
            for i in range(data.shape[0] - 1):
                upper_row = data[i]
                lower_row = data[i+1]
                i_upper = find_first_nonzero_row(upper_row) # i_upper is the index of the first non-zero element in the upper row
                i_lower = find_first_nonzero_row(upper_row) # i_lower is the index of the first non-zero element in the lower row
                if i_upper >= i_lower:
                    return False
            return True

        # function to find the indices of zero rows
        def find_zero_rows(data:np.ndarray) -> list:
            '''
            Find indices of all zero rows in a matrix.
            '''
            # create an empty list to store the indices of the zero rows
            zero_rows = []

            # iterate through the rows of the matrix to find the indices of the zero rows
            for i in range(data.shape[0]):
                if np.all(data[i] == 0):
                    zero_rows.append(i)

            return zero_rows 

        # perform gaussian elimination
        
        while not is_row_echelon_form(data):
            # create an empty array to store the completed rows
            completedrows = np.empty((0, data.shape[1]))
        
            # create an empty array to store the zero rows
            zerorows = np.empty((0, data.shape[1]))

            for col in range(data.shape[1]):
                # store zero rows in the zerorows array
                indices = find_zero_rows(data)
                zerorows = np.append(zerorows, data[indices], 0)

                # delete the zero rows from the matrix
                data = np.delete(data, indices, 0)
                # moving from the left-most to the right-most column, sort the rows from largest to smallest
                indices = np.argsort(data[:, col])
                data = data[indices[::-1]]
                # divide the first row by 1/a where a is the first non-zero element in the first row to get a leading 1
                a = data[0, col]
                data[0] = data[0] / a
                # temporary variable to store the first row
                temp1 = data[0]

                # check if there are any zeros below the leading 1
                first_zero_col = find_first_zero_col(data[:, col])

                if first_zero_col == 1: # there are only zeros below the leading 1
                    continue
                elif first_zero_col == False: # there are no zeros below the leading 1
                    row_count = data[:, col].size - 1 # number of rows to perform operations on to get zeroes below the leading 1
                    for i in range(row_count):
                        row = i + 1 # find index of the row to perform operations on
                        temp2 = - data[row,col] 
                        data[row] = data[row] + temp1 * temp2 # add multiple of the first row to the row to get a zero below the leading 1 
                    completedrows = np.append(completedrows, data[0, np.newaxis], 0) 
                    data = np.delete(data, 0, 0)
                else:
                    row_count = first_zero_col # number of rows to perform operations on to get zeroes below the leading 1
                    for i in range(row_count):
                        row = i + 1 # find index of the row to perform operations on
                        temp2 = - data[row,col] 
                        data[row] = data[row] + temp1 * temp2 # add multiple of the first row to the row to get a zero below the leading 1 
                    completedrows = np.append(completedrows, data[0, np.newaxis], 0) 
                    data = np.delete(data, 0, 0)
        data1 = np.append(completedrows, zerorows, 0)
        return data1

testarray = np.array(
    [[1,1,2,9],[2,4,3,1],[3,6,5,0]],                 
    dtype=np.int64
)

testarray1 = 1
print(f'this is the orginal array:\n{testarray}')

array = LinearAlgebra().test(testarray)
