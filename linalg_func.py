from decorators import timer, validate_matrix
import numpy as np

# TODO:
# Add class.method to check if a matrix is row echelon form
# add test cases for class.method
# check gaus_elim for errors
# add test cases for gaus_elim 

class LinearAlgebra():
    """
    Initialise the LinearAlgebra class to perform linear algebra operations
    """
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        pass

    def find_nonzero_index(matrix:np.ndarray) -> list:
        '''
        Finds the index of the first non-zero element in each row.
        '''
        rows_num = matrix.shape[0]
        indices = []
        # iterate through the row to find the index of the first non-zero element
        if matrix.ndim == 1:
            for i in range(matrix.size):
                if matrix[i] != 0:
                    indices.append(i)
                    break
            return indices
        elif matrix.ndim == 2:
            for row in range(rows_num):
                if np.all(matrix[row] == 0):
                    indices.append(-1)
                else:
                    for i in range(matrix[row].size):
                        if (matrix[row][i] != 0):
                            indices.append(i)
                            break
            return indices
        
    def find_zero_rows(matrix:np.ndarray) -> list:
        '''
        Find indices of all rows consisting entirely of zeros in a matrix.
        '''
        # create an empty list to store the indices of the zero rows
        zero_rows = []

        # iterate through the rows of the matrix to find the indices of the zero rows
        if matrix.ndim == 1:
            if np.all(matrix == 0):
                zero_rows.append(0)
            else:
                return zero_rows
        else:
            for i in range(matrix.shape[0]):
                if np.all(matrix[i] == 0):
                    zero_rows.append(i)

        return zero_rows

    def move_zerorows_bottom(matrix:np.ndarray) -> np.ndarray:
        '''
        Moves all zero rows to the bottom of the matrix.
        '''
        # find indices of all zero rows
        zerorow_indices = LinearAlgebra.find_zero_rows(matrix)
        
        # check if matrix is a 1darray with only zeros
        if (matrix.ndim == 1) and (zerorow_indices != []):
            return matrix

        # move all zero rows to the bottom of the matrix
        zerorow_arr = matrix[zerorow_indices]
        matrix = np.delete(matrix, obj=zerorow_indices, axis=0)
        matrix = np.append(matrix, zerorow_arr, axis=0)
        
        return matrix

    @validate_matrix
    def is_row_echelon(matrix:np.array) -> bool:
        '''
        Checks if a given matrix is in row echelon form.
        '''
        # Finds the index of the first non-zero element in each row.
        indices = LinearAlgebra.find_nonzero_index(matrix)

        # check if matrix of 2darray contains entirely of zeros
        if all(i == -1 for i in indices):
            return True

        # check each element in the indices list to see if they are in ascending order        
        for i in range(len(indices)-1):
            if indices[i] >= indices[i+1]:
                return False
        
        return True

    def is_zero_inrowofcol(matrix:np.ndarray, col:int) -> bool:
        '''
        Checks if the first row is a zero in a given column.
        '''
        if matrix[0, col] == 0:
            return True
        else:
            return False

    @timer
    @validate_matrix
    def gauss_elim(matrix:np.ndarray) -> np.ndarray: # answers might be different depending on how the matrix is arranged
        '''
        Inputs a matrix and performs Gaussian elimination on it.

        Returns a numpy array of the matrix in row echelon form.
        '''        
        while True:
            # create an empty array to store the completed rows
            completedrows = np.empty((0, matrix.shape[1]))
            
            # create an empty array to store the zero rows
            zerorows = np.empty((0, matrix.shape[1]))

            for col in range(matrix.shape[1]):
                # store zero rows in the zerorows array
                indices = LinearAlgebra.find_zero_rows(matrix)
                if len(indices) == 1:
                    zerorows = np.append(zerorows, matrix[indices, np.newaxis], 0)
                else:
                    zerorows = np.append(zerorows, matrix[indices], 0)

                # delete the zero rows from the matrix
                matrix = np.delete(matrix, indices, 0)

                # moving from the left-most to the right-most column, sort the rows from largest to smallest
                indices = np.argsort(matrix[:, col])
                matrix = matrix[indices[::-1]]
                
                if np.all(matrix[:, col] == 0):
                    continue
                else:
                    while True:

                        if np.all(matrix[:, col] == 0):
                            break

                        # find index of the first zero element in the column
                        zero_inrowofcol = LinearAlgebra.is_zero_inrowofcol(matrix, col)
                       
                        # move the row with zero in the column to the bottom
                        if zero_inrowofcol == True:
                            temp = matrix[0, np.newaxis]
                            matrix = np.delete(matrix, 0, 0)
                            matrix = np.append(matrix, temp, 0)
                        else:
                            break

                # divide the first row by 1/a where a is the first non-zero element in the first row to get a leading 1
                a = matrix[0, col]
                matrix[0] = matrix[0] / a
                # temporary variable to store the first row
                temp1 = matrix[0] 

                # check if there are any zeros below the leading 1
                first_nonzero_col = LinearAlgebra.find_nonzero_index(matrix[1:, col])

                if (0 in first_nonzero_col) or (not first_nonzero_col): # there is a nonzero element below the leading 1
                    row_count = matrix[:, col].size - 1 # number of rows to perform operations on to get zeroes below the leading 1
                    for i in range(row_count):
                        row = i + 1 # find index of the row to perform operations on
                        temp2 = - matrix[row, col] 
                        matrix[row] = matrix[row] + temp1 * temp2 # add multiple of the first row to the row to get a zero below the leading 1 
                    completedrows = np.append(completedrows, matrix[0, np.newaxis], 0) 
                    matrix = np.delete(matrix, 0, 0)
            matrix1 = np.append(completedrows, zerorows, 0)
            if LinearAlgebra.is_row_echelon(matrix1):
                break
        return matrix1
