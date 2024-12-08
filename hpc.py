import numpy as np
import time
#from joblib import Parallel, delayed

# Standard Matrix Multiplication
def standard_matrix_multiplication(A, B):
    n = A.shape[0]  # Size of the matrix
    C = np.zeros((n, n))  # Initialize the result matrix with zeros
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C
# Cache-Aware Matrix Multiplication with Parallelization and Optimized Block Size
def cache_aware_matrix_multiplication(A, B):
    n = A.shape[0]  # Size of the matrix
    block_size = 64  # Block size is chosen based on the cache line size
    C = np.zeros_like(A, dtype=np.int64)  # Ensure the result matrix is writable and of the correct type
    
    # Divide the matrices into smaller blocks and process each block
    for i in range(0, n, block_size):  # Iterate over row blocks
        for j in range(0, n, block_size):  # Iterate over column blocks
            for k in range(0, n, block_size):  # Iterate over shared dimension blocks
                # Process the elements within the current block
                for ii in range(i, min(i + block_size, n)):
                    for jj in range(j, min(j + block_size, n)):
                        for kk in range(k, min(k + block_size, n)):
                            # Multiply and accumulate values for each block element in C
                            C[ii, jj] += A[ii, kk] * B[kk, jj]
    return C


# Testing the performance of both implementations
n = 1024  # Size of the matrices (adjust size for testing)
A = np.random.randint(1, 100, size=(n, n))  # Generate a random matrix A
B = np.random.randint(1, 100, size=(n, n))  # Generate a random matrix B

# Measure time for the standard approach
start_time = time.time()
standard_matrix_multiplication(A, B)
standard_duration = time.time() - start_time

# Measure time for the cache-aware approach
start_time = time.time()
cache_aware_matrix_multiplication(A, B)
cache_aware_duration = time.time() - start_time

# Print the execution times
print(f"Standard Duration: {standard_duration:.2f} seconds")
print(f"Cache-Aware Duration: {cache_aware_duration:.2f} seconds")