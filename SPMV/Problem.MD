Implement a GPU program that performs sparse matrix-vector multiplication. Given a sparse matrix 
 of dimensions M*Nand a dense vector of length N compute the product vector  which will have length M 
. A is stored in row-major order. nnz is the number of non-zero elements in A.

Constraints
1 ≤ M, N ≤ 10,000   
The matrix 
 is approximately 60-70% sparse (i.e., 60-70% of elements are zero)
Performance is measured with M = 1,000, N = 10,000