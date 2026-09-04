2D Convolution
Medium
Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

The input consists of:

input: A 2D matrix of 32-bit floating-point numbers, represented as a 1D array in row-major order.
kernel: A 2D kernel (filter) of 32-bit floating-point numbers, also represented as a 1D array in row-major order.
The output should be written to the output matrix (also a 1D array in row-major order). The output matrix will have dimensions:

output_rows = input_rows - kernel_rows + 1
output_cols = input_cols - kernel_cols + 1
The convolution operation is defined as:


Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output

Constraints
1 ≤ input_rows, input_cols ≤ 3072
1 ≤ kernel_rows, kernel_cols ≤ 31
kernel_rows ≤ input_rows
kernel_cols ≤ input_cols
Performance is measured with input_cols = 3,072, input_rows = 3,072, kernel_cols = 15, kernel_rows = 15