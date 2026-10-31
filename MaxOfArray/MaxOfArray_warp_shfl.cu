
#include <iostream>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// Kernel to find the maximum value in an array using warp shuffle primitives
__global__ void max_find_kernel(const int* arr, int* d_maxi, int N) {
    __shared__ int block_max;
    
    int tid = threadIdx.x; 
    int gid = threadIdx.x + blockIdx.x * blockDim.x; 
    int laneId = tid % 32;

    // Thread 0 initializes the block's shared maximum
    if (tid == 0) {
        block_max = INT_MIN;
    }
    __syncthreads();

    // Load data from global memory into register, padding out-of-bound elements with INT_MIN
    int val = (gid < N) ? arr[gid] : INT_MIN;

    // Warp-level reduction to find maximum value within each warp
    for (int offset = 16; offset > 0; offset /= 2) {
        int neighbor = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, neighbor);
    }

    // Lane 0 of each warp updates the block-level maximum
    if (laneId == 0) {
        atomicMax(&block_max, val);
    }
    __syncthreads();

    // Thread 0 of each block updates the global maximum
    if (tid == 0) {
        atomicMax(d_maxi, block_max);
    }
}

// Host launcher function
__host__ void get_max(const int* d_arr, int* d_maxi, int N) {
    int TPB = 256; 
    int Blocks = (N + TPB - 1) / TPB; // Ceiling division to cover all N elements

    max_find_kernel<<<Blocks, TPB>>>(d_arr, d_maxi, N); 
    cudaDeviceSynchronize();
}

// Helper to assign random values to the array
__host__ void assign_random_val(int* arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 10000;
    }
}

int main() {
    int N = 1034; 
    
    // Allocate host memory
    int* h_arr = (int*)malloc(N * sizeof(int)); 
    int h_maxi; 
    int golden_value = INT_MIN; 

    // Seed random and initialize array
    srand(42);
    assign_random_val(h_arr, N); 

    // Calculate golden maximum on host
    for (int i = 0; i < N; i++) {
        golden_value = max(golden_value, h_arr[i]); 
    }

    // Allocate and copy device array memory
    int* d_arr;
    cudaMalloc(&d_arr, N * sizeof(int)); 
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice); 

    // Allocate device memory for global max and initialize with INT_MIN
    int* d_maxi; 
    cudaMalloc(&d_maxi, sizeof(int)); 
    int init_val = INT_MIN;
    cudaMemcpy(d_maxi, &init_val, sizeof(int), cudaMemcpyHostToDevice); 

    // Run GPU kernel
    get_max(d_arr, d_maxi, N); 

    // Copy result back to host
    cudaMemcpy(&h_maxi, d_maxi, sizeof(int), cudaMemcpyDeviceToHost); 

    // Verification output
    cout << "HOST (CPU) MAX  : " << golden_value << "\n"; 
    cout << "DEVICE (GPU) MAX: " << h_maxi << "\n"; 

    if (golden_value == h_maxi) {
        cout << "SUCCESS: Results Match!" << endl;
    } else {
        cout << "FAILURE: Results Do Not Match!" << endl;
    }

    // Free memory
    free(h_arr);
    cudaFree(d_arr);
    cudaFree(d_maxi);

    return 0;
}