#include <iostream> 
#include <vector>
#include <cstdlib>
#include <cuda.h> 
#include <cuda_runtime.h>

using namespace std; 

__global__ void Reduce_normal(const float *input, float *output, const int N)
{   
    extern __shared__ float sdata[]; 
    int tid = threadIdx.x; 
    int gid = threadIdx.x + blockDim.x * blockIdx.x; 
    
    // 1. Load data into shared memory using GLOBAL ID (gid) bounds check
    if (gid < N) {
        sdata[tid] = input[gid]; 
    } else {
        sdata[tid] = 0.0f; 
    }

    // 2. CRITICAL: Wait for all threads to finish loading
    __syncthreads(); 

    // 3. Stride loop corrected
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride]; 
        }
        __syncthreads(); 
    }

    // 4. Thread 0 safely adds block total to global total
    if (tid == 0) {
        atomicAdd(output, sdata[0]); 
    }
}
int main(int argc, char** argv) {

    int N = 100000; // Default array size
    int TPB = 256;  // Default threads per block
    
    if (argc >= 3) {
        N = atoi(argv[1]); 
        TPB = atoi(argv[2]); 
    }

    // HOST variables
    vector<float> h_array(N); 
    float h_sum = 0.0f; 
    float golden_sum;
    
    // Devcie pointers 
    float *d_array; 
    float *d_sum; 

    
    for (int i = 0; i < N; i++) {
        h_array[i] = (float)rand() / (float)RAND_MAX;
        golden_sum += h_array[i]; 
    }

    // Allocate memory on devce
    cudaMalloc(&d_array, N * sizeof(float)); 
    cudaMalloc(&d_sum, sizeof(float)); 

    // set the sum as 0 ;
    cudaMemset(d_sum, 0, sizeof(float));

    // copy host array 
    cudaMemcpy(d_array, h_array.data(), N * sizeof(float), cudaMemcpyHostToDevice); 

    // shared memory and number of blokcs 
    size_t shrd_mem = TPB * sizeof(float); 
    int blocks = (N + TPB - 1) / TPB;

    // call the kernel 
    Reduce_normal<<<blocks, TPB, shrd_mem>>>(d_array, d_sum, N); 

    // copy back the results 
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost); 
    
    cout << "Sum of array by GPU is: " << h_sum << "\n"; 
    cout << "Sum of array by CPU is: " << golden_sum  << "\n"; 

    cudaFree(d_array);
    cudaFree(d_sum);

    return 0; 
}