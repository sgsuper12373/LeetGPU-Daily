#include <cuda.h> 
#include <cuda_runtime.h>
#include <bits/stdc++.h> 

using namespace std; 


__global__ void softmax_sum_kernel(const float* input, float* global_deno, int N) {
    __shared__ float s_deno; 
    
    // Initialize shared memory ( compute locally update globally approach)
    int gid = threadIdx.x + blockIdx.x * blockDim.x; 
    int tid = threadIdx.x; 

    if (tid == 0) {
        s_deno = 0.0f;
    }
    __syncthreads(); 

    
    // Calculate local expf and add to shared memory safely
    if (gid < N) {
        float e_xi = expf(input[gid]); // FIXED: Changed int to float
        atomicAdd(&s_deno, e_xi); 
    }
    __syncthreads(); 
    
    // Thread 0 of each block adds the block's sum to the global sum

    if (tid == 0) {
        atomicAdd(global_deno, s_deno);
    }
}


__global__ void softmax_normalize_kernel(const float* input, float* output, const float* global_deno, int N) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x; 
    
    if (gid < N) {
        float e_xi = expf(input[gid]);
        output[gid] = e_xi / (*global_deno); 
    }
}

void softmax_host(const vector<float>& input, vector<float>& output) {
    float denominator = 0.0f; 
    
    for (int i = 0; i < input.size(); i++) {
        denominator += expf(input[i]); 
    }

    for (int i = 0; i < input.size(); i++) {
        output[i] = expf(input[i]) / denominator; 
    }
}

int main() {
    int N = 1024; 
    int TPB = 256; 
    int Blocks = (N+TPB-1)/TPB; 

    // Initialize the Host variables
    vector<float> h_arr(N); 
    vector<float> h_ans(N); 
    vector<float> golden_trace(N); 

    // Define device varaibles and allocate space for them 
    float* d_arr; 
    float* d_ans; 
    float* d_deno;

    cudaMalloc(&d_arr, N * sizeof(float)); 
    cudaMalloc(&d_ans, N * sizeof(float)); 
    cudaMalloc(&d_deno, sizeof(float)); 

    // Initialize d_deno to 0 on the device
    cudaMemset(d_deno, 0, sizeof(float));

    // Initialize the host array with random values (0 to 1)
    for (int i = 0; i < N; i++) {
        h_arr[i] = (float)rand() / (float)RAND_MAX; 
    }

    softmax_host(h_arr, golden_trace); 

    // copy date from Host to Device 
    cudaMemcpy(d_arr, h_arr.data(), N * sizeof(float), cudaMemcpyHostToDevice); 

    // Launch Pass 1: Sum
    softmax_sum_kernel<<<Blocks, TPB>>>(d_arr, d_deno, N); 
    
    // Launch Pass 2: Normalize (Implicit global sync happens between these two calls)
    softmax_normalize_kernel<<<Blocks, TPB>>>(d_arr, d_ans, d_deno, N); 

    // Copy values from device to host 
    cudaMemcpy(h_ans.data(), d_ans, N * sizeof(float), cudaMemcpyDeviceToHost); 

    // Validate the results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (abs(h_ans[i] - golden_trace[i]) > 1e-5) {
            cout << "Error at index " << i << ": Expected " << golden_trace[i] << " got " << h_ans[i] << "\n";
            success = false;
            break;
        }
    }
    
    if (success) cout << "Softmax computed successfully!\n";

    // Clean up
    cudaFree(d_arr);
    cudaFree(d_ans);
    cudaFree(d_deno);

    return 0; 
}