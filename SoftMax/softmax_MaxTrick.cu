#include <cuda.h> 
#include <cuda_runtime.h>
#include <bits/stdc++.h> 

using namespace std; 



__device__ float atomicFloatMax(float* address, float val) {
    int* address_as_int = (int*)address; 
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        
        // Convert the assumed int back to a float using pointer casting
        float assumed_float = *((float*)&assumed);
        
        // Calculate the maximum
        float max_val = fmaxf(val, assumed_float);
        
        // Convert the max float back to an int for atomicCAS
        int max_int = *((int*)&max_val);
        
        old = atomicCAS(address_as_int, assumed, max_int);
    } while (assumed != old);
    
    // Return the old value as a float
    return *((float*)&old);
}


__global__ void max_of_arr( const float* input, float* output, int N ){
    // compute locally update globally thing. I am plannig to do this by using the reduction kind of thing 
    extern __shared__ float sdata[]; 
    int tid = threadIdx.x; 
    int gid = threadIdx.x + blockDim.x * blockIdx.x; 

    if( gid < N ){
        sdata[tid] = input[gid]; 
    }else{
        // sdata[tid] = (float)INT_MIN; 
        sdata[tid] = -INFINITY;
    }

    // wait for all threads to load data in the shared memory 
    __syncthreads(); 

    for( int stride = blockDim.x/2; stride > 0 ; stride/=2){
        if( tid < stride){
            sdata[tid] = fmaxf(sdata[tid],sdata[tid+stride]); 
        }
        __syncthreads(); 
    }

    if( tid == 0 ){
        atomicFloatMax(output,sdata[0]); 
    }


}
__global__ void softmax_sum_kernel(const float* input, float* global_deno, const float* maxi, int N) {
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
        float e_xi = expf(input[gid]- *maxi); // FIXED: Changed int to float
        atomicAdd(&s_deno, e_xi); 
    }
    __syncthreads(); 
    
    // Thread 0 of each block adds the block's sum to the global sum

    if (tid == 0) {
        atomicAdd(global_deno, s_deno);
    }
}


__global__ void softmax_normalize_kernel(const float* input, float* output, const float* global_deno, const float *maxi,  int N) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x; 
    
    if (gid < N) {
        float e_xi = expf(input[gid] - *maxi);
        output[gid] = e_xi / (*global_deno); 
    }
}

void softmax_host(const vector<float>& input, vector<float>& output, float maxi ) {
    float denominator = 0.0f; 
    
    for (int i = 0; i < input.size(); i++) {
        denominator += expf(input[i] - maxi); 
    }

    for (int i = 0; i < input.size(); i++) {
        output[i] = expf(input[i] - maxi ) / denominator; 
    }
}

int main() {
    int N = 1024; 
    int TPB = 256; 
    int Blocks = (N+TPB-1)/TPB; 
    size_t shared_mem = TPB*sizeof(float);

    // Initialize the Host variables
    vector<float> h_arr(N); 
    vector<float> h_ans(N); 
    vector<float> golden_trace(N); 
    float maxi; 

    // Define device varaibles and allocate space for them 
    float* d_arr; 
    float* d_ans; 
    float* d_deno;
    float* d_maxi; 

    cudaMalloc(&d_arr, N * sizeof(float)); 
    cudaMalloc(&d_ans, N * sizeof(float)); 
    cudaMalloc(&d_deno, sizeof(float)); 
    cudaMalloc(&d_maxi, sizeof(float)); 

    // Initialize d_deno to 0 on the device & d_maxi = INT_MIN; 
    cudaMemset(d_deno, 0, sizeof(float));
    float init_max = -INFINITY;
    cudaMemcpy(d_maxi, &init_max, sizeof(float), cudaMemcpyHostToDevice); 

    // Initialize the host array with random values (0 to 1)
    for (int i = 0; i < N; i++) {
        h_arr[i] = (float)rand() / (float)RAND_MAX; 
    }
    // copy date from Host to Device 
    cudaMemcpy(d_arr, h_arr.data(), N * sizeof(float), cudaMemcpyHostToDevice); 

    // Lauch Pass 0 : maxi
    max_of_arr<<< Blocks, TPB,shared_mem>>>(d_arr, d_maxi, N); 
    cudaMemcpy(&maxi, d_maxi, sizeof(float), cudaMemcpyDeviceToHost); 

    // Golden Trace 
    softmax_host(h_arr, golden_trace, maxi); 

    // Launch Pass 1: Sum
    softmax_sum_kernel<<<Blocks, TPB >>>(d_arr, d_deno,d_maxi, N); 
    
    // Launch Pass 2: Normalize (Implicit global sync happens between these two calls)
    softmax_normalize_kernel<<<Blocks, TPB>>>(d_arr, d_ans, d_deno,d_maxi, N); 

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
    cudaFree(d_maxi); 

    return 0; 
}