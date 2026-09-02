#include<cuda.h> 
#include<cuda_runtime.h>
#include<bits/stdc++.h> 

using namespace std; 

__device__ float deno = 0.0f;
__global__ void softmax_kernel( const float* input, float* output, int N ){

    __shared__ float s_deno ; 
    int tid = threadIdx.x; 
    int gid = threadIdx.x + blockIdx.x*blockDim.x; 

    // first calculate the denominator for each of sigma(xi); 
    int e_xi = expf(input[gid]); 
    atomicAdd(&s_deno, e_xi); 
    // Wait for  all threads to write the value to the 
    __syncthreads(); 
    if( tid == 0 ){
        atomicAdd(&deno,s_deno);
    }

    // here need to sync the threads across all the blocks but this is not possible, logical mistake baby 
    
    // get the value of sigma(xi);
    output[gid] = e_xi/s_deno; 
}

void softmax_host( const vector<float>& input, vector<float>& output){
    float denominator = 0.0f; 
    
    for( int i = 0 ; i < input.size(); i++ ){
        denominator += expf(input[i]); 
    }

    for( int i = 0; i < input.size(); i++ ){
        output[i] = expf(input[i])/denominator; 
    }
}






int main(){
    int N = 1024; 
    int TPB = 256; 
    int Blocks = (N+TPB-1)/TPB; 

    // Initialize the Host varaibles
    vector<float> h_arr(N); 
    vector<float> h_ans(N); 
    vector<float> golden_trace(N); 

    // Define device varaibles and allocate space for them 
    float* d_arr; 
    float* d_ans; 
    cudaMalloc(&d_arr, N*sizeof(float)); 
    cudaMalloc(&d_ans, N*sizeof(float)); 

    // Initilize the host array with some random values 
    for( int i = 0 ; i < N ; i++ ){
        h_arr[i] = (float)rand()/RAND_MAX; 
    }

    softmax_host( h_arr, golden_trace); 

    // Copy data from Host to Device 
    cudaMemcpy(d_arr,h_arr.data(),N*sizeof(int),cudaMemcpyHostToDevice); 

    // Launch the kernel 
    softmax_kernel<<< Blocks, TPB>>>(d_arr,d_ans,N); 

    // copy values from device to hoset 
    cudaMemcpy(h_ans.data(),d_ans,N*sizeof(float),cudaMemcpyDeviceToHost); 

    // Validate the resulta 
    for( int i = 0; i < N ; i++ ){
        if( abs(h_ans[i] - golden_trace[i]) > 1e-5){
            cout << "Error excceded the maximum limit"; 
        }
    }




    return 0; 
}