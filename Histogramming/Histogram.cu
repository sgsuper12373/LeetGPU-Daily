
#include<bits/stdc++.h>
#include<cuda.h>
#include<cuda_runtime.h>

using namespace std; 


__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins){
    

    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    extern __shared__ int shared_hist[]; 

    // We use a loop in case num_bins is larger than blockDim.x
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    
    // Wait for all threads to finish initializing shared memory
    __syncthreads(); 

    // Compute the local histogram in fast shared memory
    if (tid < N) {
        int bin = input[tid];
        // Ensure the data doesn't exceed our allocated bins
        if (bin >= 0 && bin < num_bins) { 
            atomicAdd(&shared_hist[bin], 1); 
        }
    }

    // Wait for all threads in the block to finish their local counting
    __syncthreads(); 

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (shared_hist[i] > 0) { // Optimization: only write if non-zero
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

extern  void solve(const int* input, int* histogram, int N, int num_bins) {
    int TPB = 512; 
    int Blocks = (N + TPB - 1) / TPB; 

    size_t shared_mem_size = num_bins * sizeof(int);

    histogram_kernel<<<Blocks, TPB, shared_mem_size>>>(input, histogram, N, num_bins);
    
    cudaDeviceSynchronize(); 
}

void solve_cpu(const int* input, int* histogram, int N, int num_bins){
    for( int i = 0 ; i < N ; i++ ){
        int bin = input[i]; 
        histogram[bin]+=1; 
    }
}

void assign_rand_val( int* input, int N , int maxi ){
    for( int i = 0 ; i < N ; i++ ){
        input[i] = rand()%maxi; 
    }
}

int main(){
    int N = 12312; 
    int num_bins = 1231; 

    //Host varaibles; 
    int *h_input = (int*)malloc(N*sizeof(int)); 
    int *h_histogram = (int*) malloc(num_bins*sizeof(int)); 
    int *golden_trace = (int*) malloc(num_bins*sizeof(int)); 

    // initialize them with some values 
    assign_rand_val( h_input, N,num_bins ); 
    memset(h_histogram, 0, num_bins*sizeof(int));
    memset(golden_trace, 0, num_bins*sizeof(int));

    // Device varaible decalration, memory allocation and initilizatin 
    int *d_input, *d_histogram; 
    cudaMalloc(&d_input, N*sizeof(int)); 
    cudaMalloc(&d_histogram, num_bins*sizeof(int)); 
    cudaMemset(d_histogram,0,num_bins*sizeof(int));
    cudaMemcpy(d_input,h_input,N*sizeof(int),cudaMemcpyHostToDevice);

    // solve using GPU and copy back the results
    solve(d_input,d_histogram,N,num_bins); 
    cudaMemcpy(h_histogram,d_histogram,num_bins*sizeof(int),cudaMemcpyDeviceToHost); 


    // get the golden trace 
    solve_cpu(h_input,golden_trace,N,num_bins);

    // compare results 
    bool isSucccess = 1; 
    for( int i = 0 ; i < num_bins; i++ ){
        if( abs(golden_trace[i] - h_histogram[i]) > 1e-5){
            isSucccess = false; 
            break; 
        }
    }

    if(isSucccess){
        cout<< " Results of Host and Device matched successfully\n"; 
    }else{
        cout << " Maximum error threshold excceded \n"; 
    }

    cudaFree(d_input);
    cudaFree(d_histogram);
    free(h_input);   
    free(h_histogram); 
    free(golden_trace); 

    return 0; 
    
}