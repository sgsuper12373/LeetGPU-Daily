#include<cuda.h> 
#include<cuda_runtime.h> 
#include<bits/stdc++.h> 

using namespace std; 

/**
 * @brief Compute locally the dot prodcut of vector falling in block. one thread of block updates the global sum
 * 
 * @param A 
 * @param B 
 * @param result 
 * @param N 
 * @return __global__ 
 */
__global__ void dot_product_kernel( const float* A, const float* B, float*result, int N ){

    // Shared memory to store the local sum 
    __shared__ float block_sum; 

    // Initialize the shared memory
    if( threadIdx.x == 0){
        block_sum = 0.0f; 
    }

    // sync all threads so all threads see the consistent value 
    __syncthreads(); 


    // calculate global threadId and exit if exceeds the bounds 
    int gid = threadIdx.x + blockDim.x*blockIdx.x; 
    if( gid >= N ) return; 

    // update the local sum 
    atomicAdd(&block_sum,A[gid]*B[gid]); 

    // Wait for all threads to finish atomic add 
    __syncthreads(); 

    // thread 0 updates the global value
    if( threadIdx.x == 0 ){
        atomicAdd(result,block_sum); 
    }


}


/**
 * @brief Launch dot_product kernel with TPB and Blocks computatoin
 * 
 */
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int TPB = 256; 
    int Blocks = (N+TPB-1)/TPB; 
    dot_product_kernel<<<Blocks,TPB>>>(A,B,result,N); 
    cudaDeviceSynchronize(); 
}


/**
 * @brief Assings random values to the input array on host 
 * 
 * @param input 
 * @param N 
 */
void assign_rand_val(float* input, int N ){
    for( int i = 0 ; i < N ; i++ ){
        input[i] = (float)rand()/(float)RAND_MAX; 
    }
}

/**
 * @brief dot product on HOST using for loop
 * 
 * @param A 
 * @param B 
 * @param res 
 * @param N 
 */
void solve_cpu(float*A, float*B, float*res, int N ){
    *res = 0.0f;
    for( int i = 0 ; i < N ; i++ ){
            *res += A[i]*B[i]; 
    }
}

int main(){
    int N = 1028; 
    size_t Arr_size = N*sizeof(float); 

    // Host varaibles declaration and allocation 
    float* h_A = (float*)malloc(Arr_size);
    float* h_B = (float*)malloc(Arr_size);
    float* h_result = (float*)malloc(sizeof(float)); 
    float* goldenTrace = (float*)malloc(sizeof(float)); 

    // Iniitalize h_A and h_B with random values 
    assign_rand_val(h_A,N);
    assign_rand_val(h_B,N);

    // device varaibles decalartion and memory allocation 
    float* d_A, *d_B, *d_result; 
    cudaMalloc(&d_A,Arr_size); 
    cudaMalloc(&d_B,Arr_size); 
    cudaMalloc(&d_result,sizeof(float)); 

    //set devcie result to 0; 
    cudaMemset(d_result,0,sizeof(float)); 
    

    // copy host data to device 
    cudaMemcpy(d_A, h_A, Arr_size,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, h_B, Arr_size,cudaMemcpyHostToDevice); 

    // compute result on device and copy back the result 
    solve(d_A, d_B, d_result,N); 
    cudaMemcpy(h_result,d_result,sizeof(float),cudaMemcpyDeviceToHost);


    // Get the goldenTrace
    solve_cpu(h_A,h_B,goldenTrace,N); 

    // validate the results 
    bool flag = true; 
    for( int i = 0 ; i < N ; i++ ){
        if( fabs(goldenTrace[i] - h_result[i]) > 1e-6){
            flag = false; 
            break; 
        }
    }

    if( flag ){
        cout << "Device Results matchess Host\n" ; 
    }else{
        cout << "Device and host resutl does not match \n"; 
    }

    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_result); 
    free(h_A); 
    free(h_B); 
    free(h_result);
    free(goldenTrace); 


}