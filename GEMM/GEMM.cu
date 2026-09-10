#include<bits/stdc++.h> 
#include<cuda.h> 
#include<cuda_runtime.h>
#include<cuda_fp16.h>

using namespace std; 

#define half __fp16



// simple matrix multiplication. can be improved using the blocked matrix multiplcation 
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, float alpha,float beta){


    int g_col = threadIdx.x + blockDim.x*blockIdx.x; // global column
    int g_row = threadIdx.y + blockDim.y*blockIdx.y; // global row 

    if( g_row >= M || g_col >= N ) return; 

    float sum = 0.0f; 
    for( int i = 0 ; i < K ; i++ ){
        sum += A[g_row*K + i] * B[i*N + g_col]; 
    }

    C[g_row*N + g_col ] = beta* C[g_row*N + g_col] + alpha*sum; 
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta)
{      
    dim3 TPB(16, 16); 
    dim3 Blocks((N + TPB.x - 1) / TPB.x, (M + TPB.y - 1) / TPB.y); 

    gemm_kernel<<<Blocks, TPB>>>(A, B, C, M, N, K, alpha, beta); 
    cudaDeviceSynchronize(); 
}



void assing_rand_val(float*A, size_t size ){
    int elems = size / sizeof(float); 

    for( int i = 0 ; i < elems; i++ ){
        A[i] = (float)rand()/ (float)RAND_MAX;
    }
}


void solve_cpu(const float* A, const float* B, float* C, int M, int N, int K, float alpha,float beta){
    for( int i = 0 ; i < M ; i++ ){
        for( int j = 0 ; j < N; j++ ){
            float sum = 0.0f; 
            for( int d = 0 ; d < K ; d++ ){
                sum += A[i*K + d] * B[d*N + j]; 
            }
            C[i*N + j ] = beta* C[i*N + j] + alpha*sum; 
        }
    }
}

int main(){
    int M = 231; 
    int N = 312; 
    int K = 103; 
    
    float alpha = 0.2f; 
    float beta = 0.1f; 

    size_t A_size = M*K*sizeof(float);
    size_t B_size = K*N*sizeof(float);
    size_t C_size = M*N*sizeof(float);

    // Host varaible declaration and memory allocation 
    float* h_A = (float*) malloc(A_size); 
    float* h_B = (float*) malloc(B_size); 
    float* h_C = (float*) malloc(C_size); 
    float* goldenTrace = (float*) malloc(C_size); 

    // Assign random value to the matrices; 
    assing_rand_val(h_A, A_size); 
    assing_rand_val(h_B, B_size); 
    assing_rand_val(h_C, C_size); 

    // Initialize goldenTrace with the same starting values as h_C
    memcpy(goldenTrace, h_C, C_size);

    // Device variables and memory allocation
    float* d_A, *d_B, *d_C; 
    cudaMalloc(&d_A,A_size); 
    cudaMalloc(&d_B,B_size); 
    cudaMalloc(&d_C,C_size); 

    // copy data from host to device 
    cudaMemcpy(d_A,h_A,A_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,B_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C,C_size,cudaMemcpyHostToDevice);

    // solve on Device 
    solve(d_A,d_B,d_C,M,N,K,alpha,beta);

    // copy results from DEVICE to HOST 
    cudaMemcpy(h_C, d_C, C_size,cudaMemcpyDeviceToHost); 

    // compute the CPU results; 
    solve_cpu(h_A, h_B, goldenTrace, M, N, K, alpha, beta); 

    // validate the GPU results 
    int flag = 0; 
    for( int i = 0 ; i < M ; i++ ){
        // if( flag == true){
        //     break; 
        // }
        for( int j = 0 ; j < N ; j++ ){
            if( fabs(goldenTrace[i*N+j] - h_C[i*N+j]) > 1e-5){
                cout << "Host[" << i << "][" << j <<"] = " << goldenTrace[i*N + j] << "  "; 
                cout << "Device[" << i << "][" << j <<"] = " << h_C[i*N + j]<< "\n"; 
                flag = true; 
                break; 
            }
        }
    }
    
    if( flag == true ) {
        cout << " Device computation Failed "; 
    }else{
        cout << " Results on DEVICE and HOST matched succesfull\n"; 
    }

    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C); 
    free(h_A); 
    free(h_B); 
    free(h_C); 
    free(goldenTrace); 



}