#include<bits/stdc++.h> 
#include<cuda.h> 
#include<cuda_runtime.h>
using namespace std; 


__global__ void matrixMultKernel(const float *A, const float* x, float* y, int M , int N ){
    // int tid = threadIdx.x; 
    int row = threadIdx.x + blockDim.x*blockIdx.x; 

    float sum = 0.0f;
    for( int col = 0; col < N ; col++ ){
        sum+=A[row*N + col ] * x[col]; 
    }

    if( row < M ){
        y[row] = sum; 
    }

}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    // Isn't it just normal matrix multiplication 
    int TPB = 256; 
    int Blocks = (M+TPB-1)/TPB; 

    matrixMultKernel<<<Blocks, TPB>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

}


void solve_cpu(const float* A, const float* x, float* y, int M, int N, int) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            y[i] += A[i * N + j] * x[j];
        }
    }
}

void assign_rand_val(float*input, int N, int sparsity){
    for( int i = 0 ; i < N ; i++){

        if( (rand()%100) >= sparsity){
            input[i] = (float)rand()/ (float)RAND_MAX; 
        }else{
            input[i] = 0 ; 
        }
    }
}


int main(){
    int M = 1032; 
    int N = 2312; 
    size_t matrix_size = M*N*sizeof(float); 
    size_t x_size = N*sizeof(float); 
    size_t y_size = M*sizeof(float); 

    // Host variables
    float *h_A = (float*) malloc(matrix_size); 
    float *h_x = (float*) malloc(x_size); 
    float *h_y = (float*) malloc(y_size); 
    float *goldenTrace = (float*) malloc(y_size); 

    // assing random values to matrix based on sparsity  ; 
    assign_rand_val(h_A,M*N,70);
    assign_rand_val(h_x,N,0);

    // device variable decalration and memory allocatoin 
    float* d_A, *d_x, *d_y; 
    cudaMalloc(&d_A,matrix_size); 
    cudaMalloc(&d_x,x_size); 
    cudaMalloc(&d_y,y_size); 

    // copy data from HOST to DEVICE 
    cudaMemcpy(d_A,h_A,matrix_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,h_x,x_size,cudaMemcpyHostToDevice);

    // launch solve function 
    solve(d_A, d_x, d_y, M, N, 10);
    
    // copy results from device to Host 
    cudaMemcpy(h_y,d_y,y_size,cudaMemcpyDeviceToHost); 

    // golden trace calculation 
    solve_cpu(h_A,h_x,goldenTrace,M,N,10); 

    int flat = false; 
    for( int i = 0 ; i < M ; i++ ){
        if( fabs(goldenTrace[i]-h_y[i]) > 1e-5){
            flat = true; 
            break; 
        }
    }

    if( flat == true ){
        cout <<"Host and Device result don't match \n"; 
    }else{
        cout << "Host and Device results matched succesfully \n"; 
    }

    cudaFree(d_A); 
    cudaFree(d_x); 
    cudaFree(d_y); 
    free(h_A); 
    free(h_x); 
    free(h_y); 
    free(goldenTrace); 

    return 0; 
}