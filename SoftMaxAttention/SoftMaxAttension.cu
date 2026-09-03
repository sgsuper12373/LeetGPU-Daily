#include<cuda.h> 
#include<cuda_runtime.h> 
#include<bits/stdc++.h> 

using namespace std; 


__global__ void matrix_mult_kernel(const float* A, const float* B, float* output, int M, int N, int K, float scale) {
    int col = threadIdx.x + blockIdx.x * blockDim.x; 
    int row = threadIdx.y + blockIdx.y * blockDim.y; 

    if (col >= N || row >= M) return; 

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += (A[row * K + i] * B[i * N + col]); 
    }
    output[row * N + col] = sum / scale; 
}

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    int col = threadIdx.x + blockDim.x * blockIdx.x; 
    int row = threadIdx.y + blockDim.y * blockIdx.y; 

    if (col >= N || row >= M) return; 

    B[col * M + row] = A[row * N + col]; 
}

// New kernel to apply softmax row-wise
__global__ void softmax_kernel(float* mat, int M, int N) {
    // One thread per row for simplicity 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Find the maximum value in the row for numerical stability
    float max_val = mat[row * N];
    for (int i = 1; i < N; i++) {
        max_val = fmaxf(max_val, mat[row * N + i]);
    }

    // Compute exponents and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = expf(mat[row * N + i] - max_val);
        mat[row * N + i] = e;
        sum_exp += e;
    }

    // Normalize the row
    for (int i = 0; i < N; i++) {
        mat[row * N + i] /= sum_exp;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    dim3 TPB(16, 16);
    dim3 transposeBlocks((d + 15) / 16, (N + 15) / 16);
    dim3 qktBlocks((N + 15) / 16, (M + 15) / 16);
    dim3 outputBlocks((d + 15) / 16, (M + 15) / 16);

    float* d_KT;
    float* temp_res;

    cudaMalloc(&d_KT, N * d * sizeof(float));
    cudaMalloc(&temp_res, M * N * sizeof(float));

    // 1. K Transpose
    matrix_transpose_kernel<<<transposeBlocks, TPB>>>(K, d_KT, N, d);

    // 2. Q * K^T / sqrt(d)
    matrix_mult_kernel<<<qktBlocks, TPB>>>(Q, d_KT, temp_res, M, N, d, sqrtf(d));

    // 3. Apply Softmax row-wise to temp_res (M rows)
    int softmax_threads = 256;
    int softmax_blocks = (M + softmax_threads - 1) / softmax_threads;
    softmax_kernel<<<softmax_blocks, softmax_threads>>>(temp_res, M, N);

    // 4. (Softmax Result) * V
    matrix_mult_kernel<<<outputBlocks, TPB>>>(temp_res, V, output, M, d, N, 1.0f);

    cudaFree(temp_res); 
    cudaFree(d_KT); 
}


void assign_rand_val(float* A, int M, int N){
    for( int i = 0 ; i < M; i++ ){
        for( int j = 0; j < N; j++ ){
            A[i*N+j]= (float)rand()/(float)RAND_MAX; 
        }
    }
}


void matrix_tranpose_cpu(float* A, float* B, int M, int N ){
    // A = M*N 
    // B = N*M
    for( int i = 0 ; i < M; i++ ){
        for( int j = 0; j < N ; j++ ){
            B[j*M + i] = A[i*N + j]; 
        }
    }
}


void matrix_mult_host(const float* A, const float* B, float* C, int M, int N, int d, float scale) {
    // A = M * d
    // B = d * N
    // C = M * N

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            for (int k = 0; k < d; k++) {
                // A has width 'd', B has width 'N'
                sum += A[i * d + k] * B[k * N + j]; 
            }
            
            // C has width 'N'. Write the final scaled sum once.
            C[i * N + j] = sum / scale; 
        }
    }
}

void softmax_host(float* matrix, int M, int N) {
    for (int row = 0; row < M; row++) {
        float max_value = matrix[row * N];
        for (int col = 1; col < N; col++) {
            max_value = max(max_value, matrix[row * N + col]);
        }

        float sum_exp = 0.0f;
        for (int col = 0; col < N; col++) {
            matrix[row * N + col] = exp(matrix[row * N + col] - max_value);
            sum_exp += matrix[row * N + col];
        }

        for (int col = 0; col < N; col++) {
            matrix[row * N + col] /= sum_exp;
        }
    }
}


void solve_cpu(float* Q, float* K ,float* V, float* output,  int M, int N , int d ){
    float* KT = (float*)malloc(d*N*sizeof(float)); 
    float* temp_res = (float*) malloc(M*N*sizeof(float)); 
    matrix_tranpose_cpu(K,KT,N,d); 

    // Q = M*d
    // KT = N*d
    // temp res = M*N
    matrix_mult_host(Q,KT,temp_res,M,N,d,sqrt(d)); 
    softmax_host(temp_res, M, N);

    // temp res = M*N
    // V = N*d
    // output = M*d
    matrix_mult_host(temp_res,V,output,M,d,N,1);
    free(KT); 
    free(temp_res); 
}



int main(){
    int M = 40, N = 20, d = 12 ;

    // declare Host varaibles 
    float* h_Q  =   (float*)malloc(M*d*sizeof(float)); 
    float* h_K  =   (float*)malloc(N*d*sizeof(float)); 
    float* h_V  =   (float*)malloc(N*d*sizeof(float)); 
    float* h_out =  (float*)malloc(M*d*sizeof(float)); 
    float* glolden_trace = (float*)malloc(M*d*sizeof(float)); 
    

    // initilize Host arrays 
    assign_rand_val(h_Q,M,d); 
    assign_rand_val(h_K,N,d); 
    assign_rand_val(h_V,N,d); 


    // Declare Device Varaibles and assign memory
    float* d_Q, *d_K, *d_V, *d_out; 
    cudaMalloc(&d_Q,M*d*sizeof(float)); 
    cudaMalloc(&d_K,N*d*sizeof(float)); 
    cudaMalloc(&d_V,N*d*sizeof(float)); 
    cudaMalloc(&d_out,M*d*sizeof(float)); 

    // copy Host data to Device 
    cudaMemcpy(d_Q, h_Q, M*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, M*d*sizeof(float), cudaMemcpyHostToDevice);

    // call the solve function 
    solve(d_Q,d_K,d_V,d_out,M,N,d);

    // copy results back to host 
    cudaMemcpy(h_out,d_out,  M*d*sizeof(float), cudaMemcpyDeviceToHost);

    // compute the golder results 
    solve_cpu(h_Q,h_K,h_V,glolden_trace,M,N,d);

    for( int i = 0 ; i < M ; i++ ){
        for( int j =0 ; j < d; j++ ){
            if( fabs(h_out[i*d + j] - glolden_trace[i*d + j]) >= 1e-5){
                cout << "CPU and GPU error diff is greater than bounds \n"; 
                
                cudaFree(d_K); 
                cudaFree(d_V); 
                cudaFree(d_Q); 
                cudaFree(d_out); 

                return 0; 
            }
        }
    }

    cout << "Result of CPU and GPU matches succesfully \n"; 
    cudaFree(d_K); 
    cudaFree(d_V); 
    cudaFree(d_Q); 
    cudaFree(d_out); 
    free(h_Q); 
    free(h_K); 
    free(h_V); 
    free(h_out); 

    return 0; 
}