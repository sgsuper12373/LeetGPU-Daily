#include<cuda.h> 
#include<cuda_runtime.h> 
#include<bits/stdc++.h> 

using namespace std; 



__global__ void matrix_mult_kernel( const float* A, const float* B, float* output, int M , int N, int K, float scale){

    // A = M*K 
    // B = K*N 
    
    int col = threadIdx.x + blockIdx.x*blockDim.x; // column
    int row = threadIdx.y + blockIdx.y*blockDim.y; // row 

    if( col >= N || row >= M )

    // to calculate the output[x][y] multiply the xth row of A with yth columsn of B 
    // output matrix initilized with 0's
    for( int i = 0 ; i < K; i++ ){
        output[row*N + col] += ( A[ row*N + i ] * B[ i*M + col] ); 
    }
    output[row*N+ col] /= scale ; 
}

__global__ void matrix_transpose_kernel( const float* A, float* B, int M , int N ){

    int col = threadIdx.x + blockDim.x*blockIdx.x; // column
    int row = threadIdx.y + blockDim.y*blockIdx.y; // row 

    // if gid goes outof bound exit the thread 
    if( col >= N || row >= M ) return; 

    B[ col*M+ row] = A[ row*N + col]; 

}



extern void solve (const float*Q, const float* K , const float* V, float* output,  int M, int N , int d ){
    // first multiply the matrix Q and matrix K multiply each element by root d and then multiply with v 
    /*
        The plan is that we create general matrix multiplication function which takes input matrix A, B and their sizes with extra parameter scale 
        Then we need some other method to transpose the input matrix K, 
        but if we don't want to calculate the transpose then we 

        Q of size M×d
        K of size N×d
        V of size N×d
    */

    dim3 TPB(16, 16);

    dim3 transposeBlocks((d + 15) / 16,(N + 15) / 16);
    dim3 qktBlocks((N + 15) / 16,(M + 15) / 16);
    dim3 outputBlocks( (d + 15) / 16, (M + 15) / 16);

    float* d_KT;
    float* temp_res;

    cudaMalloc(&d_KT, N * d * sizeof(float));
    cudaMalloc(&temp_res, M * N * sizeof(float));

    matrix_transpose_kernel<<<transposeBlocks, TPB>>>(
        K, d_KT, N, d
    );

    matrix_mult_kernel<<<qktBlocks, TPB>>>(
        Q, d_KT, temp_res,
        M, N, d, sqrt(d)
    );

    matrix_mult_kernel<<<outputBlocks, TPB>>>(
        temp_res, V, output,
        M, d, N, 1.0f
    );


}


void assign_rand_val(vector<vector<float>>& A){
    for( int i = 0 ; i < A.size(); i++ ){
        for( int j = 0; j < A[0].size(); j++ ){
            A[i][j]= (float)rand()/RAND_MAX; 
        }
    }
}


void matrix_tranpose_cpu(const vector<vector<float>>& A, vector<vector<float>>&B){
    // A = M*N 
    // B = N*M
    int M = A.size(); 
    int N = A[0].size(); 
    for( int i = 0 ; i < M; i++ ){
        for( int j = 0; j < N ; j++ ){
            B[j][i] = A[i][j]; 
        }
    }
}

void matrix_mult_host( const vector<vector<float>>&A, const vector<vector<float>>&B, vector<vector<float>>& C, float scale ){
    // A = M*d
    // B = d*N
    int M = A.size(); 
    int d = A[0].size(); 
    int N = B[0].size(); 

    for( int i = 0 ; i < M ; i++ ){
        for( int j = 0; j < N ; j++ ){
            for( int k = 0 ; k < d ; k++ ){
                C[i][j] += A[i][k] * B[k][j]; 
            }
        }
    }
}


void solve_cpu(const vector<vector<float>>&Q, const vector<vector<float>>& K , const vector<vector<float>>&V, vector<vector<float>>&output,  int M, int N , int d ){
    vector<vector<float>> KT(d,vector<float>(N)); 
    vector<vector<float>> temp_res(M,vector<float>(d,0)); 
    matrix_tranpose_cpu(K,KT); 
    matrix_mult_host(Q,KT,temp_res,sqrt(d)); 
    matrix_mult_host(temp_res,V,output,1);
}



int main(){
    int M = 40, N = 20, d = 12 ;

    // declare Host varaibles 
    vector<vector<float>> h_Q(M,vector<float>(d,0)); 
    vector<vector<float>> h_K(N,vector<float>(d,0)); 
    vector<vector<float>> h_V(N,vector<float>(d,0)); 
    vector<vector<float>> h_out(M,vector<float>(d,0)); 
    vector<vector<float>> glolden_trace(M,vector<float>(d,0)); 
    

    // initilize Host arrays 
    assign_rand_val(h_Q); 
    assign_rand_val(h_K); 
    assign_rand_val(h_V); 


    // Declare Device Varaibles and assign memory
    float* d_Q, *d_K, *d_V, *d_out; 
    cudaMalloc(&d_Q,M*d*sizeof(float)); 
    cudaMalloc(&d_K,N*d*sizeof(float)); 
    cudaMalloc(&d_V,N*d*sizeof(float)); 
    cudaMalloc(&d_out,M*d*sizeof(float)); 

    // copy Host data to Device 
    cudaMemcpy(d_Q, h_Q.data(), M*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_Q.data(), N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_Q.data(), N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_Q.data(), M*d*sizeof(float), cudaMemcpyHostToDevice);

    // call the solve function 
    solve(d_Q,d_K,d_V,d_out,M,N,d);

    // copy results back to host 
    cudaMemcpy(h_Q.data(),d_out,  M*d*sizeof(float), cudaMemcpyDeviceToDevice);

    // compute the golder results 
    solve_cpu(h_Q,h_K,h_V,glolden_trace,M,N,d);

    for( int i = 0 ; i < M ; i++ ){
        for( int j =0 ; j < d; j++ ){
            if( fabs(h_K[i][j] - glolden_trace[i][j]) >= 1e-5){
                cout << "CPU and GPU error diff is greater than bounds \n"; 
                return 0; 
            }
        }
    }

    cout << "Result of CPU and GPU matches succesfully \n"; 

    return 0; 
}