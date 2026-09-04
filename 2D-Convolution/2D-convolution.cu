#include<cuda.h> 
#include<cuda_runtime.h>
#include<bits/stdc++.h>

using namespace std; 



/**
 * @brief This basically does the dot product of two matrices and return the value 
 * @note instead of doing this we can write some global kernel which lauches from the running kernel itself so this computation can be also parallelized 
 * @param input 
 * @param kernel 
 * @param M 
 * @param N 
 * @return __device__ 
 */
__host__ __device__ float compute_elem(const float* input, const float* kernel, int kernel_rows, int kernel_cols, int input_cols){
    float ans = 0.0f; 
    for(int m = 0; m < kernel_rows; m++){
        for(int n = 0; n < kernel_cols; n++){
            // input strides by the full image width, kernel strides by kernel width
            ans += input[m * input_cols + n] * kernel[m * kernel_cols + n]; 
        }
    }
    return ans; 
}



/**
 * @brief apply given kenel filter to the input matrix
 *         Lauhces the kernel with 2d blocks, kernel will handle how to manage the boudaries 
 * @note The kernel applied only where the filter completly overlaps with the input. 
 *       for index i,j we need to apply kernel filter on input starting at i,j if can  overlap with the input matrix 
 *        
 * 
 * @param input : pointer to input matrix stored in the row-major format 
 * @param kernel : pointer to filter matrix also stored in row-major format 
 * @param output : pointer to output after applying the 2D convolution kernel 
 * @param input_rows : number of rows in input matrix, which will be also equal to the number of rows in output matrix
 * @param input_cols : number of cols in output matrix, which will be also eqal to the number of cols in output matrix
 * @param kernel_rows : number of rows in kernel matrix
 * @param kernel_cols : number of cols in kernel matrix
 */
__global__ void convolution_kernel( const float*  input, const float* kernel, float* output, const int M, const int N, const int kernel_row, const int kernel_col){
    int global_col = threadIdx.x + blockDim.x * blockIdx.x; 
    int global_row = threadIdx.y + blockDim.y * blockIdx.y; 
    int output_cols = N-kernel_col+1; 

    // now we will directly apply kernel without any optimization. directly operating on the global memory. 
    // first check if the kernel overlaps with matrix for given cell. 
    //if yes-> calculate the output matrix  
    //else -> copy value directly 

    // if out of bound then 5-star
    // if within the bounds check if can be overlapped, if yes then apply kerel else copy value
    if( global_row > M - kernel_row  || global_col > N - kernel_col){
        return ; 
    }else{
        output[global_row*output_cols+global_col] = compute_elem( input + global_row*N+global_col , kernel, kernel_row, kernel_col,N); 
    }


}



extern void solve(const float* input, const float* kernel, float* output, int input_rows,int input_cols, int kernel_rows, int kernel_cols){
    dim3 TPB(16,16); 
    dim3 Blocks( (input_cols+TPB.x-1)/TPB.x , (input_rows + TPB.y-1)/TPB.y); 

    // lauch the kernel 
    convolution_kernel<<<Blocks,TPB>>>(input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols); 
}


void assign_random_val(float* matrix, int M , int N ){
    for( int i = 0 ; i < M ; i++ ){
        for(int j = 0 ; j < N ; j++){
            matrix[i*N+j] = (float)rand()/(float)RAND_MAX; 
        }
    }
}

void solve_cpu(const float* input, const float* kernel, float* output, int M,int N, int kernel_rows, int kernel_cols){
    int output_rows = M-kernel_rows+1;
    int output_cols = N-kernel_cols+1; 
    for( int i = 0; i < output_rows; i++ ){
        for( int j = 0; j < output_cols; j++ ){
            if( i > M - kernel_rows || j > N-kernel_cols){
                return; 
            }else{
                output[i*output_cols+j] = compute_elem(input+(i*N+j),kernel,kernel_rows,kernel_cols,N); 
            }
        }
    }
}

int main(){
    int rows = 123; 
    int cols = 134; 
    int kernel_rows = 12; 
    int kernel_cols = 13; 
    int output_rows = rows-kernel_rows+1; 
    int output_cols = cols-kernel_cols+1;

    // Host variables declaration 
    float* h_input = (float*)malloc(rows*cols*sizeof(float)); 
    float* h_output = (float*)malloc((output_rows)*(output_cols)*sizeof(float)); 
    float* h_kernel = (float*)malloc(kernel_cols*kernel_rows*sizeof(float)); 
    float* golden_trace = (float*)malloc(output_cols*output_rows*sizeof(float)); 

    // initilize the input matrix and filter kernel with random values 
    assign_random_val(h_input,rows,cols); 
    assign_random_val(h_kernel,kernel_rows,kernel_cols); 

    // Initilize the device variables 
    float* d_input, *d_output, *d_kernel; 
    cudaMalloc(&d_input,rows*cols*sizeof(float)); 
    cudaMalloc(&d_output,(output_rows)*(output_cols)*sizeof(float)); 
    cudaMalloc(&d_kernel,kernel_rows*kernel_cols*sizeof(float)); 

    // copy data from Host to Device 
    cudaMemcpy(d_input,h_input,rows*cols*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(d_kernel,h_kernel,kernel_rows*kernel_cols*sizeof(float),cudaMemcpyHostToDevice); 

    // solve and get the values 
    solve(d_input,d_kernel,d_output,rows,cols,kernel_rows,kernel_cols); 

    // copy result from Device to host 
    cudaMemcpy(h_output,d_output,(output_rows)*(output_cols)*sizeof(float),cudaMemcpyDeviceToHost); 

    // generate the golden trace
    solve_cpu(h_input,h_kernel,golden_trace,rows,cols, kernel_rows,kernel_cols);

    for( int i = 0 ; i < output_rows; i++ ){
        for( int j = 0 ; j < output_cols; j++ ){
            if(fabs(golden_trace[i*output_cols+j] - h_output[i*output_cols+j]) > 1e-5){
                cout << " absolute Error diff excceded the threshold \n"; 

                cudaFree(d_input) ; 
                cudaFree(d_output); 
                cudaFree(d_kernel); 
                free(h_input); 
                free(h_output); 
                free(h_kernel); 
                free(golden_trace); 
                return 0 ; 
            }
        }
    }


    cout << "Output Matched succesfully \n"; 

    cudaFree(d_input) ; 
    cudaFree(d_output); 
    cudaFree(d_kernel); 
    free(h_input); 
    free(h_output); 
    free(h_kernel); 
    free(golden_trace); 
    return 0 ; 

}