/**
 * @file GaussianBlur.cu
 * @author Sumit Garad (sgsuper12373@gmail.com)
 * @brief Guassian blur kernel code initial implementation. 
 * @version 0.1
 * @date 2026-09-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include<iostream>
#include<cuda.h> 
#include<cuda_runtime.h>
#include<cmath>
#include<cstdlib>


    __global__ void guassian_blur(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    int tid_x = threadIdx.x; // column 
    int tid_y = threadIdx.y; // row
    int gid_x = threadIdx.x + blockDim.x*blockIdx.x; 
    int gid_y = threadIdx.y + blockIdx.y*blockDim.y;  

    if( gid_x >= input_cols || gid_y >= input_rows ){
        return; 
    }

    // each thread in blocks loads the one tile into the shared memory but will it really useful? -> NO I guess
    // just run direct for loop to get the blur value 
    float blur_val = 0.0f; 
    for( int m = 0; m < kernel_rows; m++ ){
        for( int n  = 0; n < kernel_cols; n++ ){
            int input_row = gid_y + m - kernel_rows/2; 
            int input_col = gid_x + n - kernel_cols/2; 

            if( input_row >= 0 && input_row < input_rows && input_col >= 0 && input_col < input_cols ){
                blur_val += input[input_row*input_cols + input_col] * kernel[m*kernel_cols + n]; 
            }
        }
    }   

    output[gid_y*input_cols + gid_x] = blur_val; 
}


extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    dim3 TPB(16,16); 
    dim3 Blocks( (input_cols+TPB.x-1)/TPB.x , (input_rows + TPB.y-1)/TPB.y); 

    // lauch the kernel 
    guassian_blur<<<Blocks,TPB>>>(input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols); 
}


void assign_random_val(float* matrix, int M, int N){
    for( int i = 0; i < M; i++ ){
        for( int j = 0; j < N; j++ ){
            matrix[i*N+j] = (float)rand()/(float)RAND_MAX;
        }
    }
}

void solve_cpu(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols){
    for( int i = 0; i < input_rows; i++ ){
        for( int j = 0; j < input_cols; j++ ){
            float blur_val = 0.0f;

            for( int m = 0; m < kernel_rows; m++ ){
                for( int n = 0; n < kernel_cols; n++ ){
                    int input_row = i + m - kernel_rows/2;
                    int input_col = j + n - kernel_cols/2;

                    if( input_row >= 0 && input_row < input_rows && input_col >= 0 && input_col < input_cols ){
                        blur_val += input[input_row*input_cols + input_col] * kernel[m*kernel_cols + n];
                    }
                }
            }

            output[i*input_cols+j] = blur_val;
        }
    }
}




int main(){
    int rows = 123;
    int cols = 134;
    int kernel_rows = 5;
    int kernel_cols = 5;

    // Host variables declaration
    float* h_input = (float*)malloc(rows*cols*sizeof(float));
    float* h_output = (float*)malloc(rows*cols*sizeof(float));
    float* h_kernel = (float*)malloc(kernel_rows*kernel_cols*sizeof(float));
    float* golden_trace = (float*)malloc(rows*cols*sizeof(float));

    // initialize the input matrix and filter kernel with random values
    assign_random_val(h_input,rows,cols);
    assign_random_val(h_kernel,kernel_rows,kernel_cols);

    // Initialize the device variables
    float* d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input,rows*cols*sizeof(float));
    cudaMalloc(&d_output,rows*cols*sizeof(float));
    cudaMalloc(&d_kernel,kernel_rows*kernel_cols*sizeof(float));

    // copy data from Host to Device
    cudaMemcpy(d_input,h_input,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel,h_kernel,kernel_rows*kernel_cols*sizeof(float),cudaMemcpyHostToDevice);

    // solve and get the values
    solve(d_input,d_kernel,d_output,rows,cols,kernel_rows,kernel_cols);

    // copy result from Device to host
    cudaMemcpy(h_output,d_output,rows*cols*sizeof(float),cudaMemcpyDeviceToHost);

    // generate the golden trace
    solve_cpu(h_input,h_kernel,golden_trace,rows,cols,kernel_rows,kernel_cols);

    for( int i = 0; i < rows; i++ ){
        for( int j = 0; j < cols; j++ ){
            if(fabs(golden_trace[i*cols+j] - h_output[i*cols+j]) > 1e-5){
                std::cout << " absolute Error diff excceded the threshold \n";

                cudaFree(d_input);
                cudaFree(d_output);
                cudaFree(d_kernel);
                free(h_input);
                free(h_output);
                free(h_kernel);
                free(golden_trace);
                return 0;
            }
        }
    }

    std::cout << "Output Matched succesfully \n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_input);
    free(h_output);
    free(h_kernel);
    free(golden_trace);


    return 0; 
}