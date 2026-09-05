#include<cuda.h>
#include<cuda_runtime.h>
#include<bits/stdc++.h> 

using namespace std;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        exit(EXIT_FAILURE);
    }
}

__global__ void conv_3d(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    if (i < output_depth && j < output_rows && k < output_cols) {
        float sum = 0.0f;
        for(int ki = 0; ki < kernel_depth; ++ki) {
            for (int kj = 0; kj < kernel_rows; ++kj) {
                for (int kk = 0; kk < kernel_cols; ++kk) {
                    sum += input[(i + ki) * input_rows * input_cols + (j + kj) * input_cols + (k + kk)] 
                    * kernel[ki * kernel_rows * kernel_cols + kj * kernel_cols + kk];
                }
            }
        }
        output[i * output_rows * output_cols + j * output_cols + k] = sum;
    }
}

// input, kernel, output are device pointers
extern void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {
    const int output_depth = input_depth - kernel_depth + 1;
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;

    // Keep the block below CUDA's limit of 1024 threads per block.
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (output_depth + threadsPerBlock.z - 1) / threadsPerBlock.z);
    conv_3d<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_depth, input_rows,
                                         input_cols, kernel_depth, kernel_rows, kernel_cols);
    check_cuda(cudaGetLastError(), "3D convolution kernel launch");
}

void assign_random_values(float* values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void solve_cpu(const float* input, const float* kernel, float* output, int input_depth,
               int input_rows, int input_cols, int kernel_depth, int kernel_rows,
               int kernel_cols) {

    int output_rows = input_rows-kernel_rows+1; 
    int output_cols = input_cols-kernel_cols+1; 
    int output_depth = input_depth-kernel_depth+1; 

    for (int i = 0; i < output_depth; ++i) {
        for (int j = 0; j < output_rows; ++j) {
            for (int k = 0; k < output_cols; ++k) {
                float sum = 0.0f;
                for (int ki = 0; ki < kernel_depth; ++ki) {
                    for (int kj = 0; kj < kernel_rows; ++kj) {
                        for (int kk = 0; kk < kernel_cols; ++kk) {
                            sum += input[(i + ki) * input_rows * input_cols +
                                         (j + kj) * input_cols + (k + kk)] *
                                   kernel[ki * kernel_rows * kernel_cols +
                                          kj * kernel_cols + kk];
                        }
                    }
                }
                output[i * output_rows * output_cols + j * output_cols + k] = sum;
            }
        }
    }
}
 

int main (){
    
    // input, output and kernel dimentions 
    int input_rows = 24; 
    int input_cols = 12; 
    int input_depth = 18; 
    int kernel_rows = 4; 
    int kernel_cols = 7; 
    int kernel_depth = 6;
    int output_rows = input_rows-kernel_rows+1; 
    int output_cols = input_cols-kernel_cols+1; 
    int output_depth = input_depth-kernel_depth+1; 

    // input, output and kenel sizes
    size_t input_count = static_cast<size_t>(input_rows) * input_cols * input_depth;
    size_t kernel_count = static_cast<size_t>(kernel_rows) * kernel_cols * kernel_depth;
    size_t output_count = static_cast<size_t>(output_rows) * output_cols * output_depth;
    size_t input_size = input_count * sizeof(float);
    size_t kernel_size = kernel_count * sizeof(float);
    size_t output_size = output_count * sizeof(float);

    // Host varaibles declaration and memory allocation 
    float *h_input = (float*)malloc(input_size); 
    float *h_kernel = (float*)malloc(kernel_size);
    float *h_output = (float*)malloc(output_size); 
    float *goldenTrace = (float*)malloc(output_size); 

    if (!h_input || !h_kernel || !h_output || !goldenTrace) {
        cerr << "Host allocation failed\n";
        free(h_input);
        free(h_kernel);
        free(h_output);
        free(goldenTrace);
        return EXIT_FAILURE;
    }

    assign_random_values(h_input, input_count);
    assign_random_values(h_kernel, kernel_count);

    // Device variables declaration and memory allocation 
    float *d_input , *d_output , *d_kernel ;
    check_cuda(cudaMalloc(&d_input,input_size), "cudaMalloc input");
    check_cuda(cudaMalloc(&d_output,output_size), "cudaMalloc output");
    check_cuda(cudaMalloc(&d_kernel,kernel_size), "cudaMalloc kernel");

    // copy Host data to Device
    check_cuda(cudaMemcpy(d_input,h_input,input_size,cudaMemcpyHostToDevice), "copy input to device");
    check_cuda(cudaMemcpy(d_kernel,h_kernel,kernel_size,cudaMemcpyHostToDevice), "copy kernel to device");

    //Solve the things on Device 
    solve(d_input,d_kernel,d_output,input_depth,input_rows,input_cols,kernel_depth,kernel_rows,kernel_cols); 

    // copy back results from device to host 
    check_cuda(cudaMemcpy(h_output,d_output,output_size,cudaMemcpyDeviceToHost), "copy output to host");

    // get the CPU resutls 
    solve_cpu(h_input,h_kernel,goldenTrace,input_depth,input_rows,input_cols,kernel_depth,kernel_rows,kernel_cols); 


    bool success = true;
    for (size_t i = 0; i < output_count; ++i) {
        if (fabsf(h_output[i] - goldenTrace[i]) > 1e-5f) {
            cerr << "Mismatch at output index " << i << ": GPU=" << h_output[i]
                 << ", CPU=" << goldenTrace[i] << '\n';
            success = false;
            break;
        }
    }


    free(h_input);
    free(h_kernel);
    free(h_output);
    free(goldenTrace);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);


    
    if (success) {
        cout << "3D convolution output matched successfully\n";
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
} 
