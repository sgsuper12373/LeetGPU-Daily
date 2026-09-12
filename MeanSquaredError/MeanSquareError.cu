#include<bits/stdc++.h>
#include<cuda.h> 
#include<cuda_runtime.h>


using namespace std; 


/**
 * @brief Reduction based MSE loss function. 
 *        store the diff to the  shared array and the do parallel reduction 
 * 
 * @param predictions 
 * @param targets 
 * @param mse 
 * @param N 
 * @return __global__ 
 */
__global__ void MSE_kernel_v2(const float* predictions, const float* targets, float* mse, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x; 
    int gid = threadIdx.x + blockDim.x*blockIdx.x; 

    // store the diff in block_local array 
    if( gid < N ){
        float diff = predictions[gid] - targets[gid]; 
        sdata[tid] = diff*diff; 
    }else{
        sdata[tid] = 0 ; 
    }

    // wait for all threads to load the data 
    __syncthreads(); 

    // paralllel reduction 
    for( int stride = blockDim.x/2 ; stride > 0 ; stride /= 2 ){
        if( tid < stride ){
            sdata[tid] += sdata[tid+stride]; 
        }
        __syncthreads();
    }

    if( tid == 0 ) {
        atomicAdd(mse,sdata[0]/N); 
    }
}

__global__ void MSE_kernel_v1(const float* predictions, const float* targets, float* mse, int N){

    __shared__ float sdata; 
    int tid = threadIdx.x; 
    int gid = threadIdx.x + blockDim.x*blockIdx.x; 

    if( tid == 0 ) sdata = 0.0f; 
    __syncthreads(); 

    if( gid < N ) {
        // data race hence use atomic add 
        // sdata += (predictions[gid]-targets[gid])*(predictions[gid]-targets[gid]); 
        atomicAdd(&sdata,(predictions[gid]-targets[gid])*(predictions[gid]-targets[gid])); 
    }


    __syncthreads(); 

    if( tid == 0){
        atomicAdd(mse,sdata/N); 
    }
}

extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int TPB = 256; 
    int Blocks = (N+TPB-1)/TPB; 
    int shrdBytes = TPB*sizeof(float); 

    // mse should be zero before computation 
    cudaMemset(mse, 0, sizeof(float));

    // v1 is basically squental because of atomics. we can use something like reduction...
    // of each thread will compute local sum and update globally using atomic 
    // MSE_kernel_v1<<<Blocks,TPB>>>(predictions,targets,mse,N); 

    MSE_kernel_v2<<<Blocks,TPB,shrdBytes>>>(predictions,targets,mse,N); 

    cudaDeviceSynchronize(); 

}



void assign_rand_val(float* arr, int N ){
    for( int i = 0 ; i < N ; i++){
        arr[i] = (float) rand() / (float)RAND_MAX; 
    }
}

void solve_cpu(float* predictions, float* targets, float* mse, int N ){
    float sum = 0.0f; 
    for( int i = 0 ; i < N ; i++ ){
        sum += (predictions[i] - targets[i]) *(predictions[i] - targets[i]); 
    } 
    
    *mse = sum/N;   

}


int main(){

    int N = 3534; 
    size_t arr_size = N*sizeof(float); 

    // Host variables declaration and memory allocatoin
    float *h_pred = (float*)malloc(arr_size); 
    float *h_trgt = (float*)malloc(arr_size); 
    float *h_MSE = (float*)malloc(sizeof(float)); 
    float *goldenTrace = (float*)malloc(sizeof(float)); 

    // assing random values to prediction and targets 
    assign_rand_val(h_pred,N);
    assign_rand_val(h_trgt,N);

    // Device variables declaration and memory allocatoin
    float *d_pred,*d_trgt,*d_MSE; 
    cudaMalloc(&d_pred,arr_size);
    cudaMalloc(&d_trgt,arr_size);
    cudaMalloc(&d_MSE,arr_size);


    // Copy data from HOST to DEVICE 
    cudaMemcpy(d_pred,h_pred,arr_size,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_trgt,h_trgt,arr_size,cudaMemcpyHostToDevice); 

    // Device computation 
    solve(d_pred,d_trgt,d_MSE,N);

    //copy back results from HOST to DEVICE 
    cudaMemcpy(h_MSE,d_MSE,sizeof(float),cudaMemcpyDeviceToHost); 

    // CPU results 
    solve_cpu(h_pred,h_trgt,goldenTrace,N); 


    cout << "HOST: " << *goldenTrace << "\n"; 
    cout << "DEVICE: " << *h_MSE << "\n"; 
    if( fabs(*goldenTrace - *h_MSE) > 1e-4){
        cout << "DEVICE and HOST results doesn't match \n"; 
    }else{
        cout << "DEVICE and HOST results matched succesfully \n"; 
    }


    cudaFree(d_pred); 
    cudaFree(d_trgt); 
    cudaFree(d_MSE); 
    free(h_pred);
    free(h_trgt); 
    free(h_MSE);
    free(goldenTrace); 





    return 0;
}