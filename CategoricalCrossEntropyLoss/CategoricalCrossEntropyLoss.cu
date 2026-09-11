#include<bits/stdc++.h> 
#include<cuda.h> 
#include<cuda_runtime.h> 

using namespace std; 



__global__ void reduce_sum(const float* input, float* sum , int N ){

    // shared memory for storing block level array 
    extern __shared__ float sdata[]; 

    // threadId to store the indices 
    int tid = threadIdx.x; 
    int gid = threadIdx.x + blockDim.x*blockIdx.x; 

    // populated shared memory with data 
    if( gid >= N ){
        sdata[tid] = 0 ;
    }else{
        sdata[tid] = input[gid]; 
    }

    // wait for all threads to write the data 
    __syncthreads(); 

    // do parallel reduction 
    for( int stride = blockDim.x/2 ; stride > 0; stride/=2 ){
        if( tid < stride ){
            sdata[tid]  += sdata[tid+stride]; 
        }
        __syncthreads(); 
    }

    // atomically update the sum varaible by one of thread
    if( tid == 0 ){
        atomicAdd(sum,sdata[0]/N); 
    }

    
}

__global__ void loss_j_kernel(const float* logits,const int* true_labels,float* loss_vector,int N,int C) {

    // global Id 
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // out of bound condition
    if (gid >= N)
        return;

    // Find maximum logit -> As C ranges from 1 to 1000, 
    // for loop is best instead of some reduction device function or
    // kenel lauch via dynamic parallelism which will have more overhead 
    float max_logit = logits[gid * C];

    for (int k = 1; k < C; k++) {
        max_logit = fmaxf(max_logit, logits[gid * C + k]);
    }

    // Compute sum(exp(logit - max_logit))
    float sum_exp = 0.0f;

    for (int k = 0; k < C; k++) {
        sum_exp += expf(logits[gid * C + k] - max_logit);
    }

    // log(sum(exp(logits)))
    float log_sum_exp = max_logit + logf(sum_exp);

    // Cross entropy loss
    loss_vector[gid] = log_sum_exp - logits[gid * C + true_labels[gid]];
}

extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    
    int TPB = 256; 
    int Blocks = (N+TPB-1)/TPB;
    size_t shrdBytes = TPB*sizeof(float); 
    // temprory vector for storing loss of each element 
    float* loss_vector; 
    cudaMalloc(&loss_vector, N*sizeof(float)); 
    cudaMemset(loss,0,sizeof(float)); // set the loss to 0 already 

    // lauch kernel to calculate loss at each point 
    loss_j_kernel<<<Blocks,TPB>>>(logits,true_labels,loss_vector,N,C);

    // accumulate the loss
    reduce_sum<<<Blocks,TPB,shrdBytes>>>(loss_vector,loss,N); 
    cudaDeviceSynchronize();

    // free the temprory loss vector 
    cudaFree(loss_vector); 

}

void solve_cpu(const float* logits, const int* true_labels, float* loss, int N, int C){
    float* loss_vector = (float*)malloc(N*sizeof(float)); 

    for( int i = 0 ; i < N ; i++ ){

        float max_logit = logits[i * C];
        for (int k = 1; k < C; k++) {
            max_logit = fmaxf(max_logit, logits[i * C + k]);
        }

        // Compute sum(exp(logit - max_logit))
        float sum_exp = 0.0f;

        for (int k = 0; k < C; k++) {
            sum_exp += expf(logits[i * C + k] - max_logit);
        }

        // log(sum(exp(logits)))
        float log_sum_exp = max_logit + logf(sum_exp);

        // Cross entropy loss
        loss_vector[i] = log_sum_exp - logits[i * C + true_labels[i]];
    }

    for( int i = 0;  i < N ; i++ ){
        *loss += loss_vector[i]; 
    }

    *loss /= N ; 

}

int main(){
    // Matrix of input predicted logits Z of size N * C and vector true_lables of size N 
    // loss_j = log( sum_over_k=1_to_k=c ( exp(zjk))) - z_j,yj

    /*
        Input:  N = 2, C = 3
        logits = [1.0, 2.0, 0.5]
                 [0.1, 3.0, 1.5]
        true_labels = [1, 1]

        loss[0] = 
Output: loss = [0.3548926]
    */


    int N = 1032;
    int C = 324; 
    size_t logit_size = N*C*sizeof(float); 
    size_t label_size = N*sizeof(int); 

    // HOST varaibles and allocation 
    float* h_logit = (float*)malloc(logit_size); 
    int* h_lable = (int*)malloc(label_size);
    float* h_loss = (float*)malloc(sizeof(float)); 
    float* goldenTrace = (float*)malloc(sizeof(float)); 

    for (int i = 0; i < N * C; i++) {
        h_logit[i] = static_cast<float>((i * 17) % 100) / 10.0f - 5.0f;
    }
    for (int i = 0; i < N; i++) {
        h_lable[i] = i % C;
    }
    *h_loss = 0.0f;
    *goldenTrace = 0.0f;

    // Device variables delcaration and memory allocatoin 
    float *d_logit, *d_loss; 
    int *d_lable; 
    cudaMalloc(&d_logit,logit_size); 
    cudaMalloc(&d_lable,label_size); 
    cudaMalloc(&d_loss,sizeof(float));

    // copy data from HOST -> DEVICE 
    cudaMemcpy(d_logit, h_logit, logit_size,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_lable, h_lable, label_size,cudaMemcpyHostToDevice); 

    solve(d_logit,d_lable,d_loss,N,C); 
    cudaMemcpy(h_loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost);

    solve_cpu(h_logit,h_lable,goldenTrace,N,C); 

    cout<< "Host: " << *goldenTrace << "\nDevice : " << *h_loss << "\n"; 
    if( fabs(*goldenTrace - *h_loss) > 1e-4){
        cout<< "HOST DEVCIE result don't match \n"; 
    }else{
        cout << "HOST DEVICE results matched successfully \n"; 
    }

    cudaFree(d_logit); 
    cudaFree(d_loss); 
    cudaFree(d_lable); 

    free(h_logit); 
    free(h_loss); 
    free(h_lable); 
    free(goldenTrace); 

    return 0; 
}