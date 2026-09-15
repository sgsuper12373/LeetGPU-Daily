#include<iostream> 
#include<cuda.h> 
#include<cuda_runtime.h>

using namespace std; 

/**
 * @brief My basic Idea is to sort the array in descending order and pick the first k  elements...
 *        This could be lil hard as we are talking about array large enough that can occupy the multiple blocks. 
 *        Lets think from the block level array perspective -> sort the block array and update it in the global array 
 *        now I what I want to do is something like merge sort but expect I more than two array to work 
 *        
 *        // in each block thread 0 checks the largest element and looks for global element and update if it's larger than current element 
 *           this is done atomically. we have some global index to track current position we are trying to fill in. 
 *           by given input constraints  1 ≤ N ≤ 100,000,000 && 1 ≤ k ≤ N
 *        
 *        -> What if I sort them on block level and then some other kernel to merge them using some strided access? 
 *              b1 b2 b3 b4 b5 b6  
 *              phase 1 
 *                 merge b1 and b2 into b1 
 *                 merge b3 and b4 into b3 
 *                 merge b5 and b6 into b5
 *              phase 2 
 *                  merge b1 and b3 
 *                  merge b5 and nothing -> b5 
 *              phase3 
 *                  merge b1 and b5 
 *         Here I guess merge can be some device function which does the two pointer merge thing 
 *         some other kernel will do the merge part 
 */





__global__ void block_sort( float* input, int N){
    // merge sort feels bad Idea here due to recursion and GPU have less stack memory so there could be stack overflow 
    // use shared memory to load the array. sort and write back into the global array 

    extern __shared__ float sdata[]; 

    int tid = threadIdx.x; 
    int gid = threadIdx.x + blockDim.x*blockIdx.x; 

    // load data into the local memory 
    if( gid < N ){
        sdata[tid] = input[gid]; 
    }else{
        sdata[tid] = -INFINITY; // we are sorting in descending order hence this is best we can do 
    }

    // wait !! for all threads to load the data 
    __syncthreads(); 

    // now do the sort -> optimzed selection sort with flag which stops if next array is already sorted 
    
    int arr_size = blockDim.x;

    for (int i = 0; i < arr_size - 1; i++) {

        int curr_max_ind = i;

        for (int j = i + 1; j < arr_size; j++) {
            if (sdata[j] > sdata[curr_max_ind]) {
                curr_max_ind = j;
            }
        }

        if (curr_max_ind != i) {
            float temp = sdata[i];
            sdata[i] = sdata[curr_max_ind];
            sdata[curr_max_ind] = temp;
        }
    }

    // update the global array to this blocked sorted array 
    if( gid  < N ){
        input[gid] = sdata[tid]; 
    }

}

__global__ void block_merge(float* input, int b1, int b2,  int N ){

}




extern "C" void solve(const float* input, float* output, int N, int k) {
    int TPB = 512; 
    int Blokcs = N/TPB; 
    size_t shrdBytes = TPB*sizeof(float); 

    // temprory input array 
    float* temp; 
    cudaMalloc(&temp,N*sizeof(float)); 
    cudaMemcpy(temp,input,N*sizeof(float),cudaMemcpyDeviceToDevice); 

    // sort block 
    block_sort(temp,N); 

    // block merge 
    int stride = Blokcs/2; 
    while( stride > 0 ){
        for( int i = 0 ; i < stride ; i++ ){
            block_merge(temp,i,i+stride,N);
        }
    }


    
}



int main(){


    return 0; 
}