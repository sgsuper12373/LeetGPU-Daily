#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

#define TPB  256
#define WARP 32

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            cerr << "CUDA error: " << cudaGetErrorString(err)              \
                 << " at line " << __LINE__ << endl;                       \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// Inclusive scan inside one warp using shuffle-up
__device__ __forceinline__ int warpInclusiveScan(int val, int lane) {
    for (int offset = 1; offset < WARP; offset <<= 1) {
        int other = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += other;
    }
    return val;
}

// scan each block of TPB elements, write each block's total to blockSums
__global__ void blockScan_kernel(int* arr, int* blockSums, int N) {
    __shared__ int warpTotals[TPB / WARP];

    int tid    = threadIdx.x;
    int gid    = blockIdx.x * blockDim.x + tid;
    int lane   = tid % WARP;
    int warpId = tid / WARP;

    int val = (gid < N) ? arr[gid] : 0;

    //  scan inside each warp
    val = warpInclusiveScan(val, lane);

    // last lane of each warp stores the warp total
    if (lane == WARP - 1) warpTotals[warpId] = val;
    __syncthreads();

    // first warp scans the warp totals
    if (warpId == 0) {
        int t = (lane < TPB / WARP) ? warpTotals[lane] : 0;
        t = warpInclusiveScan(t, lane);
        if (lane < TPB / WARP) warpTotals[lane] = t;
    }
    __syncthreads();

    // add the sum of all previous warps
    if (warpId > 0) val += warpTotals[warpId - 1];

    if (gid < N) arr[gid] = val;

    // padded elements are 0, so the last thread always holds the block total
    if (blockSums != nullptr && tid == TPB - 1) blockSums[blockIdx.x] = val;
}

// add the (scanned) total of all previous blocks
__global__ void addBlockSums_kernel(int* arr, const int* blockSums, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < N) arr[gid] += blockSums[blockIdx.x - 1];
}

// Recursive host driver: works for any N
__host__ void prefixSumDevice(int* d_arr, int N) {
    int blocks = (N + TPB - 1) / TPB;

    if (blocks == 1) {
        blockScan_kernel<<<1, TPB>>>(d_arr, nullptr, N);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int* d_blockSums;
    CUDA_CHECK(cudaMalloc(&d_blockSums, blocks * sizeof(int)));

    blockScan_kernel<<<blocks, TPB>>>(d_arr, d_blockSums, N);
    CUDA_CHECK(cudaGetLastError());

    prefixSumDevice(d_blockSums, blocks);          // scan the block sums

    addBlockSums_kernel<<<blocks, TPB>>>(d_arr, d_blockSums, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_blockSums));
}

__host__ void assign_random_val(int* arr, int N) {
    for (int i = 0; i < N; i++) arr[i] = rand() % 10000;
}

int main() {
    int N = 1034;

    int* h_arr       = (int*)malloc(N * sizeof(int));
    int* h_prefixSum = (int*)malloc(N * sizeof(int));
    int* goldenTrace = (int*)malloc(N * sizeof(int));

    srand(42);
    assign_random_val(h_arr, N);

    goldenTrace[0] = h_arr[0];
    for (int i = 1; i < N; i++) goldenTrace[i] = goldenTrace[i - 1] + h_arr[i];

    int* d_prefixSum;
    CUDA_CHECK(cudaMalloc(&d_prefixSum, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prefixSum, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    prefixSumDevice(d_prefixSum, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_prefixSum, d_prefixSum, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_prefixSum[i] != goldenTrace[i]) {
            cout << "Mismatch at " << i << ": got " << h_prefixSum[i]
                 << ", expected " << goldenTrace[i] << endl;
            ok = false;
            break;
        }
    }
    cout << (ok ? "SUCCESS: Results Match!" : "FAILURE: Results Do Not Match!") << endl;

    free(h_arr); free(goldenTrace); free(h_prefixSum);
    cudaFree(d_prefixSum);
    return 0;
}