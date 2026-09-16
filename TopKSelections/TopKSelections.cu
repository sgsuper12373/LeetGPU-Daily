#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 256

template<typename T>
__device__ void swap_values(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}

__global__ void bitonic_sort_kernel(float* input, int sort_k, int p, int N) {
    int a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = a_idx ^ p;

    if (a_idx >= N || b_idx >= N) {
        return;
    }

    if (b_idx > a_idx) {
        if ((a_idx & sort_k) ? input[a_idx] > input[b_idx] : input[a_idx] < input[b_idx]) {
            swap_values(input[a_idx], input[b_idx]);
        }
    }
}

extern "C" void solve(const float* input, float* output, int N, int k) {
    if (input == nullptr || output == nullptr || N <= 0 || k <= 0) {
        return;
    }

    k = min(k, N);

    int padded_N = 1;
    while (padded_N < N) {
        padded_N <<= 1;
    }

    float* d_padded = nullptr;
    cudaMalloc(&d_padded, padded_N * sizeof(float));

    vector<float> h_padded(padded_N, -FLT_MAX);
    for (int i = 0; i < N; i++) {
        h_padded[i] = input[i];
    }

    cudaMemcpy(d_padded, h_padded.data(), padded_N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (padded_N + TILE_SIZE - 1) / TILE_SIZE;
    for (int sort_k = 2; sort_k <= padded_N; sort_k <<= 1) {
        for (int p = sort_k >> 1; p > 0; p >>= 1) {
            bitonic_sort_kernel<<<blocks, TILE_SIZE>>>(d_padded, sort_k, p, padded_N);
        }
    }

    vector<float> h_sorted(padded_N, -FLT_MAX);
    cudaMemcpy(h_sorted.data(), d_padded, padded_N * sizeof(float), cudaMemcpyDeviceToHost);

    partial_sort(h_sorted.begin(), h_sorted.begin() + k, h_sorted.end(), greater<float>());

    vector<float> h_topk(k);
    copy(h_sorted.begin(), h_sorted.begin() + k, h_topk.begin());

    cudaMemcpy(output, h_topk.data(), k * sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(d_padded);
}

int main() {
    int N = 16;
    int k = 5;

    vector<float> h_input(N);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / (float)RAND_MAX * 100.0f;
    }

    vector<float> h_expected = h_input;
    partial_sort(h_expected.begin(), h_expected.begin() + k, h_expected.end(), greater<float>());

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, k * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(d_input, d_output, N, k);

    vector<float> h_output(k);
    cudaMemcpy(h_output.data(), d_output, k * sizeof(float), cudaMemcpyDeviceToHost);

    bool matches = true;
    for (int i = 0; i < k; i++) {
        if (fabs(h_output[i] - h_expected[i]) > 1e-4f) {
            matches = false;
            break;
        }
    }

    cout << "Top K values on GPU: ";
    for (int i = 0; i < k; i++) {
        cout << h_output[i] << " ";
    }
    cout << "\n";

    if (matches) {
        cout << "Top-K result matches the host reference.\n";
    } else {
        cout << "Top-K result does not match the host reference.\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}