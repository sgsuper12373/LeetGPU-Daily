#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void least_squares_qr_kernel( const float* X, const float* y, float* beta, double* matrix, double* rhs, int* permutation, int n_samples, int n_features) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int row = 0; row < n_samples; ++row) {
        rhs[row] = static_cast<double>(y[row]);
        for (int col = 0; col < n_features; ++col) {
            matrix[static_cast<size_t>(row) * n_features + col] =
                static_cast<double>(X[static_cast<size_t>(row) * n_features + col]);
        }
    }
    for (int col = 0; col < n_features; ++col) {
        permutation[col] = col;
    }

    const int steps = n_features;
    double first_diagonal = 0.0;
    int rank = steps;

    for (int k = 0; k < steps; ++k) {
        int pivot_col = k;
        double largest_norm_squared = -1.0;
        for (int col = k; col < n_features; ++col) {
            double norm_squared = 0.0;
            for (int row = k; row < n_samples; ++row) {
                const double value = matrix[static_cast<size_t>(row) * n_features + col];
                norm_squared += value * value;
            }
            if (norm_squared > largest_norm_squared) {
                largest_norm_squared = norm_squared;
                pivot_col = col;
            }
        }

        if (pivot_col != k) {
            for (int row = 0; row < n_samples; ++row) {
                const size_t row_offset = static_cast<size_t>(row) * n_features;
                const double temp = matrix[row_offset + k];
                matrix[row_offset + k] = matrix[row_offset + pivot_col];
                matrix[row_offset + pivot_col] = temp;
            }
            const int temp = permutation[k];
            permutation[k] = permutation[pivot_col];
            permutation[pivot_col] = temp;
        }

        double column_norm = 0.0;
        for (int row = k; row < n_samples; ++row) {
            const double value = matrix[static_cast<size_t>(row) * n_features + k];
            column_norm = hypot(column_norm, value);
        }
        if (k == 0) {
            first_diagonal = column_norm;
        }
        const double tolerance =
            first_diagonal * static_cast<double>(n_samples > n_features ? n_samples : n_features) * 1.1920928955078125e-7;
        if (column_norm <= tolerance) {
            rank = k;
            break;
        }

        const size_t diagonal_index = static_cast<size_t>(k) * n_features + k;
        const double original_diagonal = matrix[diagonal_index];
        const double alpha = -copysign(column_norm, original_diagonal);
        const double leading_reflector = original_diagonal - alpha;
        const double tau = (alpha - original_diagonal) / alpha;
        matrix[diagonal_index] = alpha;
        for (int row = k + 1; row < n_samples; ++row) {
            matrix[static_cast<size_t>(row) * n_features + k] /= leading_reflector;
        }

        for (int col = k + 1; col < n_features; ++col) {
            double projection = matrix[static_cast<size_t>(k) * n_features + col];
            for (int row = k + 1; row < n_samples; ++row) {
                projection += matrix[static_cast<size_t>(row) * n_features + k] *
                              matrix[static_cast<size_t>(row) * n_features + col];
            }
            projection *= tau;
            matrix[static_cast<size_t>(k) * n_features + col] -= projection;
            for (int row = k + 1; row < n_samples; ++row) {
                matrix[static_cast<size_t>(row) * n_features + col] -=
                    matrix[static_cast<size_t>(row) * n_features + k] * projection;
            }
        }

        double rhs_projection = rhs[k];
        for (int row = k + 1; row < n_samples; ++row) {
            rhs_projection += matrix[static_cast<size_t>(row) * n_features + k] * rhs[row];
        }
        rhs_projection *= tau;
        rhs[k] -= rhs_projection;
        for (int row = k + 1; row < n_samples; ++row) {
            rhs[row] -= matrix[static_cast<size_t>(row) * n_features + k] * rhs_projection;
        }
    }

    for (int col = 0; col < n_features; ++col) {
        beta[col] = 0.0f;
    }
    for (int row = rank - 1; row >= 0; --row) {
        double value = rhs[row];
        for (int col = row + 1; col < rank; ++col) {
            value -= matrix[static_cast<size_t>(row) * n_features + col] *
                     static_cast<double>(beta[permutation[col]]);
        }
        beta[permutation[row]] = static_cast<float>(
            value / matrix[static_cast<size_t>(row) * n_features + row]);
    }
}

extern "C" void solve( const float* X, const float* y, float* beta, int n_samples, int n_features) {
    if (n_features <= 0) {
        return;
    }
    if (n_samples <= 0 || n_samples < n_features) {
        cudaMemset(beta, 0, static_cast<size_t>(n_features) * sizeof(float));
        return;
    }

    double* matrix = nullptr;
    double* rhs = nullptr;
    int* permutation = nullptr;
    cudaMalloc(&matrix, static_cast<size_t>(n_samples) * n_features * sizeof(double));
    cudaMalloc(&rhs, static_cast<size_t>(n_samples) * sizeof(double));
    cudaMalloc(&permutation, static_cast<size_t>(n_features) * sizeof(int));

    least_squares_qr_kernel<<<1, 1>>>(
        X, y, beta, matrix, rhs, permutation, n_samples, n_features);
    cudaDeviceSynchronize();

    cudaFree(matrix);
    cudaFree(rhs);
    cudaFree(permutation);
}

int main() {
    constexpr int n_samples = 8;
    constexpr int n_features = 3;
    const float expected[n_features] = {2.0f, -0.5f, 1.25f};
    std::vector<float> X(n_samples * n_features);
    std::vector<float> y(n_samples);

    for (int row = 0; row < n_samples; ++row) {
        const float a = static_cast<float>(row + 1);
        const float b = static_cast<float>((row * row) % 7 - 3);
        X[row * n_features] = 1.0f;
        X[row * n_features + 1] = a;
        X[row * n_features + 2] = b;
        y[row] = expected[0] + expected[1] * a + expected[2] * b;
    }

    float* device_X = nullptr;
    float* device_y = nullptr;
    float* device_beta = nullptr;
    float result[n_features] = {};
    cudaMalloc(&device_X, X.size() * sizeof(float));
    cudaMalloc(&device_y, y.size() * sizeof(float));
    cudaMalloc(&device_beta, n_features * sizeof(float));
    cudaMemcpy(device_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice);

    solve(device_X, device_y, device_beta, n_samples, n_features);
    cudaMemcpy(result, device_beta, sizeof(result), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int col = 0; col < n_features; ++col) {
        std::printf("beta[%d] = %.6f (expected %.6f)\n", col, result[col], expected[col]);
        passed = passed && std::fabs(result[col] - expected[col]) < 1e-4f;
    }

    cudaFree(device_X);
    cudaFree(device_y);
    cudaFree(device_beta);
    return passed ? 0 : 1;
}
