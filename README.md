# LeetGPU Daily

A hands-on collection of CUDA and GPU-optimization problems solved in C++/CUDA. This repository focuses on learning core GPU programming concepts through practical implementations such as reductions, convolutions, softmax, attention, matrix multiplication, and top-k selection.

## Overview

Each directory contains a standalone CUDA program and, in many cases, a short problem note or improvement note. The goal is to practice:

- GPU memory layout and coalescing
- Shared memory usage
- Parallel reduction patterns
- Kernel design and launch configuration
- Common deep learning kernels on CUDA
- Performance-oriented code restructuring

## Repository structure

- 2D-Convolution/ — 2D convolution implementation
- 3D-Convolution/ — 3D convolution example
- CategoricalCrossEntropyLoss/ — categorical cross-entropy loss kernel
- DotProduct/ — dot product implementation and reduction-based optimization
- GEMM/ — matrix multiplication on GPU
- GuassianBlur/ — Gaussian blur filter
- Histogramming/ — histogram generation
- MeanSquaredError/ — MSE loss implementation
- Prefix_sum/ — prefix sum / scan example
- Reduction/ — reduction patterns and optimization ideas
- SoftMax/ — softmax kernels, including max-trick variants
- SoftMaxAttention/ — attention-related implementation notes and code
- SPMV/ — sparse matrix-vector multiplication
- TopKSelections/ — top-k selection using GPU sorting / selection logic

## Prerequisites

- NVIDIA GPU
- CUDA Toolkit installed
- NVCC compiler available in PATH

Verify your environment with:

```bash
nvcc --version
```

## Build and run a CUDA example

From the repository root, compile a specific example with NVCC:

```bash
nvcc TopKSelections/TopKSelections.cu -o topk
./topk
```

You can similarly compile any other example:

```bash
nvcc DotProduct/dotproduct.cu -o dotproduct
./dotproduct
```

## Typical workflow

1. Read the problem note or implementation notes in the folder.
2. Understand the data flow and parallelism opportunities.
3. Implement the CUDA kernel(s).
4. Compare against a CPU reference or expected output.
5. Refine memory access and reduction strategy for performance.

## Notes

- Some folders contain only the CUDA source file, while others include a problem description for the exercise.
- This repository is intended as an educational and practice-oriented collection rather than a production-ready library.
- A few examples demonstrate both naive and improved versions to highlight optimization techniques.

## License

This project is for learning and experimentation. Use it freely for personal study and GPU programming practice.
