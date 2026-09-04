
**Kernel Fusion**

* Avoid writing the intermediate $M \times N$ matrix to global memory entirely. By computing blocks of the output matrix and keeping the intermediate softmax results in shared memory (similar to the FlashAttention algorithm), you eliminate the most expensive global memory bottlenecks.

**Parallel Reduction for Max Value**

* Since you are planning to use a device function with parallel reduction for the softmax sum, apply the exact same technique to find the row maximum. A sequential loop for the max value will bottleneck your kernel and leave most threads in the block idle if $N$ is large.

**Shared Memory Transpose**

* Your current transpose kernel suffers from uncoalesced global memory writes. Load a 2D tile of matrix $K$ into shared memory, call `__syncthreads()`, and write it back to global memory column-wise using contiguous thread IDs to ensure fully coalesced memory accesses.

**Vectorized Memory Accesses**

* Cast your memory pointers to `float4` to process four elements at a time where memory access is contiguous. Fetching 128 bits per instruction saturates the memory bus much more efficiently than individual 32-bit loads.

**Fast Math Intrinsics**

* Swap standard math functions for their hardware-accelerated intrinsic counterparts. Use `__expf(x)` instead of `expf(x)` and `__fdividef(x, y)` instead of standard division for a significant throughput boost at the cost of a negligible drop in precision.