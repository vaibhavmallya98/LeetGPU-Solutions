#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
__global__ void copy_matrix_kernel(const float* A, float* B, int N) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;//col 
    int idy = blockDim.y * blockIdx.y + threadIdx.y;//row
    
    if(idx < N && idy < N){
        B[idy * N + idx] = A[idy * N + idx]; 
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, float* B, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
                       
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
} 
