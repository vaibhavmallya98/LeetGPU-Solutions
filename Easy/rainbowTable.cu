#include "solve.h"
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    unsigned int hash = OFFSET_BASIS;
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if(idx < N){
        int hash_input = input[idx]; 
        for(int j = 0; j < R; j++){
            output[idx] = fnv1a_hash(hash_input);
            hash_input = output[idx]; 
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
