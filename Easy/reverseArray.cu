#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if(idx < N/2){
        float temp = input[idx];
        input[idx] = input[N - idx - 1]; 
        input[N - idx - 1] = temp; 
    }

}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
