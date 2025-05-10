#include "solve.h"
#include <cuda_runtime.h>

__global__ void reduction_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float mysum = 0.0f;
    float blocksum = 0.0f; 
    if (i < N) 
        mysum = input[i];
    sdata[tid] = mysum;
    __syncthreads();
    // block reduce
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) 
            sdata[tid] += sdata[tid+s];
        __syncthreads();
    }

    if (tid < 32) {
        if (blockDim.x >= 64) 
            blocksum = sdata[tid] + sdata[tid + 32];
        else 
            blocksum = sdata[tid];
        
        // Use warp shuffle to reduce
        for (int offset = 16; offset > 0; offset /= 2)
            blocksum += __shfl_down_sync(0xffffffff, blocksum, offset);
    }
    
    if (tid == 0) 
        output[blockIdx.x] = blocksum;
}

void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_sum;
    //cudaMalloc(&d_max, blocks * sizeof(float));
    cudaMalloc(&d_sum, blocksPerGrid * sizeof(float));

    reduction_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, d_sum, N);

    float* h_sum = new float[blocksPerGrid];
    cudaMemcpy(h_sum, d_sum, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    

    // Kahan summation on host
    float sum = 0.0f;
    float c = 0.0f; // Compensation term
    for (int i = 0; i < blocksPerGrid; ++i) {
        float y = h_sum[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    cudaMemcpy(output, &sum, sizeof(float), cudaMemcpyHostToDevice);

    //Free allocated memory
    cudaFree(d_sum);
    delete[] h_sum;
}
