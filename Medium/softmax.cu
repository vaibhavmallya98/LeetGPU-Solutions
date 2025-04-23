#include <cuda_runtime.h>
#include <cmath>

// 1st pass: reduce max using warp shuffles
__global__ void reduce_max(const float* input, float* max_out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Each thread loads data
    float val = (i < N) ? input[i] : -INFINITY;

    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, tmp);
    }

    // Store warp results in shared memory
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // Final block reduction using first warp
    if (tid < 32) {
        float block_max = (tid < blockDim.x/32) ? sdata[tid] : -INFINITY;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            float tmp = __shfl_down_sync(0xffffffff, block_max, offset);
            block_max = fmaxf(block_max, tmp);
        }

        if (tid == 0) {
            max_out[blockIdx.x] = block_max;
        }
    }
}

// 2nd pass: reduce exp sum using warp shuffles
__global__ void reduce_expsum(const float* input, float* sum_out, int N, float maxval) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Each thread computes exp
    float val = (i < N) ? expf(input[i] - maxval) : 0.0f;

    // Warp-level sum reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Store warp results in shared memory
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // Final block reduction using first warp
    if (tid < 32) {
        float block_sum = (tid < blockDim.x/32) ? sdata[tid] : 0.0f;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }

        if (tid == 0) {
            sum_out[blockIdx.x] = block_sum;
        }
    }
}

// (softmax_final remains unchanged)


// 3rd pass: final softmax
__global__ void softmax_final(const float* input, float* output, int N, float maxval, float sumval) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        output[i] = expf(input[i] - maxval) / sumval;
}

void solve(const float* input, float* output, int N) {
    int threads = 512;
    int blocks = (N + threads - 1) / threads;

    size_t smem_size = ((threads + 31)/32) * sizeof(float); 

    float *d_max, *d_sum;
    cudaMallocManaged(&d_max, blocks * sizeof(float));
    cudaMallocManaged(&d_sum, blocks * sizeof(float));

    // 1. Reduce max
    reduce_max<<<blocks, threads, smem_size>>>(input, d_max, N);
    float* h_max = new float[blocks];
    cudaMemcpy(h_max, d_max, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float maxval = -INFINITY;
    for (int i = 0; i < blocks; ++i) 
        maxval = fmaxf(maxval, h_max[i]);

    // 2. Reduce exp sum
    reduce_expsum<<<blocks, threads, smem_size>>>(input, d_sum, N, maxval);
    float* h_sum = new float[blocks];
    cudaMemcpy(h_sum, d_sum, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float sumval = 0.0f;
    for (int i = 0; i < blocks; ++i) 
        sumval += h_sum[i];

    // 3. Final softmax
    softmax_final<<<blocks, threads>>>(input, output, N, maxval, sumval);

    cudaFree(d_max); cudaFree(d_sum);
    delete[] h_max; delete[] h_sum;
}
