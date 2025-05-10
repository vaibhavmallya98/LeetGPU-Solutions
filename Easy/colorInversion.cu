#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if(idx < width * height ){  
        image[4 * idx] = 255 - image[4 * idx]; 
        image[4 * idx + 1] = 255 - image[4 * idx + 1];
        image[4 * idx + 2] = 255 - image[4 * idx + 2]; 
    }

}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
