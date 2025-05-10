#include "solve.h"
#include <cuda_runtime.h>


#define BLOCK_SIZE 8
//#define BDIMY 4
//#define BDIMX 4

//NAIVE TRANSPOSE IMPLEMENTATION 

/*
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y; 

    if(ix < cols && iy < rows){
        output[ix * rows + iy] = input[iy * cols + ix]; 
    }

}
*/ 


//MATRIX TRANSPOSE USING SHARED MEMORY 
//Execution Time = 1.86ms
//0.58ms faster than naive matrix transpose  
#define TILE_DIM 8
//#define BLOCK_ROWS 8

__global__ void matrix_transpose_kernel(const float* input, float* output, 
                                       int rows, int cols) {
    __shared__ float tile[TILE_DIM * TILE_DIM];  
    
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int tileX = threadIdx.x;
    int tileY = threadIdx.y; 

    if(x < cols && y < rows){
        tile[tileY * TILE_DIM + tileX] = input[y * cols + x]; 
    }

    __syncthreads(); 

    int outX = blockIdx.y * TILE_DIM + threadIdx.x;
    int outY = blockIdx.x * TILE_DIM + threadIdx.y;

    if(outX < rows && outY < cols){
        output[outY * rows + outX] = tile[tileX * TILE_DIM + tileY]; 
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
