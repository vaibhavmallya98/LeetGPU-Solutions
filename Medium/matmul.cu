#include "solve.h"
#include <cuda_runtime.h>
/*
//Naive Matrix multiplication without shared memory - 971ms - very slow 
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    float Cvalue = 0;

    //row number is along the height which is the y axis
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //column number is along the width which is the x axis 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < K){
        for (int e = 0; e < N; ++e){
            Cvalue += A[row * N + e] * B[e * K + col];
        }
        C[row * K + col] = Cvalue;
    }
}
*/ 

#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}
// Thread block size


//Matrix Multiplication using Shared Memory - Expected to be faster 
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix A_Matrix; 

    A_Matrix.width = N; 
    A_Matrix.height = M; 
    A_Matrix.stride = N; 
    A_Matrix.elements = (float*)A;

    Matrix B_Matrix; 

    B_Matrix.width = K; 
    B_Matrix.height = N; 
    B_Matrix.stride = K; 
    B_Matrix.elements = (float*)B; 

    Matrix C_Matrix; 

    C_Matrix.width = K; 
    C_Matrix.height = M; 
    C_Matrix.stride = K; 
    C_Matrix.elements = C; 

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C_Matrix, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        Matrix Asub = GetSubMatrix(A_Matrix, blockRow, m);
        Matrix Bsub = GetSubMatrix(B_Matrix, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Add bounds checking for A
        As[row][col] = (row < M && col < N) ? 
                                      GetElement(Asub, row, col) : 0.0f;

        // Add bounds checking for B
        Bs[row][col] = (row < N && col < K) ? 
                                      GetElement(Bsub, row, col) : 0.0f;

        __syncthreads();

        // Compute only valid elements
        if (row < Csub.height && col < Csub.width) {
            for (int e = 0; e < BLOCK_SIZE; ++e) {
                Cvalue += As[row][e] * Bs[e][col];
            }
        }
        __syncthreads();
    }

    // Add output bounds check
    if (row < M && col < K) {
        SetElement(Csub, row, col, Cvalue);
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
