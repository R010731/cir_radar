#include "cuda_transpose.cuh"

// CUDA 核函数：进行矩阵转置

__global__ void transposeMatrix(cuComplex* input, cuComplex* output, int row_old, int col_old){
    // 计算当前线程处理的元素的行和列
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 线程的全局索引
    if (idx < row_old * col_old) {
        int row = idx / col_old;
        int col = idx % col_old;

        // 转置：将输入矩阵 (row, col) 的元素放到输出矩阵 (col, row)
        output[col * row_old + row] = input[row * col_old + col];
    }
}

void transpose(cuComplex* d_input, cuComplex* d_output, int row_old, int col_old) {
    // 定义每个块和线程的维度
    dim3 blockSize(256);  // 每个块的线程数（256个线程）

    // 网格配置：根据数据的大小来计算网格数
    dim3 gridSize((row_old * col_old + blockSize.x - 1) / blockSize.x);

    // 调用转置核函数
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, row_old, col_old);

    // 检查错误
    cudaDeviceSynchronize();
}