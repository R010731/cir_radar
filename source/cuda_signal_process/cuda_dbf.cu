#include "cuda_dbf.cuh"
#include <cstdio>
#include "iostream"
#include "common.cuh"

// void gpu_matrix_multiply_cublas_complex(MatrixDBF_coeff& A, MatrixDDC_sector& B, MapDDC_beam& C, int A_rows, int A_cols, int B_cols,int prf_index)
// {
//     cublasHandle_t handle;
//     cublasCreate(&handle);  // 创建cuBLAS句柄
//     // cout<<"A_rows:"<< A_rows << " A_cols:"<<A_cols<<" B_cols:"<<B_cols<<endl;
//     // alpha 用于对矩阵乘法的结果进行缩放，计算时，C = alpha * A * B + beta * C。
//     // beta 是用来缩放矩阵 C 原本的值，默认通常是 0.0f，表示不考虑 C 原本的内容（即仅计算 A * B）。

//     cuComplex alpha = make_cuComplex(1.0f, 0.0f);  // 矩阵乘法的标量因子 (实部: 1.0f, 虚部: 1.0f)
//     cuComplex beta = make_cuComplex(0.0f, 0.0f);   // 矩阵C的初始值标量因子

//     cuComplex *d_A, *d_B, *d_C;
    
//     // 为矩阵分配GPU内存
//     cudaMalloc((void**)&d_A, A_rows * A_cols * sizeof(cuComplex));
//     cudaMalloc((void**)&d_B, A_cols * B_cols * sizeof(cuComplex));
//     cudaMalloc((void**)&d_C, A_rows * B_cols * sizeof(cuComplex));

//     // 将数据从主机复制到设备
//     cudaMemcpy(d_A, A.data(), A_rows * A_cols * sizeof(cuComplex), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B.data(), A_cols * B_cols * sizeof(cuComplex), cudaMemcpyHostToDevice);
//     // 执行矩阵乘法：C = alpha * A * B + beta * C
//     cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, A_rows, B_cols, A_cols,
//                 &alpha, d_A, A_cols, d_B, B_cols, &beta, d_C, A_rows);
//     // 将计算结果从GPU内存复制回主机

//     cudaMemcpy(C.data() + prf_index * A_rows * B_cols, d_C, A_rows * B_cols * sizeof(cuComplex), cudaMemcpyDeviceToHost);

//     // 释放GPU内存
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     // 销毁cuBLAS句柄
//     cublasDestroy(handle);
// }

// offset 是偏移数
void gpu_matrix_multiply_cublas_complex(MatrixDBF_coeff& A, MatrixDDC_sector& B, cuComplex* d_C, int A_rows, int A_cols, int B_cols, int prf_index,int offset)
{
    cublasHandle_t handle;
    cublasCreate(&handle);  // 创建cuBLAS句柄

    // alpha 用于对矩阵乘法的结果进行缩放，计算时，C = alpha * A * B + beta * C。
    // beta 是用来缩放矩阵 C 原本的值，默认通常是 0.0f，表示不考虑 C 原本的内容（即仅计算 A * B）。

    cuComplex alpha = make_cuComplex(1.0f, 0.0f);  // 矩阵乘法的标量因子 (实部: 1.0f, 虚部: 1.0f)
    cuComplex beta = make_cuComplex(0.0f, 0.0f);   // 矩阵C的初始值标量因子

    cuComplex *d_A, *d_B;
    
    // 为矩阵分配GPU内存
    cudaMalloc((void**)&d_A, A_rows * A_cols * sizeof(cuComplex));
    cudaMalloc((void**)&d_B, A_cols * B_cols * sizeof(cuComplex));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A.data(), A_rows * A_cols * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), A_cols * B_cols * sizeof(cuComplex), cudaMemcpyHostToDevice);
    
    // 执行矩阵乘法：C = alpha * A * B + beta * C
    cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, A_rows, B_cols, A_cols,
                &alpha, d_A, A_cols, d_B, B_cols, &beta, d_C + prf_index*offset, A_rows);
    
    // 释放GPU内存
    cudaFree(d_A);
    cudaFree(d_B);

    // 销毁cuBLAS句柄
    cublasDestroy(handle);
}


extern "C" void complexAdd(cuComplex* A, const cuComplex* B, int N, cudaStream_t &stream){
    dim3 complexAdd_threadsPerBlock(256);
    dim3 complexAdd_numBlocks((N + complexAdd_threadsPerBlock.x - 1) / complexAdd_threadsPerBlock.x);
    complexAdd_Kernel << <complexAdd_numBlocks, complexAdd_threadsPerBlock, 0, stream >> > (A, B, N);
}