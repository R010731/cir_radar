#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void transposeMatrix(cuComplex* input, cuComplex* output, int row_old, int col_old);
void transpose(cuComplex* d_input, cuComplex* d_output, int row_old, int col_old);

