#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cuComplex.h>
#include <complex>
#include <Eigen/Dense>

using namespace::std;
using namespace::Eigen;

typedef Matrix<complex<float>, 1,           Dynamic,        RowMajor>       MatrixDDC_chn;
typedef Matrix<complex<float>, Dynamic,     Dynamic,        RowMajor>       MatrixDDC_sector;
typedef Matrix<complex<float>, 1,           Dynamic,        RowMajor>       MatrixDBF_coeff;
typedef Matrix<complex<float>, Dynamic,     Dynamic,        RowMajor>       MatrixDBF_beam;
typedef Map<MatrixDDC_chn> MapDDC_chn;
typedef Map<MatrixDBF_beam> MapDDC_beam;



// void gpu_matrix_multiply_cublas_complex(MatrixDBF_coeff& A, MatrixDDC_sector& B, MapDDC_beam& C, int A_rows, int A_cols, int B_cols,int prf_index);
void gpu_matrix_multiply_cublas_complex(MatrixDBF_coeff& A, MatrixDDC_sector& B, cuComplex* d_C, int A_rows, int A_cols, int B_cols, int prf_index,int offset);
extern "C" void complexAdd(cuComplex* A, const cuComplex* B, int N, cudaStream_t &stream);