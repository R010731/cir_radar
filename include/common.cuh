#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <time.h>


cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber);
void checkCufftError(cufftResult result) ;
void setGPU();
__global__ void filter_kernel(cufftComplex* input, int input_length, float* filter, int filter_length, cufftComplex* output);
__global__ void decimate_kernel(cufftComplex* input, cufftComplex* output, int input_len, int M) ;
__global__ void complexMultiply(const cuFloatComplex* a, const cuFloatComplex* b, cuFloatComplex* result, int numSegments, int SampleNumber);
__global__ void complexAdd_Kernel(cuComplex* A, const cuComplex* B, int N);
__global__ void multiplyArrays(float* a, float* b, float* c, cufftComplex* d, int size);
__global__ void complexMultiplyfloat(const cuFloatComplex* a, const float* b, cuFloatComplex* result, int numSegments, int SampleNumber);
__global__ void abs_kernel(cufftComplex* idata1, float* odata, int num);
__global__ void CA_CFAR_kernel(float* Sig, int Range_gate_length, int Velocity_gate_length,
    int Num_range_gate_ref, int Num_velocity_gate_ref,
    int Num_range_gate_protect, int Num_velocity_gate_protect,
    float Threshold, float* CFAR_matrix, int* Flag_matrix) ;
__global__ void subtractMti(cufftComplex* a, cufftComplex* b, int numRows, int numCols);
void writeComplexToFile(const char* filename, cufftComplex* d_data, size_t size) ;
__global__ void TargetsCohensionKernel(const float* CFAR_matrix, float* Cohension_matrix,int Velocity_gate_length, int Range_gate_length) ;
void writeDoubleToFile(const char* filename, double* d_data, size_t size) ;
void writeIntToFile(const char* filename, int* d_data, size_t size) ;
void writeFloatToFile(const char* filename, float* d_data, size_t size) ;
long long getTimeInMicroseconds();