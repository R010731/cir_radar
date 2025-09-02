#pragma once
#include "common.cuh"
#include <cuComplex.h>


extern "C" void MTD_abs(cuComplex* d_i_data,float* d_o_data,int count,cudaStream_t &stream);