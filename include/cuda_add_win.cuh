#pragma once
#include <cuComplex.h>
#include "common.cuh"


extern "C" void add_win(cuComplex* d_i_data1,float *d_i_data2,cuComplex* d_o_data,int d_i_data1_length,int d_i_data2_length,cudaStream_t &stream);