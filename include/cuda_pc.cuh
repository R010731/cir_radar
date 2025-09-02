#pragma once
#include "common.cuh"
#include "rsl_data_structure.hpp"
extern "C" void pc_complexMultiply(cuComplex *d_a, cuComplex *d_b, cuComplex *d_c, const int a_length, const int b_length, cudaStream_t &stream);