#include "cuda_abs.cuh"


extern "C" void MTD_abs(cuComplex* d_i_data,float* d_o_data,int count,cudaStream_t &stream){
    dim3 abs_threadsPerBlock(256);
    dim3 abs_numBlocks((count + abs_threadsPerBlock.x - 1) / abs_threadsPerBlock.x);
    abs_kernel << <abs_numBlocks, abs_threadsPerBlock, 0, stream >> > (d_i_data, d_o_data, count);
}