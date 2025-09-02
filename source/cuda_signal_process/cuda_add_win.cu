#include "cuda_add_win.cuh"

extern "C" void add_win(cuComplex* d_i_data1,float *d_i_data2,cuComplex* d_o_data,int d_i_data1_length,int d_i_data2_length,cudaStream_t &stream){
    dim3 add_win_threadsPerBlock(256);
    dim3 add_win_numBlocks((d_i_data1_length + add_win_threadsPerBlock.x - 1) / add_win_threadsPerBlock.x);
    complexMultiplyfloat << <add_win_numBlocks, add_win_threadsPerBlock,0, stream>> > (d_i_data1,d_i_data2, d_o_data, d_i_data1_length,d_i_data2_length);
}