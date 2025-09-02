#include "cuda_pc.cuh"

// 复数信号GPU相乘
// 输入a，b
// a更长,C接收
extern "C" void pc_complexMultiply(cuComplex *d_a, cuComplex *d_b, cuComplex *d_c, const int a_length, const int b_length, cudaStream_t &stream){
    dim3 complexMultiply_threadsPerBlock(256);
    dim3 complexMultiply_numBlocks((a_length + complexMultiply_threadsPerBlock.x - 1) / complexMultiply_threadsPerBlock.x);

    // 将流参数传递给内核
    complexMultiply <<<complexMultiply_numBlocks, complexMultiply_threadsPerBlock, 0, stream>>> (d_a, d_b, d_c, a_length, b_length);
}


// class cuda_pc{
// private:
//     cufftHandle pc_narrow, pc_wide;
//     cuComplex *d_narrow_coeff, *d_wide_coeff;
//     cuComplex *d_narrow_sum_step1, *d_narrow_diff_step1;
//     cuComplex *d_wide_sum_step1, *d_wide_diff_step1;
//     cuComplex *d_narrow_sum_step2, *d_narrow_diff_step2;
//     cuComplex *d_wide_sum_step2, *d_wide_diff_step2;
//     cuComplex *d_narrow_sum_step3, *d_narrow_diff_step3;
//     cuComplex *d_wide_sum_step3, *d_wide_diff_step3;
//     int narrow_pulse_pc_fft_points;
//     int wide_pulse_pc_fft_points;
//     int prf_total_pulse;

// public:
//     cuda_pc(size_t narrow_points, size_t wide_points, int pulses)
//         : narrow_pulse_pc_fft_points(narrow_points), wide_pulse_pc_fft_points(wide_points), prf_total_pulse(pulses) {
//         // 初始化CUDA资源
//         cudaMalloc((void**)&d_narrow_coeff, narrow_pulse_pc_fft_points * sizeof(cuComplex));
//         cudaMalloc((void**)&d_wide_coeff, wide_pulse_pc_fft_points * sizeof(cuComplex));
//         // 初始化其他内存资源...
//     }

//     ~cuda_pc() {
//         // 清理CUDA资源
//         cudaFree(d_narrow_coeff);
//         cudaFree(d_wide_coeff);
//         cudaFree(d_narrow_sum_step1);
//         cudaFree(d_narrow_diff_step1);
//         cudaFree(d_wide_sum_step1);
//         cudaFree(d_wide_diff_step1);
//         cudaFree(d_narrow_sum_step2);
//         cudaFree(d_narrow_diff_step2);
//         cudaFree(d_wide_sum_step2);
//         cudaFree(d_wide_diff_step2);
//         cudaFree(d_narrow_sum_step3);
//         cudaFree(d_narrow_diff_step3);
//         cudaFree(d_wide_sum_step3);
//         cudaFree(d_wide_diff_step3);
//         cufftDestroy(pc_narrow);
//         cufftDestroy(pc_wide);
//     }

    
//     void initializeFFTPlans() {
//         cufftPlan1d(&pc_narrow, narrow_pulse_pc_fft_points, CUFFT_C2C, prf_total_pulse);
//         cufftPlan1d(&pc_wide, wide_pulse_pc_fft_points, CUFFT_C2C, prf_total_pulse);
//     }

//     void executeFFT() {
//         // 计算FFT
//         cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step1, (cufftComplex*)d_narrow_sum_step2, CUFFT_FORWARD);
//         cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step1, (cufftComplex*)d_narrow_diff_step2, CUFFT_FORWARD);
//         cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step1, (cufftComplex*)d_wide_sum_step2, CUFFT_FORWARD);
//         cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step1, (cufftComplex*)d_wide_diff_step2, CUFFT_FORWARD);
//     }

//     void multiplySignal() {
//         // 对FFT后的信号进行相乘
//         pc_complexMultiply(d_narrow_sum_step2, d_narrow_coeff, d_narrow_sum_step3, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points);
//         pc_complexMultiply(d_narrow_diff_step2, d_narrow_coeff, d_narrow_diff_step3, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points);
//         pc_complexMultiply(d_wide_sum_step2, d_wide_coeff, d_wide_sum_step3, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points);
//         pc_complexMultiply(d_wide_diff_step2, d_wide_coeff, d_wide_diff_step3, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points);
//     }

//     void executeInverseFFT() {
//         // 计算逆FFT
//         cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step3, (cufftComplex*)d_narrow_sum_step1, CUFFT_INVERSE);
//         cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step3, (cufftComplex*)d_narrow_diff_step1, CUFFT_INVERSE);
//         cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step3, (cufftComplex*)d_wide_sum_step1, CUFFT_INVERSE);
//         cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step3, (cufftComplex*)d_wide_diff_step1, CUFFT_INVERSE);
//     }

//     // void process(cpi_data_t &cpi_data, cuComplex &)
//     // 更多操作，比如数据的复制、文件保存等
// };