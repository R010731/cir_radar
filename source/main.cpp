#include "cuda_signal_process.hpp"
#include <spdlog/spdlog.h>


int main(){
    setGPU();
    uint32_t* ram_ptr;
    //设置进程优先级最高；
    int retval = nice(-20);
    int cal_count = 0;    
    // 文件读取指针的位置    
    streampos current_pos;

    // 数据指针
    uint32_t* cpi_data_begin = nullptr;


    // cuda流创建，DBF句柄创建，PC句柄创建，MTD句柄创建
    cudaStream_t streams[2];
    cublasHandle_t handle_dbf[2];
    // pc句柄 6个句柄，0和1是一号波束的宽窄，2和3是二号波束的宽窄，4和5是三号波束的宽窄
    // MTD句柄 6个句柄，0和1是一号波束的宽窄，2和3是二号波束的宽窄，4和5是三号波束的宽窄
    cufftHandle pc_handle[6];

    // MTD句柄
    cufftHandle MTD_handle[6];

    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&streams[i]);               // 创建CUDA流
        cublasCreate(&handle_dbf[i]);                // 创建CUBLAS句柄
        cublasSetStream(handle_dbf[i], streams[i]);  // 将CUBLAS句柄与流关联
    }

    int prf_total_pulse = 2048;
    int wide_pulse_pc_fft_points = 512;
    int narrow_pulse_pc_fft_points = 256;
    checkCufftError(cufftPlan1d(&pc_handle[0], wide_pulse_pc_fft_points, CUFFT_C2C, prf_total_pulse));
    checkCufftError(cufftPlan1d(&pc_handle[1], narrow_pulse_pc_fft_points, CUFFT_C2C, prf_total_pulse));
    checkCufftError(cufftSetStream(pc_handle[0], streams[0]));
    checkCufftError(cufftSetStream(pc_handle[1], streams[1]));

    // MTD参数设置
    // 在这里设置避免等待fft句柄创建时阻塞，重叠数据迁移与CPU操作
    int narrow_points = 144;
    int wide_points = 320;

    int MTD_narrow_rank = 1;   //  维
    int MTD_narrow_n[1] = { prf_total_pulse };    // 每一维变换数目
    int MTD_narrow_inembed[2] = { narrow_pulse_pc_fft_points, prf_total_pulse }; // 输入数据的步幅
    int MTD_narrow_istride = narrow_pulse_pc_fft_points; // 输入每个 FFT 的数据步幅
    int MTD_narrow_idist = 1; // 不同 FFT 之间的距离
    int MTD_narrow_onembed[2] = { prf_total_pulse, narrow_points }; // 输出数据的步幅
    int MTD_narrow_ostride = narrow_points; // 输出每个 FFT 的数据步幅
    int MTD_narrow_odist = 1; //
    int MTD_narrow_batch = narrow_points;   // narrow_pulse_pc_fft_points次FFT

    int MTD_wide_rank = 1;   //  维
    int MTD_wide_n[1] = { prf_total_pulse };    // 每一维变换数目
    int MTD_wide_inembed[2] = { wide_pulse_pc_fft_points, prf_total_pulse }; // 输入数据的步幅
    int MTD_wide_istride = wide_pulse_pc_fft_points; // 输入每个 FFT 的数据步幅
    int MTD_wide_idist = 1; // 不同 FFT 之间的距离
    int MTD_wide_onembed[2] = { prf_total_pulse, wide_points }; // 输出数据的步幅
    int MTD_wide_ostride = wide_points; // 输出每个 FFT 的数据步幅
    int MTD_wide_odist = 1; //
    int MTD_wide_batch = wide_points;   // wide_pulse_pc_fft_points次FFT

    checkCufftError(cufftPlanMany(&MTD_handle[0], MTD_wide_rank, MTD_wide_n, MTD_wide_inembed, MTD_wide_istride, MTD_wide_idist, MTD_wide_onembed, MTD_wide_ostride, MTD_wide_odist, CUFFT_C2C, MTD_wide_batch));
    checkCufftError(cufftPlanMany(&MTD_handle[1], MTD_narrow_rank, MTD_narrow_n, MTD_narrow_inembed, MTD_narrow_istride, MTD_narrow_idist, MTD_narrow_onembed, MTD_narrow_ostride, MTD_narrow_odist, CUFFT_C2C, MTD_narrow_batch));

    checkCufftError(cufftSetStream(MTD_handle[0], streams[0]));
    checkCufftError(cufftSetStream(MTD_handle[1], streams[1]));


    while(cal_count < 2){
        cal_count++;
        // 读取文件，解析数据
        cpi_data_begin = data_read(current_pos);

        // 设置读取第几个CPI
        // if (cal_count < 1){
        //     continue;
        // }
        // data_read(current_pos, cpi_data_begin);
        // -------------------构建cpi数据-----------------------------//
        // 找到ddc_data起始位置
        cpi_data_t cpi_data(cpi_data_begin);
        uint32_t* ddc_data_begin;
        // 保留原始指针，cudaFreeHost
        ddc_data_begin = cpi_data_begin;
        ddc_data_begin += CMD_PARAMS_LEN / 4;
        #if STATUS_INFO_VALID == 1
                ddc_data_begin += STATUS_INFO_LEN / 4;
        #endif
        ddc_data_begin += SELF_TEST_LEN / 4;
        long long data_transfer_begin = getTimeInMicroseconds();
        cuComplex* ddc_data_begin_64 = (cuComplex*) ddc_data_begin;
        int debug = 0;

        // ddc数据迁移
        // 参数获取
        int prf_total_pulse = cpi_data.cmd_params->pulse_params.prf_total_pulse;
        int narrow_pulse_pc_fft_points = cpi_data.cmd_params->PC_params.narrow_pulse_pc_fft_points;
        int wide_pulse_pc_fft_points = cpi_data.cmd_params->PC_params.wide_pulse_pc_fft_points;
        int narrow_points = cpi_data.cmd_params->pulse_params.narrow_pulse_valid_point/8;
        int wide_points = cpi_data.cmd_params->pulse_params.wide_pulse_valid_point/8;
        int chn_total = 36;
        uint8_t pitch_beam_num = cpi_data.cmd_params->work_mode.pitch_beam_num;
        uint8_t azi_beam_num[3];
        azi_beam_num[0] = cpi_data.cmd_params->work_mode.azimuth_beam_num[0];
        azi_beam_num[1] = cpi_data.cmd_params->work_mode.azimuth_beam_num[1];
        azi_beam_num[2] = cpi_data.cmd_params->work_mode.azimuth_beam_num[2];

        // ddc数据迁移
        cudaStream_t stream_memcpy;  // 数据拷贝流
        cudaStreamCreate(&stream_memcpy);
        cuComplex* d_ddc_data = nullptr;


        // 测试是否阻塞
        long long is_async_begin = getTimeInMicroseconds();
 
        ErrorCheck(cudaMalloc((void**)&d_ddc_data,prf_total_pulse * chn_total  * (narrow_points + wide_points)* sizeof(cuComplex)), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpyAsync(d_ddc_data, ddc_data_begin_64, prf_total_pulse * chn_total * (narrow_points + wide_points)* sizeof(cuComplex), cudaMemcpyHostToDevice, stream_memcpy), __FILE__, __LINE__);
        long long is_async_end = getTimeInMicroseconds();
        // cout << is_async_end - is_async_begin << endl;
        long long data_transfer_end = getTimeInMicroseconds();
        spdlog::info("数据迁移时间：{}", data_transfer_end - data_transfer_begin);
        // 实例化类
        int azi_sector_num[3]={0,1,2};
        size_t GPU_MTD_abs = prf_total_pulse * (narrow_points + wide_points) * sizeof(float) * 4;
        cuda_signal_porcess cuda_signal_porcess(cpi_data, debug);
        // 指针数组
        float *d_MTD_abs[3] = {nullptr};

        ErrorCheck(cudaMalloc((void**)&d_MTD_abs[0], GPU_MTD_abs), __FILE__, __LINE__);
        ErrorCheck(cudaMalloc((void**)&d_MTD_abs[1], GPU_MTD_abs), __FILE__, __LINE__);   
        ErrorCheck(cudaMalloc((void**)&d_MTD_abs[2], GPU_MTD_abs), __FILE__, __LINE__);

        spdlog::info("prf_cnt:{}, wide_points:{}, narrow_points:{}", prf_total_pulse, wide_points, narrow_points);
       long long cal_begin = getTimeInMicroseconds();
        // #pragma omp parallel for shared(cpi_data, d_ddc_data, azi_sector_num, debug) schedule(dynamic)
        // process里面的DBF系数的cudaMemcpy起到同步流的作用，在ddc数据的拷贝完成后才会进行默认流操作
        for(int i = 0;i < 3;i++){
            cuda_signal_porcess.process(cpi_data, d_ddc_data, azi_sector_num[i], debug, d_MTD_abs[i], streams, handle_dbf, pc_handle[0], pc_handle[1], MTD_handle[0], MTD_handle[1]);
        }

        // #pragma omp parallel sections shared(cpi_data, d_ddc_data, azi_sector_num, debug)
        // {
        //     #pragma omp section
        //     {
        //         cuda_signal_porcess cuda_signal_porcess_0(cpi_data, d_ddc_data, azi_sector_num[0], debug, d_MTD_abs[0]);
        //         cuda_signal_porcess_0.process(cpi_data, d_ddc_data, azi_sector_num[0], debug, d_MTD_abs[0]);
        //     }
        //     #pragma omp section
        //     {
        //         cuda_signal_porcess cuda_signal_porcess_1(cpi_data, d_ddc_data, azi_sector_num[1], debug, d_MTD_abs[1]);
        //         cuda_signal_porcess_1.process(cpi_data, d_ddc_data, azi_sector_num[1], debug, d_MTD_abs[1]);
        //     }
        //     #pragma omp section
        //     {
        //         cuda_signal_porcess cuda_signal_porcess_2(cpi_data, d_ddc_data, azi_sector_num[2], debug, d_MTD_abs[2]);
        //         cuda_signal_porcess_2.process(cpi_data, d_ddc_data, azi_sector_num[2], debug, d_MTD_abs[2]);
        //     }
        // }

        // cuda_signal_porcess cuda_signal_porcess_0(cpi_data, d_ddc_data, azi_sector_num[0], debug, d_MTD_abs[0]);
        // cuda_signal_porcess_0.process(cpi_data, d_ddc_data, azi_sector_num[0], debug, d_MTD_abs[0]);
        // cuda_signal_porcess cuda_signal_porcess_1(cpi_data, d_ddc_data,azi_sector_num[1], debug, d_MTD_abs[1]);
        // cuda_signal_porcess_1.process(cpi_data, d_ddc_data, azi_sector_num[1], debug, d_MTD_abs[1]);
        // cuda_signal_porcess cuda_signal_porcess_2(cpi_data, d_ddc_data, azi_sector_num[2], debug, d_MTD_abs[2]);
        // cuda_signal_porcess_2.process(cpi_data, d_ddc_data, azi_sector_num[2], debug, d_MTD_abs[2]);

        // cudaDeviceSynchronize();
        long long cal_end = getTimeInMicroseconds();
        cudaDeviceSynchronize();
        spdlog::info("三扇区运算时间: {}", cal_end -cal_begin);
        // cout << "三扇区运算时间"<<cal_end -cal_begin << endl;
        // 数据释放

        ErrorCheck(cudaFreeAsync(d_MTD_abs[0], stream_memcpy), __FILE__, __LINE__);
        ErrorCheck(cudaFreeAsync(d_MTD_abs[1], stream_memcpy), __FILE__, __LINE__);
        ErrorCheck(cudaFreeAsync(d_MTD_abs[2], stream_memcpy), __FILE__, __LINE__);
        ErrorCheck(cudaFreeAsync(d_ddc_data, stream_memcpy), __FILE__, __LINE__);
        ErrorCheck(cudaFreeHost(cpi_data_begin), __FILE__, __LINE__);
        ErrorCheck(cudaStreamDestroy(stream_memcpy), __FILE__, __LINE__);
    }



    // 释放 cuBLAS Handle
    for (int i = 0; i < 2; ++i) {
        cublasDestroy(handle_dbf[i]);
    }
    // 释放 cuFFT Handle
    for (int i = 0; i < 6; ++i) {
        cufftDestroy(pc_handle[i]);
        cufftDestroy(MTD_handle[i]);
    }
    for (int i = 0; i < 2; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    return 0;
}