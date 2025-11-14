#pragma once
#include "iostream"
#include "common.cuh"
#include "data_read.hpp"
// #include "rsl_cfar.hpp"
#include "stdio.h"
#include "malloc.h"
#include "cuda_dbf.cuh"
#include "cuda_add_win.cuh"
#include "cuda_abs.cuh"
#include <cufft.h>
#include <device_launch_parameters.h>
#include "cuda_pc.cuh"
#include "rsl_win_func_manager.hpp"
#include "cuda_transpose.cuh"
#include "rsl_rsp_data_save.hpp"
#include "omp.h"
#include <spdlog/spdlog.h>

int cuda_radar_process (cpi_data_t& cpi_data, cuComplex* ddc_data_begin_64, int debug);


class cuda_signal_porcess{
    private:
        // 参数定义
        int prf_total_pulse;
        int narrow_pulse_pc_fft_points;
        int wide_pulse_pc_fft_points;
        int narrow_points;
        int wide_points;
        int chn_total = 36;
        uint8_t azi_beam_num[3];
        int dbf_channel_number = 12;
        int azi_sector_total_num = 3;  //总扇区数

        // 显存内各个部分数据指针
        cuComplex* d_total_memory = nullptr;
        cuComplex* d_narrow_sum_step1 = nullptr;
        cuComplex* d_narrow_diff_step1 = nullptr;
        cuComplex* d_wide_sum_step1 = nullptr;
        cuComplex* d_wide_diff_step1 = nullptr;
        cuComplex* d_narrow_sum_step1_f = nullptr;
        cuComplex* d_narrow_diff_step1_f = nullptr;
        cuComplex* d_wide_sum_step1_f = nullptr;
        cuComplex* d_wide_diff_step1_f = nullptr;
        cuComplex* d_narrow_sum_step2 = nullptr;
        cuComplex* d_narrow_diff_step2 = nullptr;
        cuComplex* d_wide_sum_step2 = nullptr;
        cuComplex* d_wide_diff_step2 = nullptr;
        cuComplex* d_narrow_sum_step3 = nullptr;
        cuComplex* d_narrow_diff_step3 = nullptr;
        cuComplex* d_wide_sum_step3 = nullptr;
        cuComplex* d_wide_diff_step3 = nullptr;
        cuComplex* d_narrow_sum_add_win = nullptr;
        cuComplex* d_narrow_diff_add_win = nullptr;
        cuComplex* d_wide_sum_add_win = nullptr;
        cuComplex* d_wide_diff_add_win = nullptr;
        cuComplex* d_MTD_narrow_sum = nullptr;
        cuComplex* d_MTD_narrow_diff = nullptr;
        cuComplex* d_MTD_wide_sum = nullptr;
        cuComplex* d_MTD_wide_diff = nullptr;

        // DBF系数
        cuComplex *d_dbf_sum_coe = nullptr;
        cuComplex *d_dbf_diff_coe = nullptr;
        cuComplex *dbf_coeff_sum = nullptr;
        cuComplex *dbf_coeff_diff = nullptr; 

        // PC系数
        cufftComplex* narrow_coeff = nullptr;
        cufftComplex* wide_coeff = nullptr;
        cuComplex *d_wide_coeff = nullptr;
        cuComplex *d_narrow_coeff = nullptr;

        // 窗系数
        float* d_win = nullptr;

        // cuda流与句柄建立
        // cuda流建立
        // int cuda_stream_num = 2;
        // // 先窄后宽 
        // cudaStream_t streams[2];
        // cublasHandle_t handle_dbf[2];

        // // 避免阻塞的数据拷贝和memset流
        cudaStream_t stream_memcpy;
        // // pc句柄
        // cufftHandle pc_wide;
        // cufftHandle pc_narrow;

        // // MTD句柄
        // cufftHandle MTD_wide;
        // cufftHandle MTD_narrow;


    public:
        /*
            构造函数：
                -参数：
                    -cpi_data:完整的CPI数据
                    -d_ddc_data:显存里的GPU数据
                    -azi_sector_num:第几号扇区 
                    -debug:debug参数
                    -d_MTD_abs:输出的显存MTD数据
        */
        
        cuda_signal_porcess(cpi_data_t& cpi_data, int debug){
            spdlog::info("构造函数开始");
            long long init_begin = getTimeInMicroseconds();
            // 参数获取
            prf_total_pulse = cpi_data.cmd_params->pulse_params.prf_total_pulse;
            narrow_pulse_pc_fft_points = cpi_data.cmd_params->PC_params.narrow_pulse_pc_fft_points;
            wide_pulse_pc_fft_points = cpi_data.cmd_params->PC_params.wide_pulse_pc_fft_points;
            narrow_points = cpi_data.cmd_params->pulse_params.narrow_pulse_valid_point / 8;
            wide_points = cpi_data.cmd_params->pulse_params.wide_pulse_valid_point / 8;
            chn_total = 36;
            
            // 这个是什么呢？
            azi_beam_num[0] = cpi_data.cmd_params->work_mode.azimuth_beam_num[0];
            azi_beam_num[1] = cpi_data.cmd_params->work_mode.azimuth_beam_num[1];
            azi_beam_num[2] = cpi_data.cmd_params->work_mode.azimuth_beam_num[2];

            // 数据构建
            size_t GPU_DBF_size = prf_total_pulse * (narrow_pulse_pc_fft_points + wide_pulse_pc_fft_points) * sizeof(cuComplex) * 4;
            size_t GPU_PC_size = GPU_DBF_size * 3;
            size_t GPU_WIN_size = GPU_DBF_size;
            size_t GPU_MTD_size = prf_total_pulse * (narrow_points + wide_points) * sizeof(cuComplex) * 4;
            size_t GPU_size = GPU_DBF_size + GPU_PC_size + GPU_WIN_size + GPU_MTD_size;


            // 创建拷贝流
            cudaStreamCreate(&stream_memcpy); 

            // 总显存分配
            ErrorCheck(cudaMallocAsync((void**)&d_total_memory, GPU_size, stream_memcpy), __FILE__, __LINE__);
            ErrorCheck(cudaMemsetAsync(d_total_memory, 0, GPU_size,stream_memcpy), __FILE__, __LINE__);

            // 设置显存数据的各个部分起始位置
            d_narrow_sum_step1 = d_total_memory;
            d_narrow_diff_step1 = d_narrow_sum_step1 + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_sum_step1 = d_narrow_diff_step1 + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_diff_step1 = d_wide_sum_step1 + prf_total_pulse * wide_pulse_pc_fft_points;

            d_narrow_sum_step1_f = d_wide_diff_step1 + prf_total_pulse * wide_pulse_pc_fft_points;
            d_narrow_diff_step1_f = d_narrow_sum_step1_f + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_sum_step1_f = d_narrow_diff_step1_f + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_diff_step1_f = d_wide_sum_step1_f + prf_total_pulse * wide_pulse_pc_fft_points;

            d_narrow_sum_step2 = d_wide_diff_step1_f + prf_total_pulse * wide_pulse_pc_fft_points;
            d_narrow_diff_step2 = d_narrow_sum_step2 + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_sum_step2 = d_narrow_diff_step2 + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_diff_step2 = d_wide_sum_step2 + prf_total_pulse * wide_pulse_pc_fft_points;

            d_narrow_sum_step3 = d_wide_diff_step2 + prf_total_pulse * wide_pulse_pc_fft_points;
            d_narrow_diff_step3 = d_narrow_sum_step3 + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_sum_step3 = d_narrow_diff_step3 + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_diff_step3 = d_wide_sum_step3 + prf_total_pulse * wide_pulse_pc_fft_points;

            d_narrow_sum_add_win = d_wide_diff_step3 + prf_total_pulse * wide_pulse_pc_fft_points;
            d_narrow_diff_add_win = d_narrow_sum_add_win + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_sum_add_win = d_narrow_diff_add_win + prf_total_pulse * narrow_pulse_pc_fft_points;
            d_wide_diff_add_win = d_wide_sum_add_win + prf_total_pulse * wide_pulse_pc_fft_points;

            d_MTD_narrow_sum = d_wide_diff_add_win + prf_total_pulse * wide_pulse_pc_fft_points;
            d_MTD_narrow_diff = d_MTD_narrow_sum + prf_total_pulse * narrow_points;
            d_MTD_wide_sum = d_MTD_narrow_diff + prf_total_pulse * narrow_points;
            d_MTD_wide_diff = d_MTD_wide_sum + prf_total_pulse * wide_points;

            // DBF系数获取
            ErrorCheck(cudaMallocAsync((void**)&d_dbf_sum_coe, chn_total * azi_sector_total_num * sizeof(cuComplex), stream_memcpy), __FILE__, __LINE__);
            ErrorCheck(cudaMallocAsync((void**)&d_dbf_diff_coe, chn_total * azi_sector_total_num * sizeof(cuComplex), stream_memcpy), __FILE__, __LINE__);;

            ErrorCheck(cudaMallocHost((void**)&dbf_coeff_sum, chn_total * azi_sector_total_num * sizeof(cuComplex)), __FILE__, __LINE__);
            ErrorCheck(cudaMallocHost((void**)&dbf_coeff_diff, chn_total * azi_sector_total_num * sizeof(cuComplex)), __FILE__, __LINE__);

            // 初始化为零
            memset(dbf_coeff_sum, 0, chn_total * azi_sector_total_num * sizeof(cuComplex));
            memset(dbf_coeff_diff, 0, chn_total * azi_sector_total_num * sizeof(cuComplex));

            for(int azi_sector_num = 0; azi_sector_num < azi_sector_total_num; azi_sector_num++){
                int dbf_group = cpi_data.cmd_params->DBF_group[azi_sector_num][0] - 1;
                int dbf_coe_offset = chn_total * azi_sector_num;            
                for (int chn_index = 0; chn_index < dbf_channel_number; chn_index++)
                {
                    if (dbf_group + chn_index <= chn_total - 1){
                        dbf_coeff_sum[dbf_coe_offset + dbf_group + chn_index].x = ((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][0][chn_index * 2]);
                        dbf_coeff_sum[dbf_coe_offset + dbf_group + chn_index].y = ((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][0][chn_index * 2 + 1]);
                        dbf_coeff_diff[dbf_coe_offset + dbf_group + chn_index].x = ((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][1][chn_index * 2]);
                        dbf_coeff_diff[dbf_coe_offset + dbf_group + chn_index].y = ((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][1][chn_index * 2 + 1]);
                    }
                    else{
                        dbf_coeff_sum[dbf_coe_offset + dbf_group + chn_index - chn_total].x = ((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][0][chn_index * 2]);
                        dbf_coeff_sum[dbf_coe_offset + dbf_group + chn_index - chn_total].y = ((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][0][chn_index * 2 + 1]);
                        dbf_coeff_diff[dbf_coe_offset + dbf_group + chn_index - chn_total].x = ((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][1][chn_index * 2]);
                        dbf_coeff_diff[dbf_coe_offset + dbf_group + chn_index - chn_total].y = ((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][1][chn_index * 2 + 1]);                    
                    }
                }
                // ErrorCheck(cudaMemsetAsync(d_dbf_sum_coe, 0, chn_total * sizeof(cuComplex), stream_memcpy), __FILE__, __LINE__);
                // ErrorCheck(cudaMemsetAsync(d_dbf_diff_coe, 0, chn_total * sizeof(cuComplex), stream_memcpy), __FILE__, __LINE__);

            }

            // PC系数获取 
            // 避免等待fft句柄创建时等待
            cufftComplex* narrow_coeff = nullptr, *wide_coeff = nullptr;
            cudaMallocHost((void**)&narrow_coeff, sizeof(cufftComplex) * narrow_pulse_pc_fft_points);
            cudaMallocHost((void**)&wide_coeff, sizeof(cufftComplex) * wide_pulse_pc_fft_points);
            for (int point_index = 0; point_index < narrow_pulse_pc_fft_points; point_index++)
            {
                narrow_coeff[point_index].x= (float)cpi_data.cmd_params->PC_params.narrow_pulse_pc_coeff[point_index * 2];
                narrow_coeff[point_index].y = (float)cpi_data.cmd_params->PC_params.narrow_pulse_pc_coeff[point_index * 2 + 1];
            }
            for (int point_index = 0; point_index < wide_pulse_pc_fft_points; point_index++)
            {
                wide_coeff[point_index].x = (float)cpi_data.cmd_params->PC_params.wide_pulse_pc_coeff[point_index * 2];
                wide_coeff[point_index].y = (float)cpi_data.cmd_params->PC_params.wide_pulse_pc_coeff[point_index * 2 + 1];
            }

            // MTD参数设置
            // 在这里设置避免等待fft句柄创建时阻塞，重叠数据迁移与CPU操作
            // int MTD_narrow_rank = 1;   //  维
            // int MTD_narrow_n[1] = { prf_total_pulse };    // 每一维变换数目
            // int MTD_narrow_inembed[2] = { narrow_pulse_pc_fft_points, prf_total_pulse }; // 输入数据的步幅
            // int MTD_narrow_istride = narrow_pulse_pc_fft_points; // 输入每个 FFT 的数据步幅
            // int MTD_narrow_idist = 1; // 不同 FFT 之间的距离
            // int MTD_narrow_onembed[2] = { prf_total_pulse, narrow_points }; // 输出数据的步幅
            // int MTD_narrow_ostride = narrow_points; // 输出每个 FFT 的数据步幅
            // int MTD_narrow_odist = 1; //
            // int MTD_narrow_batch = narrow_points;   // narrow_pulse_pc_fft_points次FFT

            // int MTD_wide_rank = 1;   //  维
            // int MTD_wide_n[1] = { prf_total_pulse };    // 每一维变换数目
            // int MTD_wide_inembed[2] = { wide_pulse_pc_fft_points, prf_total_pulse }; // 输入数据的步幅
            // int MTD_wide_istride = wide_pulse_pc_fft_points; // 输入每个 FFT 的数据步幅
            // int MTD_wide_idist = 1; // 不同 FFT 之间的距离
            // int MTD_wide_onembed[2] = { prf_total_pulse, wide_points }; // 输出数据的步幅
            // int MTD_wide_ostride = wide_points; // 输出每个 FFT 的数据步幅
            // int MTD_wide_odist = 1; //
            // int MTD_wide_batch = wide_points;   // wide_pulse_pc_fft_points次FFT

            // 窗函数获取
            typedef Map<Matrix<float, 1, Dynamic, RowMajor>>  MapWin;
            win_type_t curr_win_type = Taylor;
            MapWin win(radar_signal_process::rsl_win_func_manager::getInstance().get_win_func(curr_win_type,prf_total_pulse,60),1,prf_total_pulse);


            // DBF系数拷贝
            ErrorCheck(cudaMemcpyAsync(d_dbf_sum_coe, dbf_coeff_sum, chn_total * azi_sector_total_num * sizeof(cuComplex), cudaMemcpyHostToDevice, stream_memcpy), __FILE__, __LINE__);
            ErrorCheck(cudaMemcpyAsync(d_dbf_diff_coe, dbf_coeff_diff, chn_total * azi_sector_total_num * sizeof(cuComplex), cudaMemcpyHostToDevice, stream_memcpy), __FILE__, __LINE__);   

            // cuda流创建
            // 句柄创建
            // 句柄参数设置绑定
            // 将句柄与流绑定
            // DBF句柄与流绑定
            // 创建流和 CUBLAS 句柄
            // streams[0]:sum_wide, streams[1]:diff_wide, streams[2]:sum_narrow, streams[3]:diff_narrow 先宽后窄 先和后差
            // for (int i = 0; i < cuda_stream_num; ++i) {
            //     cudaStreamCreate(&streams[i]);               // 创建CUDA流
            //     cublasCreate(&handle_dbf[i]);                // 创建CUBLAS句柄
            //     cublasSetStream(handle_dbf[i], streams[i]);  // 将CUBLAS句柄与流关联
            // }

            // // cufftPlan会有一小段的默认流cudamemcpy
            // // pc句柄与流绑定
            // checkCufftError(cufftPlan1d(&pc_wide, wide_pulse_pc_fft_points, CUFFT_C2C, prf_total_pulse));
            // checkCufftError(cufftPlan1d(&pc_narrow, narrow_pulse_pc_fft_points, CUFFT_C2C, prf_total_pulse));


            // checkCufftError(cufftSetStream(pc_wide, streams[0]));
            // checkCufftError(cufftSetStream(pc_narrow, streams[1]));

            // MTD句柄与流绑定
            // cufftPlan会有一小段的默认流cudamemcpy
            // checkCufftError(cufftPlanMany(&MTD_wide, MTD_wide_rank, MTD_wide_n, MTD_wide_inembed, MTD_wide_istride, MTD_wide_idist, MTD_wide_onembed, MTD_wide_ostride, MTD_wide_odist, CUFFT_C2C, MTD_wide_batch));
            // checkCufftError(cufftPlanMany(&MTD_narrow, MTD_narrow_rank, MTD_narrow_n, MTD_narrow_inembed, MTD_narrow_istride, MTD_narrow_idist, MTD_narrow_onembed, MTD_narrow_ostride, MTD_narrow_odist, CUFFT_C2C, MTD_narrow_batch));

            // checkCufftError(cufftSetStream(MTD_wide, streams[0]));
            // checkCufftError(cufftSetStream(MTD_narrow, streams[1]));

            // PC系数迁移
            cudaMalloc((void**)&d_narrow_coeff, narrow_pulse_pc_fft_points * sizeof(cuComplex));
            cudaMalloc((void**)&d_wide_coeff, wide_pulse_pc_fft_points * sizeof(cuComplex));
            ErrorCheck(cudaMemcpyAsync(d_narrow_coeff, narrow_coeff, narrow_pulse_pc_fft_points * sizeof(cuComplex), cudaMemcpyHostToDevice, stream_memcpy), __FILE__, __LINE__);
            ErrorCheck(cudaMemcpyAsync(d_wide_coeff, wide_coeff, wide_pulse_pc_fft_points * sizeof(cuComplex), cudaMemcpyHostToDevice, stream_memcpy), __FILE__, __LINE__);


            // 最后默认流窗系数迁移，保证同步
            ErrorCheck(cudaMallocAsync((void**)&d_win, prf_total_pulse*sizeof(float),stream_memcpy), __FILE__, __LINE__);
            // cout<<"win_rows: "<<win.rows()<<"win_cols: " <<win.cols()<<endl;
            ErrorCheck(cudaMemcpy(d_win, win.data(), prf_total_pulse*sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
            // writeFloatToFile( "/home/ryh/radar/cir_radar/data/d_win.csv", d_win, prf_total_pulse);
            long long init_end = getTimeInMicroseconds();
            spdlog::info("数据构造时间: {}", init_end - init_begin);
            // std::cout << "构造函数结束" <<endl;
        }

        // DBF,PC,MTD信号处理
        void process(cpi_data_t& cpi_data, cuComplex* d_ddc_data, int azi_sector_num, int debug, float* d_MTD_abs, cudaStream_t* streams, cublasHandle_t* handle_dbf, cufftHandle &pc_wide, cufftHandle &pc_narrow, cufftHandle &MTD_wide, cufftHandle &MTD_narrow){
            // 文件名
            string filename = "/home/ryh/ryh/cir_radar/data/cuda_";
            long long GPU_begin = getTimeInMicroseconds();
            long long d_dbf_test_2_begin = getTimeInMicroseconds();
            //-----------------------------DBF---------------------------//

            // C = alpha * A * B + beta * C。
            cuComplex alpha_dbf = make_cuComplex(1.0f, 0.0f);  // 矩阵乘法的标量因子 (实部: 1.0f, 虚部: 1.0f)
            cuComplex beta_dbf = make_cuComplex(0.0f, 0.0f);   // 矩阵C的初始值标量因子
            int strideA = 0; // A矩阵在批次中的步长
            int strideB = (chn_total) * (narrow_points + wide_points); // B矩阵在批次中的步长
            int strideC = narrow_pulse_pc_fft_points; // C矩阵在批次中的步长
            int batch_count = prf_total_pulse;
            // // 等待流操作完成
            // cudaStreamSynchronize(stream_memcpy);
            // 宽窄分离，避免访问同时访问同一块显存
            int dbf_coe_offset = chn_total * azi_sector_num;  
            cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, chn_total,
                                    &alpha_dbf, d_dbf_sum_coe + dbf_coe_offset, chn_total, 0, d_ddc_data + chn_total * narrow_points,
                                    wide_points, strideB, &beta_dbf, d_wide_sum_step1, 1, wide_pulse_pc_fft_points, batch_count);

            cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, chn_total,
                                    &alpha_dbf, d_dbf_diff_coe + dbf_coe_offset, chn_total, 0, d_ddc_data,
                                    narrow_points, strideB, &beta_dbf, d_narrow_diff_step1, 1, narrow_pulse_pc_fft_points, batch_count);

            cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, chn_total,
                                    &alpha_dbf, d_dbf_sum_coe + dbf_coe_offset, chn_total, 0, d_ddc_data,
                                    narrow_points, strideB , &beta_dbf, d_narrow_sum_step1, 1, narrow_pulse_pc_fft_points, batch_count);

            cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, chn_total,
                                    &alpha_dbf, d_dbf_diff_coe + dbf_coe_offset, chn_total, 0, d_ddc_data + chn_total * narrow_points,
                                    wide_points, strideB, &beta_dbf, d_wide_diff_step1, 1, wide_pulse_pc_fft_points, batch_count);


            long long d_dbf_test_2_end = getTimeInMicroseconds();
            spdlog::info("{} gpu_dbf_time: {}", azi_sector_num, d_dbf_test_2_end-d_dbf_test_2_begin);
            // cout << azi_sector_num <<"gpu_dbf_time:"<<(d_dbf_test_2_end-d_dbf_test_2_begin)<<endl;

            // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_dbf_wide_coe_class_2.csv", d_dbf_sum_coe, 1 * chn_total,1,chn_total); 
            // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_ddc_data_class_2.csv", d_ddc_data , prf_total_pulse * narrow_points,prf_total_pulse,narrow_points); 
            // cudaStreamSynchronize();
            // cudaDeviceSynchronize(); 
            // cufftComplex_save_to_csv(filename + "dbf_narrow_sum_" + to_string(azi_sector_num) + ".csv", d_narrow_sum_step1, prf_total_pulse * narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "dbf_narrow_diff_" + to_string(azi_sector_num) + ".csv", d_narrow_diff_step1, prf_total_pulse * narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "dbf_wide_sum_" + to_string(azi_sector_num) + ".csv", d_wide_sum_step1, prf_total_pulse * wide_pulse_pc_fft_points,prf_total_pulse, wide_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "dbf_wide_diff_" + to_string(azi_sector_num) + ".csv", d_wide_diff_step1, prf_total_pulse * wide_pulse_pc_fft_points,prf_total_pulse, wide_pulse_pc_fft_points); 
            //-------------------------------PC----------------------------------------------//
            // 脉压step1：信号FFT
            // FFT
            //cudaDeviceSynchronize();
            long long d_PC_begin = getTimeInMicroseconds();
            cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step1, (cufftComplex*)d_wide_sum_step1_f, CUFFT_FORWARD);
            cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step1, (cufftComplex*)d_narrow_sum_step1_f, CUFFT_FORWARD);
            cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step1, (cufftComplex*)d_wide_diff_step1_f, CUFFT_FORWARD);
            cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step1, (cufftComplex*)d_narrow_diff_step1_f, CUFFT_FORWARD);

            // // cudaDeviceSynchronize();
            // cufftComplex_save_to_csv(filename + "pc1_narrow_sum_" + to_string(azi_sector_num) + ".csv", d_narrow_sum_step1_f, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc1_narrow_diff_" + to_string(azi_sector_num) + ".csv", d_narrow_diff_step1_f, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc1_wide_sum_" + to_string(azi_sector_num) + ".csv", d_wide_sum_step1_f, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc1_wide_diff_" + to_string(azi_sector_num) + ".csv", d_wide_diff_step1_f, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
            // 脉压step2：信号相乘
            // 将本地信号的FFT系数给GPU
            // GPU对应数据相乘
            // cudaDeviceSynchronize();
            pc_complexMultiply(d_wide_sum_step1_f, d_wide_coeff, d_wide_sum_step2, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points,streams[0]);
            pc_complexMultiply(d_narrow_sum_step1_f, d_narrow_coeff, d_narrow_sum_step2, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points,streams[1]);
            pc_complexMultiply(d_wide_diff_step1_f, d_wide_coeff, d_wide_diff_step2, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points,streams[0]);
            pc_complexMultiply(d_narrow_diff_step1_f, d_narrow_coeff, d_narrow_diff_step2, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points,streams[1]);

            // cufftComplex_save_to_csv(filename + "pc2_narrow_sum_" + to_string(azi_sector_num) + ".csv", d_narrow_sum_step2, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc2_narrow_diff_" + to_string(azi_sector_num) + ".csv", d_narrow_diff_step2, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc2_wide_sum_" + to_string(azi_sector_num) + ".csv", d_wide_sum_step2, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc2_wide_diff_" + to_string(azi_sector_num) + ".csv", d_wide_diff_step2, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 


            // CPU 上的执行cudaMalloc与在 GPU 上的内核执行pc_complexMultiply重叠，默认流中的内核执行与主机上的代码执行重叠
            // cudaDeviceSynchronize();
            cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step2, (cufftComplex*)d_wide_sum_step3, CUFFT_INVERSE);
            cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step2, (cufftComplex*)d_narrow_sum_step3, CUFFT_INVERSE);
            cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step2, (cufftComplex*)d_wide_diff_step3, CUFFT_INVERSE);
            cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step2, (cufftComplex*)d_narrow_diff_step3, CUFFT_INVERSE);
            // cufftComplex_save_to_csv(filename + "pc3_narrow_sum_" + to_string(azi_sector_num) + ".csv", d_narrow_sum_step3, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc3_narrow_diff_" + to_string(azi_sector_num) + ".csv", d_narrow_diff_step3, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc3_wide_sum_" + to_string(azi_sector_num) + ".csv", d_wide_sum_step3, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
            // cufftComplex_save_to_csv(filename + "pc3_wide_diff_" + to_string(azi_sector_num) + ".csv", d_wide_diff_step3, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
            
            long long d_PC_end = getTimeInMicroseconds();
            spdlog::info("{} gpu_PC_time: {}", azi_sector_num, d_PC_end- d_PC_begin);
            // cout<< azi_sector_num <<"gpu_PC_time:"<<( d_PC_end- d_PC_begin)<<endl;

            //---------------------------------加窗------------------------------------------//
            //------------------------------------------------------------------------------//
            //------------------------------------------------------------------------------//
            long long d_MTD_begin = getTimeInMicroseconds();
            // cudaDeviceSynchronize();
            add_win(d_wide_sum_step3,d_win,d_wide_sum_add_win,prf_total_pulse*wide_pulse_pc_fft_points,wide_pulse_pc_fft_points,streams[0]);
            add_win(d_narrow_sum_step3,d_win,d_narrow_sum_add_win,prf_total_pulse*narrow_pulse_pc_fft_points,narrow_pulse_pc_fft_points,streams[1]);
            add_win(d_wide_diff_step3,d_win,d_wide_diff_add_win,prf_total_pulse*wide_pulse_pc_fft_points,wide_pulse_pc_fft_points,streams[0]);
            add_win(d_narrow_diff_step3,d_win,d_narrow_diff_add_win,prf_total_pulse*narrow_pulse_pc_fft_points,narrow_pulse_pc_fft_points,streams[1]);

            // cufftComplex_save_to_csv(filename + "addwin_narrow_sum_" + to_string(azi_sector_num) + ".csv", d_narrow_sum_add_win, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
            // cufftComplex_save_to_csv(filename + "addwin_narrow_diff_" + to_string(azi_sector_num) + ".csv", d_narrow_diff_add_win, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
            // cufftComplex_save_to_csv(filename + "addwin_wide_sum_" + to_string(azi_sector_num) + ".csv", d_wide_sum_add_win, prf_total_pulse*wide_points,prf_total_pulse,wide_points);
            // cufftComplex_save_to_csv(filename + "addwin_wide_diff_" + to_string(azi_sector_num) + ".csv", d_wide_diff_add_win, prf_total_pulse*wide_points,prf_total_pulse,wide_points);

            // ---------------------------------MTD -----------------------------------------//
            //------------------------------------------------------------------------------//
            //------------------------------------------------------------------------------//
            // cudaDeviceSynchronize();
            checkCufftError(cufftExecC2C(MTD_wide, d_wide_sum_add_win, d_MTD_wide_sum, CUFFT_FORWARD));
            checkCufftError(cufftExecC2C(MTD_narrow, d_narrow_sum_add_win, d_MTD_narrow_sum, CUFFT_FORWARD));
            checkCufftError(cufftExecC2C(MTD_wide, d_wide_diff_add_win, d_MTD_wide_diff, CUFFT_FORWARD));
            checkCufftError(cufftExecC2C(MTD_narrow, d_narrow_diff_add_win, d_MTD_narrow_diff, CUFFT_FORWARD));

            // cufftComplex_save_to_csv(filename + "mtd_narrow_sum_" + to_string(azi_sector_num) + ".csv", d_MTD_narrow_sum, prf_total_pulse*narrow_points,narrow_points,prf_total_pulse);
            // cufftComplex_save_to_csv(filename + "mtd_narrow_diff_" + to_string(azi_sector_num) + ".csv", d_MTD_narrow_diff, prf_total_pulse*narrow_points,narrow_points,prf_total_pulse);
            // cufftComplex_save_to_csv(filename + "mtd_wide_sum_" + to_string(azi_sector_num) + ".csv", d_MTD_wide_sum, prf_total_pulse*wide_points,wide_points,prf_total_pulse);
            // cufftComplex_save_to_csv(filename + "mtd_wide_diff_" + to_string(azi_sector_num) + ".csv", d_MTD_wide_diff, prf_total_pulse*wide_points,wide_points,prf_total_pulse);
            // abs

            // cudaDeviceSynchronize();
            // size_t GPU_MTD_abs = prf_total_pulse * (narrow_points + wide_points) * sizeof(float) * 4;
            // float *d_MTD_abs = nullptr;
            // ErrorCheck(cudaMalloc((void**)&d_MTD_abs, GPU_MTD_abs), __FILE__, __LINE__);
            // MTD_abs
            float *d_MTD_abs_narrow_sum = d_MTD_abs;
            float *d_MTD_abs_narrow_diff = d_MTD_abs_narrow_sum + prf_total_pulse * narrow_points;
            float *d_MTD_abs_wide_sum = d_MTD_abs_narrow_diff + prf_total_pulse * narrow_points;
            float *d_MTD_abs_wide_diff = d_MTD_abs_wide_sum + prf_total_pulse * wide_points;
            MTD_abs(d_MTD_wide_sum,d_MTD_abs_wide_sum,prf_total_pulse * wide_points,streams[0]);
            MTD_abs(d_MTD_narrow_sum,d_MTD_abs_narrow_sum,prf_total_pulse * narrow_points,streams[1]);
            MTD_abs(d_MTD_wide_diff,d_MTD_abs_wide_diff,prf_total_pulse * wide_points,streams[0]);
            MTD_abs(d_MTD_narrow_diff,d_MTD_abs_narrow_diff,prf_total_pulse * narrow_points,streams[1]);
            long long d_MTD_end = getTimeInMicroseconds();
            spdlog::info("{} d_MTD_time: {}", azi_sector_num, d_MTD_end-d_MTD_begin);
            // cout<< azi_sector_num <<"d_MTD_time:"<<(d_MTD_end-d_MTD_begin)<<endl;
            long long GPU_end = getTimeInMicroseconds();
            spdlog::info("{} GPU_time: {}", azi_sector_num, GPU_end-GPU_begin);            
            // cout<< azi_sector_num <<"GPU_time:"<<(GPU_end-GPU_begin)<<endl;

            // target_map_t target_tmp;
            // radar_signal_process::rsl_cfar cfar;
            // clutter_map_t clutter_map_data;
            // cfar.cfar(cpi_data, save_MTD_abs_data, clutter_map_data, azi_sector_num, target_tmp);

            // float_save_to_csv(filename + "mtd_abs_narrow_sum_" + to_string(azi_sector_num) + ".csv", d_MTD_abs_narrow_sum, prf_total_pulse*narrow_points,narrow_points,prf_total_pulse);
            // float_save_to_csv(filename + "mtd_abs_narrow_diff_" + to_string(azi_sector_num) + ".csv", d_MTD_abs_narrow_diff, prf_total_pulse*narrow_points,narrow_points,prf_total_pulse);
            // float_save_to_csv(filename + "mtd_abs_wide_sum_" + to_string(azi_sector_num) + ".csv" , d_MTD_abs_wide_sum, prf_total_pulse*wide_points,wide_points,prf_total_pulse);
            // float_save_to_csv(filename + "mtd_abs_wide_diff_" + to_string(azi_sector_num) + ".csv" , d_MTD_abs_wide_diff, prf_total_pulse*wide_points,wide_points,prf_total_pulse);


            // float_save_to_csv(filename + "mtd_abs_narrow_sum_test" + to_string(azi_sector_num) + ".csv", d_MTD_abs_narrow_sum, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
            // float_save_to_csv(filename + "mtd_abs_narrow_diff_test" + to_string(azi_sector_num) + ".csv", d_MTD_abs_narrow_diff, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
            // float_save_to_csv(filename + "mtd_abs_wide_sum_test" + to_string(azi_sector_num) + ".csv" , d_MTD_abs_wide_sum, prf_total_pulse*wide_points,prf_total_pulse,wide_points);
            // float_save_to_csv(filename + "mtd_abs_wide_diff_test" + to_string(azi_sector_num) + ".csv" , d_MTD_abs_wide_diff, prf_total_pulse*wide_points,prf_total_pulse,wide_points);
        }

        // // openmp并行执行四个cuda流加载
        // void process_omp(cpi_data_t& cpi_data, cuComplex* d_ddc_data, int azi_sector_num, int debug, float* d_MTD_abs){
            
        //     cudaDeviceSynchronize();
        //     string filename = "/home/ryh/ryh/cir_radar/data/cuda_";

        //     long long GPU_begin = getTimeInMicroseconds();

        //     cuComplex alpha_dbf = make_cuComplex(1.0f, 0.0f);  // 矩阵乘法的标量因子 (实部: 1.0f, 虚部: 1.0f)
        //     cuComplex beta_dbf = make_cuComplex(0.0f, 0.0f);   // 矩阵C的初始值标量因子
        //     int strideA = 0; // A矩阵在批次中的步长
        //     int strideB = (chn_total) * (narrow_points + wide_points); // B矩阵在批次中的步长
        //     int strideC = narrow_pulse_pc_fft_points; // C矩阵在批次中的步长
        //     int batch_count = prf_total_pulse;
        //     int dbf_coe_offset = chn_total * azi_sector_num;  

        //     float *d_MTD_abs_narrow_sum = d_MTD_abs;
        //     float *d_MTD_abs_narrow_diff = d_MTD_abs_narrow_sum + prf_total_pulse * narrow_points;
        //     float *d_MTD_abs_wide_sum = d_MTD_abs_narrow_diff + prf_total_pulse * narrow_points;
        //     float *d_MTD_abs_wide_diff = d_MTD_abs_wide_sum + prf_total_pulse * wide_points;
        //     // 使用 OpenMP 来并行处理 CUDA 流的加载操作
        //     #pragma omp parallel sections
        //     {
        //         #pragma omp section
        //         {
        //             // CUDA 流 0 的操作 宽和
        //             cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, chn_total,
        //                         &alpha_dbf, d_dbf_sum_coe + dbf_coe_offset, chn_total, 0, d_ddc_data + chn_total * narrow_points,
        //                         wide_points, strideB, &beta_dbf, d_wide_sum_step1, 1, wide_pulse_pc_fft_points, batch_count);

        //             cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step1, (cufftComplex*)d_wide_sum_step1_f, CUFFT_FORWARD);
        //             pc_complexMultiply(d_wide_sum_step1_f, d_wide_coeff, d_wide_sum_step2, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points,streams[0]);
        //             cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step2, (cufftComplex*)d_wide_sum_step3, CUFFT_INVERSE);

        //             add_win(d_wide_sum_step3, d_win, d_wide_sum_add_win, prf_total_pulse * wide_pulse_pc_fft_points, wide_pulse_pc_fft_points, streams[0]);
        //             checkCufftError(cufftExecC2C(MTD_wide, d_wide_sum_add_win, d_MTD_wide_sum, CUFFT_FORWARD));
        //             MTD_abs(d_MTD_wide_sum, d_MTD_abs_wide_sum, prf_total_pulse * wide_points, streams[0]);
        //         }

        //         #pragma omp section
        //         {
        //             // CUDA 流 1 的操作 宽差
        //             cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, chn_total,
        //                         &alpha_dbf, d_dbf_diff_coe + dbf_coe_offset, chn_total, 0, d_ddc_data + chn_total * narrow_points,
        //                         wide_points, strideB, &beta_dbf, d_wide_diff_step1, 1, wide_pulse_pc_fft_points, batch_count);

                    
        //             cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step1, (cufftComplex*)d_wide_diff_step1_f, CUFFT_FORWARD);
        //             pc_complexMultiply(d_wide_diff_step1_f, d_wide_coeff, d_wide_diff_step2, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points, streams[0]);
        //             cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step2, (cufftComplex*)d_wide_diff_step3, CUFFT_INVERSE);    

        //             add_win(d_wide_diff_step3, d_win, d_wide_diff_add_win, prf_total_pulse * wide_pulse_pc_fft_points, wide_pulse_pc_fft_points, streams[0]);
        //             checkCufftError(cufftExecC2C(MTD_wide, d_wide_diff_add_win, d_MTD_wide_diff, CUFFT_FORWARD));
        //             MTD_abs(d_MTD_wide_diff, d_MTD_abs_wide_diff, prf_total_pulse * wide_points, streams[0]);
        //         }

        //         #pragma omp section
        //         {
        //             // CUDA 流 2 的操作 窄和
        //             cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, chn_total,
        //                                         &alpha_dbf, d_dbf_sum_coe + dbf_coe_offset, chn_total, 0, d_ddc_data,
        //                                         narrow_points, strideB , &beta_dbf, d_narrow_sum_step1, 1, narrow_pulse_pc_fft_points, batch_count);

        //             cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step1, (cufftComplex*)d_narrow_sum_step1_f, CUFFT_FORWARD);
        //             pc_complexMultiply(d_narrow_sum_step1_f, d_narrow_coeff, d_narrow_sum_step2, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points, streams[1]);
        //             cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step2, (cufftComplex*)d_narrow_sum_step3, CUFFT_INVERSE);

        //             add_win(d_narrow_sum_step3, d_win, d_narrow_sum_add_win, prf_total_pulse * narrow_pulse_pc_fft_points, narrow_pulse_pc_fft_points, streams[1]);
        //             checkCufftError(cufftExecC2C(MTD_narrow, d_narrow_sum_add_win, d_MTD_narrow_sum, CUFFT_FORWARD));
        //             MTD_abs(d_MTD_narrow_sum, d_MTD_abs_narrow_sum, prf_total_pulse * narrow_points, streams[1]);
        //         }

        //         #pragma omp section
        //         {
        //             // CUDA 流 3 的操作 窄差
        //             cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, chn_total,
        //                                     &alpha_dbf, d_dbf_diff_coe + dbf_coe_offset, chn_total, 0, d_ddc_data,
        //                                     narrow_points, strideB, &beta_dbf, d_narrow_diff_step1, 1, narrow_pulse_pc_fft_points, batch_count);

        //             cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step1, (cufftComplex*)d_narrow_diff_step1_f, CUFFT_FORWARD);
        //             pc_complexMultiply(d_narrow_diff_step1_f, d_narrow_coeff, d_narrow_diff_step2, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points, streams[1]);
        //             cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step2, (cufftComplex*)d_narrow_diff_step3, CUFFT_INVERSE);
                    
        //             add_win(d_narrow_diff_step3, d_win, d_narrow_diff_add_win, prf_total_pulse * narrow_pulse_pc_fft_points, narrow_pulse_pc_fft_points, streams[1]);
        //             checkCufftError(cufftExecC2C(MTD_narrow, d_narrow_diff_add_win, d_MTD_narrow_diff, CUFFT_FORWARD));
        //             MTD_abs(d_MTD_narrow_diff, d_MTD_abs_narrow_diff, prf_total_pulse * narrow_points, streams[1]);
        //         }
        //         long long GPU_end = getTimeInMicroseconds();
        //         cudaDeviceSynchronize();
        //         spdlog::info("{} GPU_time: {}", azi_sector_num, GPU_end-GPU_begin);    
        //         // float_save_to_csv(filename + "mtd_abs_narrow_sum_test" + to_string(azi_sector_num) + ".csv", d_MTD_abs_narrow_sum, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
        //         // float_save_to_csv(filename + "mtd_abs_narrow_diff_test" + to_string(azi_sector_num) + ".csv", d_MTD_abs_narrow_diff, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
        //         // float_save_to_csv(filename + "mtd_abs_wide_sum_test" + to_string(azi_sector_num) + ".csv" , d_MTD_abs_wide_sum, prf_total_pulse*wide_points,prf_total_pulse,wide_points);
        //         // float_save_to_csv(filename + "mtd_abs_wide_diff_test" + to_string(azi_sector_num) + ".csv" , d_MTD_abs_wide_diff, prf_total_pulse*wide_points,prf_total_pulse,wide_points);
        //     }
        // }

        // 一个流处理一个流处理
        // void process_test1(cpi_data_t& cpi_data, cuComplex* d_ddc_data, int azi_sector_num, int debug, float* d_MTD_abs){
        //     string filename = "/home/ryh/ryh/cir_radar/data/cuda_";
        //     cudaDeviceSynchronize();
        //     long long GPU_begin = getTimeInMicroseconds();

        //     cuComplex alpha_dbf = make_cuComplex(1.0f, 0.0f);  // 矩阵乘法的标量因子 (实部: 1.0f, 虚部: 1.0f)
        //     cuComplex beta_dbf = make_cuComplex(0.0f, 0.0f);   // 矩阵C的初始值标量因子
        //     int strideA = 0; // A矩阵在批次中的步长
        //     int strideB = (chn_total) * (narrow_points + wide_points); // B矩阵在批次中的步长
        //     int strideC = narrow_pulse_pc_fft_points; // C矩阵在批次中的步长
        //     int batch_count = prf_total_pulse;
        //     int dbf_coe_offset = chn_total * azi_sector_num;  

        //     float *d_MTD_abs_narrow_sum = d_MTD_abs;
        //     float *d_MTD_abs_narrow_diff = d_MTD_abs_narrow_sum + prf_total_pulse * narrow_points;
        //     float *d_MTD_abs_wide_sum = d_MTD_abs_narrow_diff + prf_total_pulse * narrow_points;
        //     float *d_MTD_abs_wide_diff = d_MTD_abs_wide_sum + prf_total_pulse * wide_points;

        //     // CUDA 流 0 的操作 宽和
        //     cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, chn_total,
        //                 &alpha_dbf, d_dbf_sum_coe + dbf_coe_offset, chn_total, 0, d_ddc_data + chn_total * narrow_points,
        //                 wide_points, strideB, &beta_dbf, d_wide_sum_step1, 1, wide_pulse_pc_fft_points, batch_count);

        //     cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step1, (cufftComplex*)d_wide_sum_step1_f, CUFFT_FORWARD);
        //     pc_complexMultiply(d_wide_sum_step1_f, d_wide_coeff, d_wide_sum_step2, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points,streams[0]);
        //     cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step2, (cufftComplex*)d_wide_sum_step3, CUFFT_INVERSE);

        //     add_win(d_wide_sum_step3, d_win, d_wide_sum_add_win, prf_total_pulse * wide_pulse_pc_fft_points, wide_pulse_pc_fft_points, streams[0]);
        //     checkCufftError(cufftExecC2C(MTD_wide, d_wide_sum_add_win, d_MTD_wide_sum, CUFFT_FORWARD));
        //     MTD_abs(d_MTD_wide_sum, d_MTD_abs_wide_sum, prf_total_pulse * wide_points, streams[0]);

        //     // CUDA 流 1 的操作 宽差
        //     cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, chn_total,
        //                 &alpha_dbf, d_dbf_diff_coe + dbf_coe_offset, chn_total, 0, d_ddc_data + chn_total * narrow_points,
        //                 wide_points, strideB, &beta_dbf, d_wide_diff_step1, 1, wide_pulse_pc_fft_points, batch_count);

            
        //     cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step1, (cufftComplex*)d_wide_diff_step1_f, CUFFT_FORWARD);
        //     pc_complexMultiply(d_wide_diff_step1_f, d_wide_coeff, d_wide_diff_step2, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points, streams[0]);
        //     cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step2, (cufftComplex*)d_wide_diff_step3, CUFFT_INVERSE);    

        //     add_win(d_wide_diff_step3, d_win, d_wide_diff_add_win, prf_total_pulse * wide_pulse_pc_fft_points, wide_pulse_pc_fft_points, streams[0]);
        //     checkCufftError(cufftExecC2C(MTD_wide, d_wide_diff_add_win, d_MTD_wide_diff, CUFFT_FORWARD));
        //     MTD_abs(d_MTD_wide_diff, d_MTD_abs_wide_diff, prf_total_pulse * wide_points, streams[0]);

        //     // CUDA 流 2 的操作 窄和
        //     cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, chn_total,
        //                                 &alpha_dbf, d_dbf_sum_coe + dbf_coe_offset, chn_total, 0, d_ddc_data,
        //                                 narrow_points, strideB , &beta_dbf, d_narrow_sum_step1, 1, narrow_pulse_pc_fft_points, batch_count);

        //     cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step1, (cufftComplex*)d_narrow_sum_step1_f, CUFFT_FORWARD);
        //     pc_complexMultiply(d_narrow_sum_step1_f, d_narrow_coeff, d_narrow_sum_step2, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points, streams[1]);
        //     cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step2, (cufftComplex*)d_narrow_sum_step3, CUFFT_INVERSE);

        //     add_win(d_narrow_sum_step3, d_win, d_narrow_sum_add_win, prf_total_pulse * narrow_pulse_pc_fft_points, narrow_pulse_pc_fft_points, streams[1]);
        //     checkCufftError(cufftExecC2C(MTD_narrow, d_narrow_sum_add_win, d_MTD_narrow_sum, CUFFT_FORWARD));
        //     MTD_abs(d_MTD_narrow_sum, d_MTD_abs_narrow_sum, prf_total_pulse * narrow_points, streams[1]);

        //     // CUDA 流 3 的操作 窄差
        //     cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, chn_total,
        //                             &alpha_dbf, d_dbf_diff_coe + dbf_coe_offset, chn_total, 0, d_ddc_data,
        //                             narrow_points, strideB, &beta_dbf, d_narrow_diff_step1, 1, narrow_pulse_pc_fft_points, batch_count);

        //     cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step1, (cufftComplex*)d_narrow_diff_step1_f, CUFFT_FORWARD);
        //     pc_complexMultiply(d_narrow_diff_step1_f, d_narrow_coeff, d_narrow_diff_step2, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points, streams[1]);
        //     cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step2, (cufftComplex*)d_narrow_diff_step3, CUFFT_INVERSE);
            
        //     add_win(d_narrow_diff_step3, d_win, d_narrow_diff_add_win, prf_total_pulse * narrow_pulse_pc_fft_points, narrow_pulse_pc_fft_points, streams[1]);
        //     checkCufftError(cufftExecC2C(MTD_narrow, d_narrow_diff_add_win, d_MTD_narrow_diff, CUFFT_FORWARD));
        //     MTD_abs(d_MTD_narrow_diff, d_MTD_abs_narrow_diff, prf_total_pulse * narrow_points, streams[1]);
        //     cudaDeviceSynchronize();
        //     long long GPU_end = getTimeInMicroseconds();
        //     spdlog::info("{} GPU_time: {}", azi_sector_num, GPU_end-GPU_begin);    

        //     // float_save_to_csv(filename + "mtd_abs_narrow_sum_test" + to_string(azi_sector_num) + ".csv", d_MTD_abs_narrow_sum, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
        //     // float_save_to_csv(filename + "mtd_abs_narrow_diff_test" + to_string(azi_sector_num) + ".csv", d_MTD_abs_narrow_diff, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
        //     // float_save_to_csv(filename + "mtd_abs_wide_sum_test" + to_string(azi_sector_num) + ".csv" , d_MTD_abs_wide_sum, prf_total_pulse*wide_points,prf_total_pulse,wide_points);
        //     // float_save_to_csv(filename + "mtd_abs_wide_diff_test" + to_string(azi_sector_num) + ".csv" , d_MTD_abs_wide_diff, prf_total_pulse*wide_points,prf_total_pulse,wide_points);
        // }


        // 析构函数
        ~cuda_signal_porcess() {
            // cout << azi_sector_num_class <<"析构函数开始"<< endl;
            cudaDeviceSynchronize();
            spdlog::info("析构函数开始");               
            long long dispose_begin = getTimeInMicroseconds();
            // 内存释放
            cudaFreeHost(narrow_coeff);
            cudaFreeHost(wide_coeff);
            cudaFreeHost(dbf_coeff_sum);
            cudaFreeHost(dbf_coeff_diff);
            // 显存释放
            if (d_total_memory != nullptr) {
                cudaFree(d_total_memory);
            }
            // DBF系数显存释放
            cudaFree(d_dbf_sum_coe);
            cudaFree(d_dbf_diff_coe);
            // PC系数显存释放
            cudaFree(d_wide_coeff);
            cudaFree(d_narrow_coeff);
            // 窗系数显存释放
            cudaFree(d_win);
            // 句柄与流释放
            // cufftDestroy(pc_narrow);
            // cufftDestroy(pc_wide);
            // cufftDestroy(MTD_narrow);
            // cufftDestroy(MTD_wide);
            // // 销毁宽窄脉冲计算流
            // for (int i = 0; i < cuda_stream_num; i++) {
            //     // 销毁流
            //     cudaStreamDestroy(streams[i]);
            //     // 销毁cublas句柄
            //     cublasDestroy(handle_dbf[i]);
            // }        
            cudaStreamDestroy(stream_memcpy);
            long long dispose_end = getTimeInMicroseconds();
            spdlog::info("析构函数时间: {}", dispose_end-dispose_begin);
            // cout<< azi_sector_num_class <<"析构函数时间:"<<(dispose_end-dispose_begin)<<endl;
        }
};


