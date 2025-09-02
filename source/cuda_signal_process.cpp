#include "cuda_signal_process.hpp"


/*
    作用：GPU信号处理DBF，PC，MTD
        -参数：
            -cpi_data:完整的cpi数据
            -cuComplex* ddc_data_begin_64：完整的ddc数据起始位置指针
            -debug:调试参数

*/
int cuda_radar_process (cpi_data_t& cpi_data, cuComplex* ddc_data_begin_64, int debug){
    //设置进程优先级最高；
    // int retval = nice(-20);
    // -------------------构建cpi数据-----------------------------//
    long long d_begin = getTimeInMicroseconds();
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
    int pit_beam_num;
    std::cout << "PRF Total Pulse: " << prf_total_pulse 
          << ", Narrow Pulse PC FFT Points: " << narrow_pulse_pc_fft_points 
          << ", Wide Pulse PC FFT Points: " << wide_pulse_pc_fft_points 
          << ", Narrow Points: " << narrow_points 
          << ", Wide Points: " << wide_points << std::endl;


    // -------------------构建gpu数据-----------------------------//
    long long data_transfer_begin = getTimeInMicroseconds();
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("free_mem:%ld MB,total_mem:%ld MB,ddc_data size:%ld MB\n",free_mem/1024/1024,total_mem/1024/1024,sizeof(ddc_data_t)/1024/1024);

    // gpu内存分配
    // 找到ddc数据开始区域位置
    cuComplex* d_ddc_data = nullptr;

    ErrorCheck(cudaMalloc((void**)&d_ddc_data,prf_total_pulse * chn_total  * (narrow_points + wide_points)* sizeof(cuComplex)), __FILE__, __LINE__);

    // ddc数据迁移
    ErrorCheck(cudaMemcpy(d_ddc_data, ddc_data_begin_64 , prf_total_pulse * chn_total * (narrow_points + wide_points)* sizeof(cuComplex), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    // DBF系数数据迁移

    // PC系数数据迁移
    // 获得脉压系数
    cufftComplex* narrow_coeff = nullptr, *wide_coeff = nullptr;
    if(narrow_coeff == nullptr){
        narrow_coeff = (cufftComplex*)malloc(sizeof(cufftComplex) * narrow_pulse_pc_fft_points);
    }
    if(wide_coeff == nullptr){
        wide_coeff = (cufftComplex*)malloc(sizeof(cufftComplex) * wide_pulse_pc_fft_points);
    }

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

    cuComplex *d_wide_coeff = nullptr;
    cuComplex *d_narrow_coeff = nullptr;

    cudaMalloc((void**)&d_narrow_coeff, narrow_pulse_pc_fft_points * sizeof(cuComplex));
    cudaMalloc((void**)&d_wide_coeff, wide_pulse_pc_fft_points * sizeof(cuComplex));

    cudaMemcpy(d_narrow_coeff, narrow_coeff, narrow_pulse_pc_fft_points * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wide_coeff, wide_coeff, wide_pulse_pc_fft_points * sizeof(cuComplex), cudaMemcpyHostToDevice);
    long long data_transfer_end = getTimeInMicroseconds();
    cout<<"data_transfer_time:"<<(data_transfer_end - data_transfer_begin)<<endl; 
    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_d_ddc_data_Extension.csv" ,d_ddc_data ,(chn_total + chn_total/3 -1 ) * (narrow_points + wide_points) * 2, (chn_total + chn_total/3 -1 ) * 2, (narrow_points + wide_points));
    // printf("%f\n",cpi_data.ddc_data->prf[25].narrow.chn[0]->real());

    // ------------------- GPU信号处理 -----------------------------//
    // ------------------- DBF ---------------------------//

    // ----------------------初始化-------------------//
    // 一次性分配内存
    long long init_begin = getTimeInMicroseconds();
    size_t GPU_DBF_size = prf_total_pulse * (narrow_pulse_pc_fft_points + wide_pulse_pc_fft_points) * sizeof(cuComplex) * 4;
    size_t GPU_PC_size = GPU_DBF_size * 3;
    size_t GPU_WIN_size = GPU_DBF_size;
    size_t GPU_MTD_size = prf_total_pulse * (narrow_points + wide_points) * sizeof(cuComplex) * 4;
    size_t GPU_size = GPU_DBF_size +  GPU_PC_size + GPU_WIN_size + GPU_MTD_size;

    cuComplex* d_total_memory = nullptr;
    ErrorCheck(cudaMalloc((void**)&d_total_memory, GPU_size), __FILE__, __LINE__);
    ErrorCheck(cudaMemset(d_total_memory, 0, GPU_size), __FILE__, __LINE__);

    cuComplex* d_narrow_sum_step1 = d_total_memory;  // 偏移为0，指向大块内存的开始
    cuComplex* d_narrow_diff_step1 = d_narrow_sum_step1 + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_sum_step1 = d_narrow_diff_step1 + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_diff_step1 = d_wide_sum_step1 + prf_total_pulse * wide_pulse_pc_fft_points;

    cuComplex* d_narrow_sum_step1_f = d_wide_diff_step1 + prf_total_pulse * wide_pulse_pc_fft_points;
    cuComplex* d_narrow_diff_step1_f = d_narrow_sum_step1_f + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_sum_step1_f = d_narrow_diff_step1_f + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_diff_step1_f = d_wide_sum_step1_f + prf_total_pulse * wide_pulse_pc_fft_points;

    cuComplex* d_narrow_sum_step2 = d_wide_diff_step1_f + prf_total_pulse * wide_pulse_pc_fft_points;
    cuComplex* d_narrow_diff_step2 = d_narrow_sum_step2 + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_sum_step2 = d_narrow_diff_step2 + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_diff_step2 = d_wide_sum_step2 + prf_total_pulse * wide_pulse_pc_fft_points;

    cuComplex* d_narrow_sum_step3 = d_wide_diff_step2 + prf_total_pulse * wide_pulse_pc_fft_points;
    cuComplex* d_narrow_diff_step3 = d_narrow_sum_step3 + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_sum_step3 = d_narrow_diff_step3 + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_diff_step3 = d_wide_sum_step3 + prf_total_pulse * wide_pulse_pc_fft_points;

    cuComplex* d_narrow_sum_add_win = d_wide_diff_step3 + prf_total_pulse * wide_pulse_pc_fft_points;
    cuComplex* d_narrow_diff_add_win = d_narrow_sum_add_win + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_sum_add_win = d_narrow_diff_add_win + prf_total_pulse * narrow_pulse_pc_fft_points;
    cuComplex* d_wide_diff_add_win = d_wide_sum_add_win + prf_total_pulse * wide_pulse_pc_fft_points;

    cuComplex* d_MTD_narrow_sum = d_wide_diff_add_win + prf_total_pulse * wide_pulse_pc_fft_points;
    cuComplex* d_MTD_narrow_diff = d_MTD_narrow_sum + prf_total_pulse * narrow_points;
    cuComplex* d_MTD_wide_sum = d_MTD_narrow_diff + prf_total_pulse * narrow_points;
    cuComplex* d_MTD_wide_diff = d_MTD_wide_sum + prf_total_pulse * wide_points;

    // float
    size_t GPU_MTD_abs = prf_total_pulse * (narrow_points + wide_points) * sizeof(float) * 4;
    float *d_MTD_abs = nullptr;
    ErrorCheck(cudaMalloc((void**)&d_MTD_abs, GPU_MTD_abs), __FILE__, __LINE__);
    // MTD_abs
    float *d_MTD_abs_narrow_sum = d_MTD_abs;
    float *d_MTD_abs_narrow_diff = d_MTD_abs_narrow_sum + prf_total_pulse * narrow_points;
    float *d_MTD_abs_wide_sum = d_MTD_abs_narrow_diff + prf_total_pulse * narrow_points;
    float *d_MTD_abs_wide_diff = d_MTD_abs_wide_sum + prf_total_pulse * wide_points;

    // cuda流建立
    int cuda_stream_num = 2;
    // 先窄后宽
    cudaStream_t streams[cuda_stream_num];
    cublasHandle_t handle_dbf[cuda_stream_num];
    // pc句柄
    cufftHandle pc_narrow;
    cufftHandle pc_wide;
    checkCufftError(cufftPlan1d(&pc_narrow, narrow_pulse_pc_fft_points, CUFFT_C2C, prf_total_pulse));
    checkCufftError(cufftPlan1d(&pc_wide, wide_pulse_pc_fft_points, CUFFT_C2C, prf_total_pulse));
    // MTD句柄
    // MTD的FFT句柄
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

    cufftHandle MTD_narrow;
    cufftHandle MTD_wide;

    checkCufftError(cufftPlanMany(&MTD_narrow, MTD_narrow_rank, MTD_narrow_n, MTD_narrow_inembed, MTD_narrow_istride, MTD_narrow_idist, MTD_narrow_onembed, MTD_narrow_ostride, MTD_narrow_odist, CUFFT_C2C, MTD_narrow_batch));
    checkCufftError(cufftPlanMany(&MTD_wide, MTD_wide_rank, MTD_wide_n, MTD_wide_inembed, MTD_wide_istride, MTD_wide_idist, MTD_wide_onembed, MTD_wide_ostride, MTD_wide_odist, CUFFT_C2C, MTD_wide_batch));
    for (int i = 0; i < cuda_stream_num; i++) {
        cudaStreamCreate(&streams[i]);
        cublasCreate(&handle_dbf[i]);
        // 将每个句柄与对应的流绑定
        cublasSetStream(handle_dbf[i], streams[i]);

    }
    // 将FFT句柄与流绑定
    checkCufftError(cufftSetStream(pc_narrow, streams[0]));  // 窄FFT绑定流
    checkCufftError(cufftSetStream(pc_wide, streams[1]));    // 宽FFT绑定流

    // 将MTD句柄与流绑定
    checkCufftError(cufftSetStream(MTD_narrow, streams[0]));  // 窄MTD绑定流
    checkCufftError(cufftSetStream(MTD_wide, streams[1]));    // 宽MTD绑定流
    long long init_end = getTimeInMicroseconds();
    cout << "init_time:" << init_end - init_begin << endl;
    //加载DDC数据,DBF计算

    // cout << "dbf_coeff_sum.rows:" << dbf_coeff_sum.rows() <<
    //  " dbf_coeff_sum.cols:" << dbf_coeff_sum.cols() << " sector_narrow.rows:" << sector_narrow.rows() <<
    //  " sector_narrow.cols:" << sector_narrow.cols() << endl;
    // ----------------------------------dbf------------------------------ //
    //初始化DBF系数和DBF组合
    int dbf_channel_number = 12;
    MatrixDBF_coeff dbf_coeff_sum(1,dbf_channel_number);
    MatrixDBF_coeff dbf_coeff_diff(1,dbf_channel_number);
    volatile uint8_t dbf_group[dbf_channel_number];
    complex<float> coeff_tmp(0,0);
    // 先进行一个扇区的DBF
    // 这边考虑三进程，因为三个扇区的dbf系数不同，三个扇区构建的数据不同
    // 函数封装，引用方式传入一个cpi_data,第azi_sector_num扇区数据，根据azi_sector_num计算cuda流和dbf系数，int debug，输出target_map_t，接收的时候用数组接收
    long long GPU_begin = getTimeInMicroseconds();
    int azi_sector_num = 0; 
    // cpu的dbf数据结构
    dbf_t dbf_data;
    dbf_data.init(prf_total_pulse,narrow_points,wide_points);
    for (int chn_index = 0; chn_index < dbf_channel_number; chn_index++)
    {
        // 
        coeff_tmp.real((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][0][chn_index * 2]); 
        coeff_tmp.imag((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][0][chn_index * 2 + 1]);
        dbf_coeff_sum.col(chn_index)<<coeff_tmp;

        coeff_tmp.real((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][1][chn_index * 2]);
        coeff_tmp.imag((float)cpi_data.cmd_params->DBF_coeff[azi_sector_num][1][chn_index * 2 + 1]);
        dbf_coeff_diff.col(chn_index)<<coeff_tmp;
        dbf_group[chn_index] = cpi_data.cmd_params->DBF_group[azi_sector_num][chn_index] - 1;
    }
    cout << "dbf_chn_begin:" << int(dbf_group[0]) << endl;
    long long d_dbf_test_2_begin = getTimeInMicroseconds();
    long long dbf_coe_begin = getTimeInMicroseconds();
    cuComplex *d_dbf_sum_coe = nullptr;
    cuComplex *d_dbf_diff_coe= nullptr;

    cudaMalloc((void**)&d_dbf_sum_coe, dbf_channel_number * sizeof(cuComplex));
    cudaMalloc((void**)&d_dbf_diff_coe, dbf_channel_number * sizeof(cuComplex));

    cudaMemcpy(d_dbf_sum_coe, dbf_coeff_sum.data(), dbf_channel_number * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dbf_diff_coe, dbf_coeff_diff.data(), dbf_channel_number * sizeof(cuComplex), cudaMemcpyHostToDevice);
    long long dbf_coe_end = getTimeInMicroseconds();
    cout << "dbf_coe_time:" << dbf_coe_end - dbf_coe_begin << endl;

    // alpha 用于对矩阵乘法的结果进行缩放，计算时，C = alpha * A * B + beta * C。
    // beta 是用来缩放矩阵 C 原本的值，默认通常是 0.0f，表示不考虑 C 原本的内容（即仅计算 A * B）。
    cuComplex alpha_dbf = make_cuComplex(1.0f, 0.0f);  // 矩阵乘法的标量因子 (实部: 1.0f, 虚部: 1.0f)
    cuComplex beta_dbf = make_cuComplex(0.0f, 0.0f);   // 矩阵C的初始值标量因子

    if (dbf_group[0] > 24){
        int strideA = 0; // A矩阵在批次中的步长
        int strideB = (chn_total) * (narrow_points + wide_points); // B矩阵在批次中的步长
        int strideC = narrow_pulse_pc_fft_points; // C矩阵在批次中的步长
        int batch_count = prf_total_pulse;

        // 上半部分通道数
        int dbf_half_upper = chn_total - dbf_group[0];
        // 下半部分通道数
        int dbf_half_lowwer = dbf_channel_number - dbf_half_upper;
        cout<< "dbf_half_upper:" << dbf_half_upper << " dbf_half_lowwer:" << dbf_half_lowwer << endl;
   
        cuComplex *d_narrow_sum_step1_half = nullptr;
        cuComplex *d_narrow_diff_step1_half = nullptr;
        cudaMalloc((void**)&d_narrow_sum_step1_half , prf_total_pulse * narrow_pulse_pc_fft_points * sizeof(cuComplex));
        cudaMalloc((void**)&d_narrow_diff_step1_half , prf_total_pulse * narrow_pulse_pc_fft_points * sizeof(cuComplex));

        cuComplex *d_wide_sum_step1_half = nullptr;
        cuComplex *d_wide_diff_step1_half = nullptr;
        cudaMalloc((void**)&d_wide_sum_step1_half, prf_total_pulse * wide_pulse_pc_fft_points * sizeof(cuComplex));
        cudaMalloc((void**)&d_wide_diff_step1_half, prf_total_pulse * wide_pulse_pc_fft_points * sizeof(cuComplex));

        // 前半部分
        cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, dbf_half_upper,
                                &alpha_dbf, d_dbf_sum_coe, dbf_half_upper, 0, d_ddc_data + dbf_group[0] * narrow_points ,
                                narrow_points, strideB , &beta_dbf, d_narrow_sum_step1, 1, narrow_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, dbf_half_upper,
                                &alpha_dbf, d_dbf_diff_coe, dbf_half_upper, 0, d_ddc_data +  dbf_group[0] * narrow_points ,
                                narrow_points, strideB, &beta_dbf, d_narrow_diff_step1, 1, narrow_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, dbf_half_upper,
                                &alpha_dbf, d_dbf_sum_coe, dbf_half_upper, 0, d_ddc_data + dbf_group[0] * wide_points  + (chn_total ) * narrow_points,
                                wide_points, strideB, &beta_dbf, d_wide_sum_step1, 1, wide_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, dbf_half_upper,
                                &alpha_dbf, d_dbf_diff_coe, dbf_half_upper, 0, d_ddc_data + dbf_group[0] * wide_points + (chn_total) * narrow_points,
                                wide_points, strideB, &beta_dbf, d_wide_diff_step1, 1, wide_pulse_pc_fft_points, batch_count);

        // 后半部分
        cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, dbf_half_lowwer,
                                &alpha_dbf, d_dbf_sum_coe + dbf_half_upper, dbf_half_lowwer, 0, d_ddc_data,
                                narrow_points, strideB , &beta_dbf, d_narrow_sum_step1_half, 1, narrow_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, dbf_half_lowwer,
                                &alpha_dbf, d_dbf_diff_coe + dbf_half_upper, dbf_half_lowwer, 0, d_ddc_data,
                                narrow_points, strideB, &beta_dbf, d_narrow_diff_step1_half, 1, narrow_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, dbf_half_lowwer,
                                &alpha_dbf, d_dbf_sum_coe + dbf_half_upper, dbf_half_lowwer, 0, d_ddc_data + (chn_total ) * narrow_points,
                                wide_points, strideB, &beta_dbf, d_wide_sum_step1_half, 1, wide_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, dbf_half_lowwer,
                                &alpha_dbf, d_dbf_diff_coe + dbf_half_upper, dbf_half_lowwer, 0, d_ddc_data + (chn_total) * narrow_points,
                                wide_points, strideB, &beta_dbf, d_wide_diff_step1_half, 1, wide_pulse_pc_fft_points, batch_count);


        // 相加
        complexAdd(d_narrow_sum_step1, d_narrow_sum_step1_half, prf_total_pulse*narrow_pulse_pc_fft_points,streams[0]);
        complexAdd(d_narrow_diff_step1, d_narrow_diff_step1_half,  prf_total_pulse*narrow_pulse_pc_fft_points, streams[0]);
        complexAdd(d_wide_sum_step1, d_wide_sum_step1_half, prf_total_pulse*wide_pulse_pc_fft_points,streams[1]);
        complexAdd(d_wide_diff_step1, d_wide_diff_step1_half,  prf_total_pulse*wide_pulse_pc_fft_points, streams[1]);
        // 显存释放
        cudaFree(d_narrow_sum_step1_half);
        cudaFree(d_narrow_diff_step1_half);
        cudaFree(d_wide_sum_step1_half);
        cudaFree(d_wide_diff_step1_half);
    }
    else { 
        int strideA = 0; // A矩阵在批次中的步长
        int strideB = (chn_total) * (narrow_points + wide_points); // B矩阵在批次中的步长
        int strideC = narrow_pulse_pc_fft_points; // C矩阵在批次中的步长
        int batch_count = prf_total_pulse;

        cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, dbf_channel_number,
                                &alpha_dbf, d_dbf_sum_coe, dbf_channel_number, 0, d_ddc_data + dbf_group[0] * narrow_points ,
                                narrow_points, strideB , &beta_dbf, d_narrow_sum_step1, 1, narrow_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[0], CUBLAS_OP_T, CUBLAS_OP_T, 1, narrow_points, dbf_channel_number,
                                &alpha_dbf, d_dbf_diff_coe, dbf_channel_number, 0, d_ddc_data +  dbf_group[0] * narrow_points ,
                                narrow_points, strideB, &beta_dbf, d_narrow_diff_step1, 1, narrow_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, dbf_channel_number,
                                &alpha_dbf, d_dbf_sum_coe, dbf_channel_number, 0, d_ddc_data + dbf_group[0] * wide_points  + (chn_total ) * narrow_points,
                                wide_points, strideB, &beta_dbf, d_wide_sum_step1, 1, wide_pulse_pc_fft_points, batch_count);

        cublasCgemmStridedBatched(handle_dbf[1], CUBLAS_OP_T, CUBLAS_OP_T, 1, wide_points, dbf_channel_number,
                                &alpha_dbf, d_dbf_diff_coe, dbf_channel_number, 0, d_ddc_data + dbf_group[0] * wide_points + (chn_total) * narrow_points,
                                wide_points, strideB, &beta_dbf, d_wide_diff_step1, 1, wide_pulse_pc_fft_points, batch_count);


    }

    long long d_dbf_test_2_end = getTimeInMicroseconds();
    cout<<"gpu_dbf_test_2_time:"<<(d_dbf_test_2_end-d_dbf_test_2_begin)<<endl;
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_dbf_narrow_sum_2.csv", d_narrow_sum_step1, prf_total_pulse * narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_dbf_narrow_diff_2.csv", d_narrow_diff_step1, prf_total_pulse * narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_dbf_wide_sum_2.csv", d_wide_sum_step1, prf_total_pulse * wide_pulse_pc_fft_points,prf_total_pulse, wide_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_dbf_wide_diff_2.csv", d_wide_diff_step1, prf_total_pulse * wide_pulse_pc_fft_points,prf_total_pulse, wide_pulse_pc_fft_points); 

    // ------------------- PC ---------------------------//
    // 脉压step1：信号FFT
    // FFT
    //cudaDeviceSynchronize();
    long long d_PC_begin = getTimeInMicroseconds();
    cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step1, (cufftComplex*)d_narrow_sum_step1_f, CUFFT_FORWARD);
    cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step1, (cufftComplex*)d_narrow_diff_step1_f, CUFFT_FORWARD);
    cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step1, (cufftComplex*)d_wide_sum_step1_f, CUFFT_FORWARD);
    cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step1, (cufftComplex*)d_wide_diff_step1_f, CUFFT_FORWARD);

    // // cudaDeviceSynchronize();
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_narrow_sum_pc_1.csv", d_narrow_sum_step1_f, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_narrow_diff_pc_1.csv", d_narrow_diff_step1_f, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_wide_sum_pc_1.csv", d_wide_sum_step1_f, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_wide_diff_pc_1.csv", d_wide_diff_step1_f, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
    // 脉压step2：信号相乘
    // 将本地信号的FFT系数给GPU
    // GPU对应数据相乘
    // cudaDeviceSynchronize();
    pc_complexMultiply(d_narrow_sum_step1_f, d_narrow_coeff, d_narrow_sum_step2, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points,streams[0]);
    pc_complexMultiply(d_narrow_diff_step1_f, d_narrow_coeff, d_narrow_diff_step2, narrow_pulse_pc_fft_points * prf_total_pulse, narrow_pulse_pc_fft_points,streams[0]);
    pc_complexMultiply(d_wide_sum_step1_f, d_wide_coeff, d_wide_sum_step2, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points,streams[1]);
    pc_complexMultiply(d_wide_diff_step1_f, d_wide_coeff, d_wide_diff_step2, wide_pulse_pc_fft_points * prf_total_pulse, wide_pulse_pc_fft_points,streams[1]);
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_narrow_sum_pc_2.csv", d_narrow_sum_step2, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_narrow_diff_pc_2.csv", d_narrow_diff_step2, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_wide_sum_pc_2.csv", d_wide_sum_step2, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_wide_diff_pc_2.csv", d_wide_diff_step2, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 


    // CPU 上的执行cudaMalloc与在 GPU 上的内核执行pc_complexMultiply重叠，默认流中的内核执行与主机上的代码执行重叠
    // cudaDeviceSynchronize();
    cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_sum_step2, (cufftComplex*)d_narrow_sum_step3, CUFFT_INVERSE);
    cufftExecC2C(pc_narrow, (cufftComplex*)d_narrow_diff_step2, (cufftComplex*)d_narrow_diff_step3, CUFFT_INVERSE);
    cufftExecC2C(pc_wide, (cufftComplex*)d_wide_sum_step2, (cufftComplex*)d_wide_sum_step3, CUFFT_INVERSE);
    cufftExecC2C(pc_wide, (cufftComplex*)d_wide_diff_step2, (cufftComplex*)d_wide_diff_step3, CUFFT_INVERSE);
    long long d_PC_end = getTimeInMicroseconds();
    cout<<"gpu_PC_time:"<<( d_PC_end- d_PC_begin)<<endl;
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_narrow_sum_pc_3.csv", d_narrow_sum_step3, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_narrow_diff_pc_3.csv", d_narrow_diff_step3, prf_total_pulse* narrow_pulse_pc_fft_points,prf_total_pulse,narrow_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_wide_sum_pc_3.csv", d_wide_sum_step3, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points); 
    // cufftComplex_save_to_csv("/home/ryh/ryh/cir_radar/data/cuda_wide_diff_pc_3.csv", d_wide_diff_step3, prf_total_pulse* wide_pulse_pc_fft_points,prf_total_pulse,wide_pulse_pc_fft_points);


    // ------------------------------ 加窗 ----------------------------
    long long d_MTD_begin = getTimeInMicroseconds();
    typedef Map<Matrix<float, 1, Dynamic, RowMajor>>  MapWin;
    win_type_t curr_win_type = Taylor;
    MapWin win(radar_signal_process::rsl_win_func_manager::getInstance().get_win_func(curr_win_type,prf_total_pulse,60),1,prf_total_pulse);
    // cout<<"win_rows: "<<win.rows()<<"win_cols: " <<win.cols()<<endl;
    

    float* d_win = nullptr;
    cudaMalloc((void**)&d_win, prf_total_pulse*sizeof(float));
    cudaMemcpy(d_win, win.data(), prf_total_pulse*sizeof(float), cudaMemcpyHostToDevice);
    // writeFloatToFile( "/home/ryh/radar/cir_radar/data/d_win.csv", d_win, prf_total_pulse);

    // cudaDeviceSynchronize();
    add_win(d_narrow_sum_step3,d_win,d_narrow_sum_add_win,prf_total_pulse*narrow_pulse_pc_fft_points,narrow_pulse_pc_fft_points,streams[0]);
    add_win(d_narrow_diff_step3,d_win,d_narrow_diff_add_win,prf_total_pulse*narrow_pulse_pc_fft_points,narrow_pulse_pc_fft_points,streams[0]);
    add_win(d_wide_sum_step3,d_win,d_wide_sum_add_win,prf_total_pulse*wide_pulse_pc_fft_points,wide_pulse_pc_fft_points,streams[1]);
    add_win(d_wide_diff_step3,d_win,d_wide_diff_add_win,prf_total_pulse*wide_pulse_pc_fft_points,wide_pulse_pc_fft_points,streams[1]);

    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_narrow_sum_addwin.csv" , d_narrow_sum_add_win, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_narrow_diff_addwin.csv" , d_narrow_diff_add_win, prf_total_pulse*narrow_points,prf_total_pulse,narrow_points);
    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_wide_sum_addwin.csv" , d_wide_sum_add_win, prf_total_pulse*wide_points,prf_total_pulse,wide_points);
    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_wide_diff_addwin.csv" , d_wide_diff_add_win, prf_total_pulse*wide_points,prf_total_pulse,wide_points);

    // --------------------- MTD ----------------------------

    // cudaDeviceSynchronize();
    checkCufftError(cufftExecC2C(MTD_narrow, d_narrow_sum_add_win, d_MTD_narrow_sum, CUFFT_FORWARD));
    cufftExecC2C(MTD_narrow, d_narrow_diff_add_win, d_MTD_narrow_diff, CUFFT_FORWARD);
    cufftExecC2C(MTD_wide, d_wide_sum_add_win, d_MTD_wide_sum, CUFFT_FORWARD);
    cufftExecC2C(MTD_wide, d_wide_diff_add_win, d_MTD_wide_diff, CUFFT_FORWARD);
    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_MTD_narrow_sum.csv" , d_MTD_narrow_sum, prf_total_pulse*narrow_points,narrow_points,prf_total_pulse);
    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_MTD_narrow_diff.csv" , d_MTD_narrow_diff, prf_total_pulse*narrow_points,narrow_points,prf_total_pulse);
    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_MTD_wide_sum.csv" , d_MTD_wide_sum, prf_total_pulse*wide_points,wide_points,prf_total_pulse);
    // cufftComplex_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_MTD_wide_diff.csv" , d_MTD_wide_diff, prf_total_pulse*wide_points,wide_points,prf_total_pulse);
    // abs

    // cudaDeviceSynchronize();
    MTD_abs(d_MTD_narrow_sum,d_MTD_abs_narrow_sum,prf_total_pulse * narrow_points,streams[0]);
    MTD_abs(d_MTD_narrow_diff,d_MTD_abs_narrow_diff,prf_total_pulse * narrow_points,streams[0]);
    MTD_abs(d_MTD_wide_sum,d_MTD_abs_wide_sum,prf_total_pulse * wide_points,streams[1]);
    MTD_abs(d_MTD_wide_diff,d_MTD_abs_wide_diff,prf_total_pulse * wide_points,streams[1]);
    long long d_MTD_end = getTimeInMicroseconds();
    cout<<"d_MTD_time:"<<(d_MTD_end-d_MTD_begin)<<endl;
    float_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_MTD_abs_narrow_sum_1.csv" , d_MTD_abs_narrow_sum, prf_total_pulse*narrow_points,narrow_points,prf_total_pulse);
    float_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_MTD_abs_narrow_diff_1.csv" , d_MTD_abs_narrow_diff, prf_total_pulse*narrow_points,narrow_points,prf_total_pulse);
    float_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_MTD_abs_wide_sum_1.csv" , d_MTD_abs_wide_sum, prf_total_pulse*wide_points,wide_points,prf_total_pulse);
    float_save_to_csv( "/home/ryh/ryh/cir_radar/data/cuda_MTD_abs_wide_diff_1.csv" , d_MTD_abs_wide_diff, prf_total_pulse*wide_points,wide_points,prf_total_pulse);
    long long GPU_end = getTimeInMicroseconds();
    long long d_end = getTimeInMicroseconds();
    cout<<"whole_time:"<<(d_end-d_begin)<<endl;
    cout<<"GPU_time:"<<(GPU_end -GPU_begin)<<endl;
    // -------------------数据保存--------------------------------//
    // long long data_save_begin = getTimeInMicroseconds();
    // // DBF
    // if (dbf_group[0] > 24){
    //     for(int i = 0;i < prf_total_pulse;i++){
    //         ErrorCheck(cudaMemcpy(dbf_data.narrow_sum[i], d_narrow_sum_step1 +  i * narrow_pulse_pc_fft_points, narrow_points * sizeof(cuComplex), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    //         ErrorCheck(cudaMemcpy(dbf_data.narrow_diff[i], d_narrow_diff_step1 +  i * narrow_pulse_pc_fft_points, narrow_points * sizeof(cuComplex), cudaMemcpyDeviceToHost),__FILE__, __LINE__);       
    //         ErrorCheck(cudaMemcpy(dbf_data.wide_sum[i], d_wide_sum_step1 +  i * wide_pulse_pc_fft_points, wide_points * sizeof(cuComplex), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    //         ErrorCheck(cudaMemcpy(dbf_data.wide_diff[i], d_wide_diff_step1 +  i * wide_pulse_pc_fft_points, wide_points * sizeof(cuComplex), cudaMemcpyDeviceToHost),__FILE__, __LINE__);        
    //     }
    // }


    // // PC
    // pc_t pc_data;
    // pc_data.init(prf_total_pulse,narrow_points,wide_points);
    // // gpu数据转置
    // cuComplex *d_narrow_sum_step3_tr = nullptr;
    // cuComplex *d_narrow_diff_step3_tr = nullptr;
    // cuComplex *d_wide_sum_step3_tr = nullptr;
    // cuComplex *d_wide_diff_step3_tr = nullptr;
    
    // cudaMalloc((void**)&d_narrow_sum_step3_tr, prf_total_pulse*narrow_pulse_pc_fft_points * sizeof(cuComplex));
    // cudaMalloc((void**)&d_narrow_diff_step3_tr, prf_total_pulse*narrow_pulse_pc_fft_points * sizeof(cuComplex));
    // cudaMalloc((void**)&d_wide_sum_step3_tr, prf_total_pulse*wide_pulse_pc_fft_points * sizeof(cuComplex));
    // cudaMalloc((void**)&d_wide_diff_step3_tr, prf_total_pulse*wide_pulse_pc_fft_points * sizeof(cuComplex));

    // transpose(d_narrow_sum_step3, d_narrow_sum_step3_tr, prf_total_pulse, narrow_points);
    // transpose(d_narrow_diff_step3, d_narrow_diff_step3_tr, prf_total_pulse, narrow_points);
    // transpose(d_wide_sum_step3, d_wide_sum_step3_tr, prf_total_pulse, wide_points);
    // transpose(d_wide_diff_step3, d_wide_diff_step3_tr, prf_total_pulse, wide_points);

    // for(int i = 0;i < narrow_points;i++){
    //     ErrorCheck(cudaMemcpy(pc_data.narrow_sum[i], d_narrow_sum_step3_tr +  i * prf_total_pulse, prf_total_pulse * sizeof(float), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    //     ErrorCheck(cudaMemcpy(pc_data.narrow_diff[i], d_narrow_diff_step3_tr +  i * prf_total_pulse, prf_total_pulse * sizeof(float), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    // }
    // for(int i = 0;i < wide_points;i++){
    //     ErrorCheck(cudaMemcpy(pc_data.wide_sum[i], d_wide_sum_step3_tr +  i * prf_total_pulse, prf_total_pulse * sizeof(float), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    //     ErrorCheck(cudaMemcpy(pc_data.wide_diff[i], d_wide_diff_step3_tr +  i * prf_total_pulse, prf_total_pulse * sizeof(float), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    // }

    // // MTD_abs
    // long long MTD_data_save_begin = getTimeInMicroseconds();
    // mtd_abs_t save_MTD_abs_data;
    // save_MTD_abs_data.init(prf_total_pulse,narrow_points,wide_points);
    // for(int i = 0;i < narrow_points;i++){
    //     ErrorCheck(cudaMemcpy(save_MTD_abs_data.narrow_sum[i], d_MTD_abs_narrow_sum +  i * prf_total_pulse, prf_total_pulse * sizeof(float), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    //     ErrorCheck(cudaMemcpy(save_MTD_abs_data.narrow_diff[i], d_MTD_abs_narrow_diff +  i * prf_total_pulse, prf_total_pulse * sizeof(float), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    // }
    // for(int i = 0;i < wide_points;i++){
    //     ErrorCheck(cudaMemcpy(save_MTD_abs_data.wide_sum[i], d_MTD_abs_wide_sum +  i * prf_total_pulse, prf_total_pulse * sizeof(float), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    //     ErrorCheck(cudaMemcpy(save_MTD_abs_data.wide_diff[i], d_MTD_abs_wide_diff +  i * prf_total_pulse, prf_total_pulse * sizeof(float), cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    // }
    // long long MTD_data_save_end = getTimeInMicroseconds();
    // cout<<"MTD_data_save_time:"<<(MTD_data_save_end - MTD_data_save_begin)<<endl;   
    // long long data_save_end = getTimeInMicroseconds();
    // cout<<"data_save_time:"<<(data_save_end -data_save_begin)<<endl;
    // // CFAR
    // long long cfar_begin = getTimeInMicroseconds();
    // target_map_t target_tmp;
    // radar_signal_process::rsl_cfar cfar;
    // clutter_map_t clutter_map_data;
    // cfar.cfar(cpi_data, save_MTD_abs_data, clutter_map_data, azi_sector_num, target_tmp);
    // long long cfar_end = getTimeInMicroseconds();
    // cout<<"cfar_time:"<<(cfar_end-cfar_begin)<<endl;
    //============================数据保存==================================
    // rsl_rsp_data_save& data_save = rsl_rsp_data_save::getInstance();
    // data_save.save(cpi_data, dbf_data, pc_data, save_MTD_abs_data, azi_beam_num, pit_beam_num);

    //============================释放资源==================================
    // dbf_data.dispose();
    // pc_data.dispose();
    // save_MTD_abs_data.dispose();

    // -------------------cuda信号处理----------------------------//


    // -------------------内存释放-------------------------------//
    // cpu内存释放
    free(narrow_coeff);
    free(wide_coeff);

    // gpu内存释放
    cudaFree(d_ddc_data);
    // dbf_test
    cudaFree(d_dbf_sum_coe);
    cudaFree(d_dbf_diff_coe);
    // PC
    cudaFree(d_wide_coeff);
    cudaFree(d_narrow_coeff);
    // cudaFree(d_narrow_sum_step3_tr);
    // cudaFree(d_narrow_diff_step3_tr);
    // cudaFree(d_wide_sum_step3_tr);
    // cudaFree(d_wide_diff_step3_tr);
    
    // 加窗
    cudaFree(d_win);
    cudaFree(d_total_memory);
    cudaFree(d_MTD_abs);
    // 转置
    // FFT句柄释放
    cufftDestroy(pc_narrow);
    cufftDestroy(pc_wide);
    cufftDestroy(MTD_narrow);
    cufftDestroy(MTD_wide);
    for (int i = 0; i < cuda_stream_num; i++) {
        // 销毁流
        cudaStreamDestroy(streams[i]);

        // 销毁cublas句柄
        cublasDestroy(handle_dbf[i]);
    }

    return 1;
}


// 
