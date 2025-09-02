--------------------------------------------------------
1. 怎么将从CPU数据导入到GPU
ddc数据结构如下所示：
#pragma pack(1)
typedef struct DDC_DATA{
    prf_data_t prf[MAX_PRF_PULSE];
}ddc_data_t;
#pragma pack()
#pragma pack(1)
typedef struct PRF_DATA{
    pulse_data_t narrow;
    pulse_data_t wide;
}prf_data_t;
#pragma pack(1)
typedef struct PULSE_DATA{
    complex<float>* chn[36];
}pulse_data_t;
#pragma pack()
MAX_PRF_PULSE = 32768,ddc_data->prf[MAX_PRF_PULSE].narrow.chn[i]->real()),怎么将这个数据导入到GPU里，并且GPU里的数据结构也是这样的呢？
----------------------------------------------------------------
2. DDC数据
2048个脉冲，36通道，一个通道宽脉冲320个点，窄脉冲144个点,浮点复数
prf[2048].chn[36].320+144个浮点复数采样点
2048*36*（320+144）= 34209792个数据
-------------------------------------------------------------------
3. DBF
[3*12]个数据dbf系数，三个dbf系数组，一组12个dbf系数矩阵，dbf系数矩阵[1*12]，矩阵乘法[1*12]DBF系数*DDC数据[12*2048]数据，变成[1*20248]数据。三组变成[3*2048]数据。
CHECK_CUBLAS(cublasCgemm_v2(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_T,  // transa 和 transb 设置为 CUBLAS_OP_T
    m, n, k,                  // 矩阵尺寸
    &alpha,                   // 标量 alpha
    d_A, k,                   // A, lda 设置为 k（A的列数）
    d_B, n,                   // B, ldb 设置为 n（B的列数）
    &beta,                    // 标量 beta
    d_C, m                    // C, ldc 设置为 m（C的行数）
));
1）.注意这个函数是优先列存储的，我们的变量是行优先的，要转置 CUBLAS_OP_T, CUBLAS_OP_T，对应的lda和ldb也要改 2）cuComplex alpha = make_cuComplex(1.0f, 0.0f); 
-----------------------------------------------------------------------
4. PC
对dbf后的数据进行补0操作？
开辟一个更大的内存空间，将dbf后的数据拷贝过去，多余的内存空间自然变成0
------------------------------------------------------------------------
5.pc的IFFT步骤转置
// 脉压step3：IFFT
int PC_IFFT_narrow_rank = 1;   //  维
int PC_IFFT_narrow_n[1] = { prf_total_pulse };    // 每一维变换数目
int PC_IFFT_narrow_inembed[2] = { narrow_pulse_pc_fft_points, prf_total_pulse }; // 输入数据的步幅
int PC_IFFT_narrow_istride = narrow_pulse_pc_fft_points; // 输入每个 FFT 的数据步幅
int PC_IFFT_narrow_idist = 1; // 不同 FFT 之间的距离
int PC_IFFT_narrow_onembed[2] = { prf_total_pulse, narrow_points }; // 输出数据的步幅
int PC_IFFT_narrow_ostride = narrow_points; // 输出每个 FFT 的数据步幅
int PC_IFFT_narrow_odist = 1; //
int PC_IFFT_narrow_batch = narrow_points;   // narrow_pulse_pc_fft_points次FFT


int PC_IFFT_wide_rank = 1;   //  维
int PC_IFFT_wide_n[1] = { prf_total_pulse };    // 每一维变换数目
int PC_IFFT_wide_inembed[2] = { narrow_pulse_pc_fft_points, prf_total_pulse }; // 输入数据的步幅
int PC_IFFT_wide_istride = narrow_pulse_pc_fft_points; // 输入每个 FFT 的数据步幅
int PC_IFFT_wide_idist = 1; // 不同 FFT 之间的距离
int PC_IFFT_wide_onembed[2] = { prf_total_pulse, narrow_points }; // 输出数据的步幅
int PC_IFFT_wide_ostride = narrow_points; // 输出每个 FFT 的数据步幅
int PC_IFFT_wide_odist = 1; //
int PC_IFFT_wide_batch = narrow_points;   // narrow_pulse_pc_fft_points次FFT

cufftHandle PC_IFFT_narrow;
cufftHandle PC_IFFT_wide;

checkCufftError(cufftPlanMany(&PC_IFFT_narrow, PC_IFFT_narrow_rank, PC_IFFT_narrow_n, PC_IFFT_narrow_inembed, PC_IFFT_narrow_istride, PC_IFFT_narrow_idist, PC_IFFT_narrow_onembed, PC_IFFT_narrow_ostride, PC_IFFT_narrow_odist, CUFFT_C2C, PC_IFFT_narrow_batch));
checkCufftError(cufftPlanMany(&PC_IFFT_wide, PC_IFFT_wide_rank, PC_IFFT_wide_n, PC_IFFT_wide_inembed, PC_IFFT_wide_istride, PC_IFFT_wide_idist, PC_IFFT_wide_onembed, PC_IFFT_wide_ostride, PC_IFFT_wide_odist, CUFFT_C2C, PC_IFFT_wide_batch));
---------------------------------------------------------------------
6.参数加载器报错
例如 range_protect_units = INT_PARAMS("Range_protect_units");

7. DDC数据结构怎么导入到GPU
#pragma pack(1)
typedef struct DDC_DATA{
    prf_data_t prf[MAX_PRF_PULSE];
}ddc_data_t;
#pragma pack()

#pragma pack(1)
//原始数据-DDC数据-单个prf数据
typedef struct PRF_DATA{
    pulse_data_t narrow;
    pulse_data_t wide;
}prf_data_t;
#pragma pack()

typedef struct PULSE_DATA{
    complex<float>* chn[36];
}pulse_data_t;
#pragma pack().

8.原始数据，不同脉冲宽窄脉冲之间相隔宽窄脉冲的数据长度：chn_total *（narrow_points+wide_points）
同一个脉冲的宽脉冲不同通道之间相差wide_points，同一个脉冲的窄脉冲不同通道之间相差narrow_points
