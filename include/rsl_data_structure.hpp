#pragma once

#ifndef __RSL_DATA_STRUCTURE_HPP
#define __RSL_DATA_STRUCTURE_HPP

#include <cstdint> 
#include <iostream>
#include <string>
#include <fstream>
#include <complex>
#include <map>
#include <memory> 

#include <stdio.h>
#include <string.h>

typedef unsigned char           uchar;

#define STATUS_INFO_VALID       1
#define MAX_PRF_PULSE           32768
#define MAX_NARROW_POINTS       512
#define MAX_WIDE_POINTS         2048

#define CMD_PARAMS_LEN          57152 * 4
#define STATUS_INFO_LEN         320 * 12 * 4
#define SELF_TEST_LEN           48 * 256 * 2 * 4
#define DDC_DATA_LEN(prfn,narrow,wide)             (prfn * 72 * (narrow + wide)) * 4   
#define JAMMING_DATA_LEN(narrow,wide)              (72 * (narrow + wide)) * 4
#define NOISE_DATA_LEN          24576 * 4

#if STATUS_INFO_VALID == 0
#define CPI_DATA_LEN(prfn,narrow,wide)              (CMD_PARAMS_LEN + SELF_TEST_LEN + DDC_DATA_LEN(prfn,narrow,wide) + JAMMING_DATA_LEN(narrow,wide) + NOISE_DATA_LEN)
#else
#define CPI_DATA_LEN(prfn,narrow,wide)              (CMD_PARAMS_LEN + STATUS_INFO_LEN + SELF_TEST_LEN + DDC_DATA_LEN(prfn,narrow,wide) + JAMMING_DATA_LEN(narrow,wide) + NOISE_DATA_LEN)
#endif

using namespace::std;

namespace radar_signal_process{

#define TRANS_CALIBRATION_MODE          0x00000001
#define RECV_CALIBRATION_MODE           0x00000002
#define NORMAL_WORK_MODE                0x00000003
#define SELF_TEST_MODE                  0x00000004

#define EXTERNAL_SOURCE_WAVEFORM        (0x0)
#define INTERNAL_SOURCE_WAVEFORM        (0x1)

#define BAND_WIDTH_2_5MHZ               (0x0)
#define BAND_WIDTH_5MHZ                 (0x1)

#define FS_NORMAL_MODE                  (0x0)
#define FS_TRANS_CALIBRATION_MODE       (0x1)
#define FS_RECV_CALIBRATION_MODE        (0x2)

#define SMA_CPI                         (0x0)
#define SMA_PRF                         (0x1)
#define SMA_ANTENNA_T_PULSE             (0x2)
#define SMA_ANTENNA_R_PULSE             (0x3)

typedef enum TARGET_TYPE{
    unidentified = 0,               //未识别目标
    ground_unidentified = 10,       //地面未识别目标
    ground_human = 11,              //地面人员
    ground_vehicle = 12,            //地面车辆
    air_unidentified = 20,          //空中未识别目标
    air_small_UAV = 21,             //小型无人机（旋翼机）
    air_large_UAV = 22,             //大型无人机（固定翼）
    air_helicopter = 23,            //直升机
    air_airliner = 24,              //民航
    air_bird = 25,                  //鸟类
    sea_surface_unidentified = 30,  //海上未识别目标
    sea_surface_small_ship = 31,    //小型船舶
    sea_surface_mid_ship = 32,      //中型船舶
    sea_surface_large_ship = 33,    //大型船舶
    sea_surface_drift = 34          //漂浮物
}target_type_t;

class cpis_ram_t{
    private:
    public:
        char * ram_ptr;
        size_t size;

        cpis_ram_t();
        cpis_ram_t(char* ptr, size_t s);
        ~cpis_ram_t();
};

#pragma pack(1)
//原始数据-命令参数包-工作模式
typedef struct WORK_MODE{
    uint32_t mode;
    uint16_t self_test_online[6];
    uint8_t  azimuth_beam_num[3];
    uint8_t  pitch_beam_num;
    uint32_t flag_last_cpi_in_beams;
    uint32_t params_id;
    uint16_t tas_range;
    uint16_t tas_azimuth;
    uint16_t tas_pitch;
    int16_t tas_target_index;
    uint32_t cpis_data_length;
    uint32_t beam_type;

    void set_work_mode(uint32_t work_mode){
        mode = (mode & ~0x0000000f) | work_mode;
    }

    void set_source(uint32_t source){
        mode = (mode & ~(0x01<<4)) | (source << 4);
    }

    void set_band_width(uint32_t band_width){
        mode = (mode & ~(0x01<<5)) | (band_width << 5);
    }

    void set_freq_point(uint32_t freq_point){
        mode = (mode & ~(0x3f<<6)) | (freq_point << 6);
    }

    void set_stc(uint32_t stc){
        mode = (mode & ~(0x7<<12)) | (stc<<12);
    }

    void set_frequency_synthesizer(uint32_t fs_mode){
        mode = (mode & ~(0x7<<15)) | (fs_mode<<15);
    }

    void set_sma_output(uint32_t sma1, uint32_t sma2){
        mode = (mode & ~(0xf<<18)) | (sma1<<18);
        mode = (mode & ~(0xf<<22)) | (sma2<<22);
    }
}work_mode_t;
#pragma pack()

#pragma pack(1)
//原始数据-命令参数包-脉冲参数
typedef struct PULSE_PARAMS{
    uint32_t cpi_cnt;
    uint32_t cpi_period;
    uint16_t prf_period;
    uint16_t prf_total_pulse;
    uint16_t wide_pulse_point;
    uint16_t narrow_pulse_point;
    uint16_t wide_pulse_valid_point;
    uint16_t narrow_pulse_valid_point;
    uint16_t AD_begin_point;
    uint16_t AD_valid_point;
    uint32_t reserved[7];
}pulse_params_t;
#pragma pack()

#pragma pack(1)
//原始数据-命令参数包-脉压参数
typedef struct PC_PARAMS{
    uint16_t wide_pulse_width;
    uint16_t narrow_pulse_width;
    uint16_t wide_pulse_pc_points;
    uint16_t narrow_pulse_pc_points;
    uint16_t wide_pulse_halfpc_points;
    uint16_t narrow_pulse_halfpc_points;
    uint16_t wide_pulse_pc_valid_points;
    uint16_t narrow_pulse_pc_valid_points;
    uint16_t wide_pulse_pc_fft_points;
    uint16_t narrow_pulse_pc_fft_points;
    int16_t wide_pulse_pc_coeff[4096];
    int16_t narrow_pulse_pc_coeff[1024];
}pc_params_t;
#pragma pack()

#pragma pack(1)
//原始数据-命令参数包
typedef struct CMD_PARAMS{
    //帧头
    uint32_t head;
    
    //数据长度
    uint32_t length;

    //工作模式
    work_mode_t work_mode;

    //脉冲参数
    pulse_params_t pulse_params;

    //DBF组合
    uint8_t DBF_group[3][20];

    //DBF系数-3个波束-2路（和/差）-18通道（18个通道，虚部加实部，共36个）
    int16_t DBF_coeff[3][2][36];

    //脉压参数
    pc_params_t PC_params;

    //天线立板控制码数据长度
    uint32_t ant_ctrl_code_length;
    uint16_t ant_ctrl_code[600];
    
    //内源信号
    uint32_t internal_source_signal[4092];

    //发射信号波形
    uint32_t trans_signal_len;
    int16_t trans_signal[100000];

    //密文
    uint32_t encryption[34];

    //预留
    uint32_t reserved[9];

    //帧尾
    uint32_t tail;
}cmd_params_t;
#pragma pack()

#pragma pack(1)
//原始数据-状态信息包
typedef struct STATUS_INFO{
    //GNSS
    char GNSS_data[1024];
    uint32_t pps_cnt;
    uint32_t reserved_0;

    //频综
    uint32_t Frequency_synthesizer_error_bit;
    uint32_t reserved_1;

    //天线
    uint16_t head;
    uint16_t attenna_board_num;
    uint16_t crc_check[3];
    uint16_t package_index;
    uint32_t attenna_temperature;
    uint32_t attenna_humidity;
    uint16_t voltage_5V;
    uint16_t current_5V;
    uint16_t voltage_12V;
    uint16_t current_12V;
    uint16_t voltage_test;
    uint16_t current_test;
    uint64_t attenna_version;

    //母板
    uint32_t mother_board_error_bit;
    uint16_t FPGA_temperature;
    uint16_t voltage_VCCINT;
    uint16_t voltage_VCCAUX;
    uint16_t voltage_VCCBRAM;
    char angle_sensor_data[12];

    //密文
    uint32_t encrypt_valid;
    uint32_t reserved_2;
    uint32_t interface_encrypt[8];
    uint32_t DSP_encrypt[8];
    uint32_t spb_FPGA_encrypt[8];
    uint32_t mb_FPGA_encrypt[8];
    uint32_t mb_FPGA_version[8];

    //保留
    uint32_t reserved_3;
    uint32_t reserved_4;
}status_info_t;
#pragma pack()

#pragma pack(1)
//原始数据-自检数据包
typedef struct SELF_TEST{
    uint32_t* data;
}self_test_t;
#pragma pack()

#pragma pack(1)
//原始数据-DDC数据-单个宽（窄）脉冲数据
typedef struct PULSE_DATA{
    complex<float>* chn[36];
}pulse_data_t;
#pragma pack()

#pragma pack(1)
//原始数据-DDC数据-单个prf数据
typedef struct PRF_DATA{
    pulse_data_t narrow;
    pulse_data_t wide;
}prf_data_t;
#pragma pack()

#pragma pack(1)
//原始数据-DDC数据
typedef struct DDC_DATA{
    prf_data_t prf[MAX_PRF_PULSE];
}ddc_data_t;
#pragma pack()

#pragma pack(1)
//原始数据-干扰数据
typedef struct JAMMING_DATA{
    pulse_data_t narrow;
    pulse_data_t wide;
}jamming_data_t;
#pragma pack()

#pragma pack(1)
//原始数据-噪声数据
typedef struct NOISE_DATA{
    uint32_t* data;
}noise_data_t;
#pragma pack()

#pragma pack(1)
//dbf输出数据格式,单个波束
typedef struct DBF_OUTPUT{
    complex<float>* narrow_sum[MAX_PRF_PULSE] = {nullptr};
    complex<float>* narrow_diff[MAX_PRF_PULSE] = {nullptr};
    complex<float>* wide_sum[MAX_PRF_PULSE] = {nullptr};
    complex<float>* wide_diff[MAX_PRF_PULSE] = {nullptr};

    //dbf输出数据初始化，分配RAM空间
    void init(int prf_total_pulse, int narrow_points, int wide_points){
        //初始化DBF输出ram空间, ram需要对齐，满足后续FFT处理的要求
        complex<float> * ram_ptr = nullptr;
        // int posix_memalign(void **memptr, size_t alignment, size_t size);
        // memptr：这是一个指向 void* 类型的指针的指针，调用成功后，这个指针将指向分配的内存块的地址。
        // alignment：指定内存块的对齐要求。它必须是 2 的幂，并且通常是比 sizeof(void*) 更大的值，例如 16 字节、32 字节等。
        // size：要分配的内存块的大小（单位是字节）。
        int retval = posix_memalign((void **)&ram_ptr, 16, 8 * 2 * prf_total_pulse * (narrow_points + wide_points));
        for (int prf_index = 0; prf_index < prf_total_pulse; prf_index++)
        {
            narrow_sum[prf_index]    = ram_ptr + prf_index * narrow_points;
            narrow_diff[prf_index]   = ram_ptr + (prf_total_pulse + prf_index) * narrow_points;
            wide_sum[prf_index]      = ram_ptr + 2 * prf_total_pulse * narrow_points + prf_index * wide_points;
            wide_diff[prf_index]     = ram_ptr + 2 * prf_total_pulse * narrow_points + (prf_total_pulse + prf_index) * wide_points;
        }
    }

    //销毁dbf输出数据，释放RAM空间
    
    void dispose(){
        if(narrow_sum[0] != nullptr)
            free(narrow_sum[0]);
    }
}dbf_t;
#pragma pack()

#pragma pack(1)
//pc输出数据格式，单个波束，按转置后的数据排列
typedef struct PC_OUTPUT{
    complex<float>* narrow_sum[MAX_NARROW_POINTS] = {nullptr};
    complex<float>* narrow_diff[MAX_NARROW_POINTS] = {nullptr};
    complex<float>* wide_sum[MAX_WIDE_POINTS] = {nullptr};
    complex<float>* wide_diff[MAX_WIDE_POINTS] = {nullptr};

    /**
     * @brief     初始化DBF输出ram空间
     * @param     [in] prf_total_pulse 
     * @param     [in] narrow_points 
     * @param     [in] wide_points 
     */
    void init(int prf_total_pulse, int narrow_points, int wide_points){
        //初始化脉压输出ram空间, ram需要对齐，满足后续FFT处理的要求
        complex<float> * ram_ptr = nullptr;
        int retval = posix_memalign((void **)&ram_ptr, 16, 8 * 2 * prf_total_pulse * (narrow_points + wide_points));

        for (int narrow_index = 0; narrow_index < narrow_points; narrow_index++)
        {
            narrow_sum[narrow_index]    = ram_ptr + narrow_index * prf_total_pulse;
            narrow_diff[narrow_index]   = ram_ptr + (narrow_index + narrow_points) * prf_total_pulse;
        }

        for (int wide_index = 0; wide_index < wide_points; wide_index++)
        {
            wide_sum[wide_index]    = ram_ptr + 2 * narrow_points * prf_total_pulse + wide_index * prf_total_pulse;
            wide_diff[wide_index]   = ram_ptr + 2 * narrow_points * prf_total_pulse + (wide_index + wide_points) * prf_total_pulse;
        }
    }

    //销毁dbf输出数据，释放RAM空间
    void dispose(){
        if(narrow_sum[0] != nullptr)
            free(narrow_sum[0]);
    }
}pc_t;
#pragma pack()

//mtd输出数据格式，单个波束，数据排列与脉压数据相同
typedef pc_t mtd_t;

#pragma pack(1)
//mtd输出数据格式，单个波束，取模后输出，数据排列与脉压数据相同
typedef struct MTD_OUTPUT{
    //窄脉冲和波束
    float* narrow_sum[MAX_NARROW_POINTS] = {nullptr};
    //窄脉冲差波束
    float* narrow_diff[MAX_NARROW_POINTS]= {nullptr};
    //宽脉冲和波束
    float* wide_sum[MAX_WIDE_POINTS]= {nullptr};
    //宽脉冲差波束
    float* wide_diff[MAX_WIDE_POINTS]= {nullptr};

    //脉冲数
    int prf_total_pulse;
    //窄脉冲点数
    int narrow_points;
    //宽脉冲点数
    int wide_points;

    /**
     * @brief     mtd输出数据初始化，分配RAM空间
     * @param     [in] prf_total_pulse_tmp 脉冲总数
     * @param     [in] narrow_points_tmp 窄脉冲点数
     * @param     [in] wide_points_tmp 宽脉冲点数
     */
    void init(int prf_total_pulse_tmp, int narrow_points_tmp, int wide_points_tmp){
        prf_total_pulse = prf_total_pulse_tmp;
        narrow_points = narrow_points_tmp;
        wide_points = wide_points_tmp;

        //初始化DBF输出ram空间, ram需要对齐，满足后续FFT处理的要求
        float * ram_ptr = nullptr;
        int retval = posix_memalign((void **)&ram_ptr, 8, 4 * 2 * prf_total_pulse * (narrow_points + wide_points));

        for (int narrow_index = 0; narrow_index < narrow_points; narrow_index++)
        {
            narrow_sum[narrow_index]    = ram_ptr + narrow_index * prf_total_pulse;
            narrow_diff[narrow_index]   = ram_ptr + (narrow_index + narrow_points) * prf_total_pulse;
        }

        for (int wide_index = 0; wide_index < wide_points; wide_index++)
        {
            wide_sum[wide_index]    = ram_ptr + 2 * narrow_points * prf_total_pulse + wide_index * prf_total_pulse;
            wide_diff[wide_index]   = ram_ptr + 2 * narrow_points * prf_total_pulse + (wide_index + wide_points) * prf_total_pulse;
        }
    }

    //销毁dbf输出数据，释放RAM空间
    void dispose(){
        if(narrow_sum[0] != nullptr)
            free(narrow_sum[0]);
    }

    //重载=
    MTD_OUTPUT operator&=(MTD_OUTPUT& t)
	{
		if (this != &t)
		{
			memcpy(this->narrow_sum[0], t.narrow_sum[0], 4 * 2 * prf_total_pulse * (narrow_points + wide_points));
            return t;
		}
	}
}mtd_abs_t;
#pragma pack()

#pragma pack(1)
//杂波图数据格式，单个波束，数据排列与脉压数据相同
typedef struct CLUTTER_MAP{
    float* narrow_sum[MAX_NARROW_POINTS] = {nullptr};
    float* wide_sum[MAX_WIDE_POINTS]= {nullptr};

    int prf_total_pulse;
    int narrow_points;
    int wide_points;
    bool clutter_map_valid;

    //杂波图数据初始化，分配RAM空间
    void init(int prf_total_pulse_tmp, int narrow_points_tmp, int wide_points_tmp){
        prf_total_pulse = prf_total_pulse_tmp;
        narrow_points = narrow_points_tmp;
        wide_points = wide_points_tmp;
        clutter_map_valid = false;

        //初始化杂波图数据ram空间, ram需要对齐
        float * ram_ptr = nullptr;
        int retval = posix_memalign((void **)&ram_ptr, 8, 4 * prf_total_pulse * (narrow_points + wide_points));

        for (int narrow_index = 0; narrow_index < narrow_points; narrow_index++)
        {
            narrow_sum[narrow_index]    = ram_ptr + narrow_index * prf_total_pulse;
        }

        for (int wide_index = 0; wide_index < wide_points; wide_index++)
        {
            wide_sum[wide_index]    = ram_ptr + narrow_points * prf_total_pulse + wide_index * prf_total_pulse;
        }
    }

    //销毁杂波图数据，释放RAM空间
    void dispose(){
        if(narrow_sum[0] != nullptr)
            free(narrow_sum[0]);
    }
}clutter_map_t;
#pragma pack()

#pragma pack(1)
/**
 * @brief     输出点迹数据格式
 */
typedef struct TARGET{
    unsigned long time_stamp;
    unsigned int r_unit;
    unsigned int v_unit;
    float r_cohesion_val;
    float v_cohesion_val;
    float target_amp;
    float target_diff_amp;
    float noise_amp;
    bool is_narrow_pulse;
    float azimuth;
    float pitch;
    unsigned int azimuth_beam_num;
    unsigned int pitch_beam_num;
    unsigned int r_total_unit;
    unsigned int v_total_unit;
    unsigned int cpi_index;
    unsigned int pri;

    unsigned int flag;

    float rcs;

    complex<float> sum_mtd_val;
    complex<float> diff_mtd_val;

    float target_amp_compensate;

    float snr;

    uint8_t tas_flag;

    unsigned int tas_target_index;

    int target_identification;
    float target_probability;

    uint8_t reserved[46];

    void init_null(void){
        time_stamp = 0;
        r_unit = 0;
        v_unit = 0;
        r_cohesion_val = 0.0f;
        v_cohesion_val = 0.0f;
        target_amp = 0.0f;
        target_diff_amp = 0.0f;
        noise_amp = 0.0f;
        is_narrow_pulse = false;
        azimuth = 0.0f;
        pitch = 0.0f;
        azimuth_beam_num = 0;
        pitch_beam_num = 0;
        r_total_unit = 0;
        v_total_unit = 0;
        cpi_index = 0;
        pri = 0;
        flag = 0;
        rcs = 0.0f;
        sum_mtd_val.real(0);
        sum_mtd_val.imag(0);
        diff_mtd_val.real(0);
        diff_mtd_val.imag(0);

        target_amp_compensate = 0.0f;
        target_identification = 0;
        snr = 0.0f;

        tas_flag = 0x00;
        tas_target_index = 0;

        memset(reserved, 0, sizeof(reserved));
    }

	// 重载 << 运算符
	friend ostream& operator<<(ostream& out, TARGET& target) {
        char str[256];
        sprintf(str,"Target:%d,\tR:(%d) %.2f, \tV:(%d) %.2f, \tAzi:(%d) %.2f,\tPit:(%d) %.2f, \tAmp: %.2e",target.is_narrow_pulse, target.r_unit, target.r_cohesion_val, target.v_unit, target.v_cohesion_val,target.azimuth_beam_num, target.azimuth, target.pitch_beam_num, target.pitch, target.target_amp);
        out<<str;
		return out;
	}

}target_t;
#pragma pack()

#pragma pack(1)
typedef struct GNSS_MSG
{
    char status;
    uint32_t date;
    float time;
    double latitude;
    char n_s;
    double longitude;
    char e_w;
    uint32_t nosv;
    double altitude;
    uint32_t pps;
    float azimuth;
    char reserved[17];

    void parse(char * msg, uint32_t pps){

        //msg_tmp解析
        char *msg_tmp;
        while ((msg_tmp = (char *)strsep(&msg, "\n")) != NULL){
            if(msg_tmp[3] == 'G' && msg_tmp[4] == 'G' && msg_tmp[5] == 'A'){
                uint16_t checkSum=0x00;
                uint16_t checkByte=0;
                for(uint16_t i=0;i<strlen(msg_tmp);i++){
                    if(msg_tmp[i]==0x2A){
                        checkByte=i+1;
                        break;
                    }
                    if(msg_tmp[i]!=0x24){
                        checkSum^=msg_tmp[i];
                    }
                }
                uint8_t tmp1=(checkSum>>4)<10?(checkSum>>4)+0x30:(checkSum>>4)+0x37;
                uint8_t tmp2=(checkSum&0x0f)<10?((checkSum&0x0f)+0x30):((checkSum&0x0f)+0x37);
                if(tmp1==msg_tmp[checkByte] && tmp2==msg_tmp[checkByte+1]){
                    char *token;
                    char *ptr;
                    ptr = (char *)msg_tmp;
                    uint8_t index = 0;
                    int tmp = 0;
                    while ((token = (char *)strsep(&ptr, ",")) != NULL){
                        switch(index){
                            case 1:
                                tmp = (int)(100 * atof(token));
                                time = 0;
                                time += (tmp%10000)/100.0f;
                                tmp /= 10000;
                                time += (tmp%100) * 60.0f;
                                tmp /= 100;
                                time += (tmp%100) * 3600.0f;
                                time += (float)pps / 10e6f;
                                break;
                            case 2:
                                latitude = atof(token);break;
                            case 3:
                                n_s = *token;break;
                            case 4:
                                longitude = atof(token);break;
                            case 5:
                                e_w = *token;break;
                            case 6:
                                status = *token;break;
                            case 7:
                                nosv = atoi(token);break;
                            case 9:
                                altitude = atof(token);
                            default:
                                break;
                        }
                        index++;
                    }
                }
            }else if(msg_tmp[3] == 'R' && msg_tmp[4] == 'M' && msg_tmp[5] == 'C'){
                uint16_t checkSum=0x00;
                uint16_t checkByte=0;
                for(uint16_t i=0;i<strlen(msg_tmp);i++){
                    if(msg_tmp[i]==0x2A){
                        checkByte=i+1;
                        break;
                    }
                    if(msg_tmp[i]!=0x24){
                        checkSum^=msg_tmp[i];
                    }
                }
                uint8_t tmp1=(checkSum>>4)<10?(checkSum>>4)+0x30:(checkSum>>4)+0x37;
                uint8_t tmp2=(checkSum&0x0f)<10?((checkSum&0x0f)+0x30):((checkSum&0x0f)+0x37);
                if(tmp1==msg_tmp[checkByte] && tmp2==msg_tmp[checkByte+1]){
                    char *token;
                    char *ptr;
                    ptr = (char *)msg_tmp;
                    uint8_t index = 0;
                    int tmp = 0;
                    while ((token = (char *)strsep(&ptr, ",")) != NULL){
                        switch(index){
                            case 1:
                                tmp = (int)(100 * atof(token));
                                time = 0;
                                time += (tmp%10000)/100.0f;
                                tmp /= 10000;
                                time += (tmp%100) * 60.0f;
                                tmp /= 100;
                                time += (tmp%100) * 3600.0f;
                                time += (float)pps / 10e6f;
                                break;
                            case 2:
                                status = *token;break;
                            case 3:
                                latitude = atof(token);break;
                            case 4:
                                n_s = *token;break;
                            case 5:
                                longitude = atof(token);break;
                            case 6:
                                e_w = *token;break;
                            case 9:
                                date = (uint32_t)atoi(token);break;
                            default:
                                break;
                        }
                        
                        index++;
                    }
                }
            }else if(msg_tmp[3] == 'H' && msg_tmp[4] == 'D' && msg_tmp[5] == 'T'){
                uint16_t checkSum=0x00;
                uint16_t checkByte=0;
                for(uint16_t i=0;i<strlen(msg_tmp);i++){
                    if(msg_tmp[i]==0x2A){
                        checkByte=i+1;
                        break;
                    }
                    if(msg_tmp[i]!=0x24){
                        checkSum^=msg_tmp[i];
                    }
                }
                uint8_t tmp1=(checkSum>>4)<10?(checkSum>>4)+0x30:(checkSum>>4)+0x37;
                uint8_t tmp2=(checkSum&0x0f)<10?((checkSum&0x0f)+0x30):((checkSum&0x0f)+0x37);
                if(tmp1==msg_tmp[checkByte] && tmp2==msg_tmp[checkByte+1]){
                    char *token;
                    char *ptr;
                    ptr = (char *)msg_tmp;
                    uint8_t index = 0;
                    while ((token = (char *)strsep(&ptr, ",")) != NULL){
                        switch(index){
                            case 1:
                                azimuth = atof(token);break;
                            default:
                                break;
                        }
                        index++;
                    }
                }
            }
        }
    }
}gnss_msg_t;
#pragma pack()

//定义map类型用于存放距离维CFAR时过门限的点，
//其中：int数据高16位保存速度门号，低16位保存距离门号
//利用map的排序特性，将过门限的点按速度门优先，距离门其次的升序排序
//方便后续的速度维凝聚
/**
 * @brief     用于保存目标数据，其中：
 *            key类型为int，可以根据需要保存对应的数据，利用map的排序特性对target排序
 *            val类型为target_t，包含目标的所有信息的结构体，参考 @link target_t target_t @endlink
 */
typedef map<int,target_t> target_map_t;

//原始数据-完整CPI数据包
class cpi_data_t
    {
    private:
    public:
        uint32_t* cpi_data_ptr;
        shared_ptr<cpis_ram_t> cpis_ram;

        cpi_data_t();
        cpi_data_t(const cpi_data_t& data);
        cpi_data_t(uint32_t * data_ptr);
        ~cpi_data_t();

        void dispose();

        cmd_params_t* cmd_params;
        status_info_t* status_info;
        self_test_t* self_test;
        ddc_data_t* ddc_data;
        jamming_data_t* jamming_data;
        noise_data_t* noise_data;

        gnss_msg_t gnss_msg;
    };
}

#endif


