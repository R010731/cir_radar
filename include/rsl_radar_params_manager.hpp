/*** 
 * @Author       : lizhengwei waiwaylee@foxmail.com
 * @Date         : 2022-10-19 10:49:15
 * @LastEditors  : lizhengwei
 * @FilePath     : /radar_signal_process/include/rsl_radar_params_manager.hpp
 * @Description  : 
 */
#pragma once 

#ifndef __RSL_RADAR_PARAMS_MANAGER_HPP
#define __RSL_RADAR_PARAMS_MANAGER_HPP

#include <iostream>
#include <string>
#include <map>
#include <unistd.h>
#include <mutex>
#include "simpleIni/SimpleIni.h"
#include "debug_tool/debug_tools.hpp"

/**
 * @brief     从参数管理器中加载int类型的参数，x为参数名字符串
 */
#define INT_PARAMS(x)      		(atoi(radar_signal_process::rsl_radar_params_manager::getInstance().params_map.find((x))->second.c_str()))
/**
 * @brief     从参数管理器中加载float类型的参数，x为参数名字符串
 */
#define FLOAT_PARAMS(x)      	((float)atof(radar_signal_process::rsl_radar_params_manager::getInstance().params_map.find((x))->second.c_str()))
/**
 * @brief     从参数管理器中加载double类型的参数，x为参数名字符串
 */
#define DOUBLE_PARAMS(x)      	(atof(radar_signal_process::rsl_radar_params_manager::getInstance().params_map.find((x))->second.c_str()))
/**
 * @brief     从参数管理器中加载string类型的参数，x为参数名字符串
 */
#define STRING_PARAMS(x)      	(radar_signal_process::rsl_radar_params_manager::getInstance().params_map.find((x))->second)

//常数
//光速
#define LIGHT_SPEED							(300000000)
//圆周率
#define PI									(M_PI)

//杂波抑制使能
#define CLUTTER_SUPPRESSION_ENABLE		0

//DBF远区不加权
#define DBF_FAR_ZONE_NO_WEIGHTING		0

//方位测角系数三次多项式拟合
#define AZI_COEFF_ORDER_3				1

//俯仰波束1分配线程数
#define THREADS_NUM_FOR_BEAM_0          (8)

//================调试参数===============
// 实时从光口加载数据
#define REAL_TIME_PROCESS				0
// 开启全流程总调试
#define ALL_DEBUG						1
// 开启DBF调试
#define DBF_DEBUG						0
// 开启脉压调试
#define PC_DEBUG						0
// 开启MTD调试	
#define MTD_DEBUG						0
// 开启CFAR调试
#define CFAR_DEBUG						0
// 开启杂波图/杂波抑制调试
#define CLUTTER_DEBUG					0
// 开启点迹过滤器
#define CLUTTER_MAP_DEBUG				0
// 开启测角调试
#define ANGLE_DEBUG						0
// 开启俯仰测角调试
#define PITCH_ANGLE_DEBUG				0
// 开启点迹过滤器
#define TARGET_FLITER_DEBUG				0
// 开启目标识别调试
#define TARGET_IDENTIFY_DEBUG			0

namespace radar_signal_process{

	/**
	 * @brief     雷达相位数据结构体
	 */
	typedef struct PHASE_VALUE{
		float * phase_data = nullptr;
		int total_freq_point;
		int total_board;
		int total_channel;

		float * operator()(int freq_point, int board, int channel){
			if(phase_data == nullptr)
				return nullptr;
			else
				return &phase_data[(freq_point * total_board + board) * total_channel + channel];
		}

	}phase_t;

	/**
	 * @brief     雷达接收幅相配平数据结构体
	 */
	typedef struct BALANCING_VALUE{
		float * balancing_value = nullptr;
		int total_freq_point;
		int total_board;

		float * operator()(int freq_point, int board){
			if(balancing_value == nullptr)
				return nullptr;
			else
				return &balancing_value[board * total_freq_point + freq_point];
		}
	}balancing_value_t;

	/**
	 * @brief     雷达波束指向修正值结构体（频点， 波位号）
	 */
	typedef struct BEAM_CALIBRATION_VALUE{
		float * calibration_value = nullptr;
		int total_freq_point;
		int total_beams;

		float * operator()(int freq_point, int beam_num){
			if(calibration_value == nullptr)
				return nullptr;
			else
				return &calibration_value[beam_num * total_freq_point + freq_point];
		}
	}beam_calibration_value_t;

	/**
	 * @brief     雷达方位测角系数结构体
	 */
#if AZI_COEFF_ORDER_3 == 0
	typedef struct AZIMUTH_ANGLE_COEFF{
		float * coeff = nullptr;
		int total_beams;
		int freq_point_num;

		float * operator()(int beam_num, int freq_point){
			if(coeff == nullptr)
				return nullptr;
			else
				return &coeff[beam_num * freq_point_num + freq_point];
		}
	}azi_angle_coeff_t;
#else
	typedef struct AZIMUTH_ANGLE_COEFF{
		float * coeff = nullptr;
		int total_beams;
		int freq_point_num;

		float * operator()(int beam_num, int freq_point){
			if(coeff == nullptr)
				return nullptr;
			else
				return &coeff[beam_num * freq_point_num * 3 + freq_point * 3];
		}
	}azi_angle_coeff_t;
#endif
	/**
	 * @brief     雷达俯仰测角系数结构体
	 */
	typedef struct PITCH_ANGLE_COEFF{
		float * coeff = nullptr;
		int total_pitch_beams;
		int angle_range;
		int max_order;
		int min_angle;

		float * operator()(int pitch_beam_num, int pitch_beam_angle, int order){
			if(coeff == nullptr)
				return nullptr;
			else
				return &coeff[pitch_beam_num * angle_range * max_order + (pitch_beam_angle - min_angle) * max_order + order];
		}
	}pit_angle_coeff_t;

	/**
	 * @brief     雷达STC曲线补偿值
	 */
	typedef struct STC_COMPENSATE_VALUE{
		std::vector<float *> narrow_coeff;
		std::vector<float *> wide_coeff;
		int pitch_total_beam_num;

		float * operator()(int pit_beam_num, bool is_narrow_pulse, int point){
			if(narrow_coeff.size() <= pit_beam_num){
				return nullptr;
			}else{
				if(is_narrow_pulse){
					return &(narrow_coeff[pit_beam_num][point]);
				}else{
					return &(wide_coeff[pit_beam_num][point]);
				}
			}
			
		}
	}stc_compensate_value_t;

	// typedef struct STC_COMPENSATE_VALUE{
	// 	float * coeff = nullptr;
	// 	int max_length;
	// 	int azimuth_total_beam_num;
	// 	int pitch_total_beam_num;

	// 	float * operator()(int azi_beam_num, int pit_beam_num, int point){
	// 		if(coeff == nullptr)
	// 			return nullptr;
	// 		else{
	// 			return &coeff[azi_beam_num * pitch_total_beam_num * max_length + pit_beam_num * max_length + point];
	// 		}
	// 	}
	// }stc_compensate_value_t;

	/**
	 * @brief     天线开关控制结构体
	 */
	typedef struct ANTENNA_SWITCH{
		uint16_t* ant_switch;

		int channel_num_of_antenna_board;

		uint16_t * operator()(int board_index, int chn_index){
			if(ant_switch == nullptr)
				return nullptr;
			else
				return &ant_switch[board_index * channel_num_of_antenna_board + chn_index];
		}
	}antenna_switch_t;

	/**
	 * @brief     软件化实时数据存储使能结构体
	 */
	typedef struct RT_DATA_SAVE_CONFIG{
		int origin_data_save_enable = 0;
		int origin_data_save_size_GB = 0;
		int origin_data_circular_storage = 0;

		int process_data_save_enable = 0;
		int DBF_data_save_enable = 0;
		int PC_data_save_enable = 0;
		int MTD_data_save_enable = 0;

		int target_data_save_enable = 0;
		int target_filter_enable = 0;

		float min_range = 0.0f;
		float max_range = 0.0f;
		float min_velocity = 0.0f;
		float max_velocity = 0.0f;
		float min_azimuth = 0.0f;
		float max_azimuth = 0.0f;
		float min_pitch = 0.0f;
		float max_pitch = 0.0f;
	}rt_data_save_config_t;

	class rsl_radar_params_manager
	{
	private:
		/* data */
		CSimpleIniA ini;

		int params_id;

		string file_path = "config/radar_params.ini";

		rsl_radar_params_manager(/* args */);
		void load_radar_params();
		void load_calibration_data();

		void load_recv_phase_data();
		void load_trans_phase_data();
		void load_recv_compensate_data();
		void load_recv_beam_calibrate_data();
		void load_trans_beam_calibrate_data();
		void load_azimuth_angle_coeff();
		void load_pitch_angle_coeff();
		void load_STC_compensate_data();

	public:
		std::map<std::string,std::string> params_map;
		phase_t recv_phase;
		phase_t trans_phase;
		balancing_value_t recv_amp_balancing_value;
		balancing_value_t recv_phase_balancing_value;
		beam_calibration_value_t recv_dbf_calibration_value;
		beam_calibration_value_t recv_target_calibration_value;
		beam_calibration_value_t trans_beam_calibration_value;
		azi_angle_coeff_t azimuth_angle_coeff;
		pit_angle_coeff_t pitch_angle_coeff;
		stc_compensate_value_t stc_compensate_value;

		antenna_switch_t t_switch;
		antenna_switch_t r_switch;

		rt_data_save_config_t rt_data_save_config;

		static std::mutex fftw_plan_mtx; 
		
		static rsl_radar_params_manager& getInstance();

		rsl_radar_params_manager(const rsl_radar_params_manager&)=delete;
    	rsl_radar_params_manager& operator=(const rsl_radar_params_manager&)=delete;
		
		~rsl_radar_params_manager();

		//参数更新
		int get_params_id();

		void update_mode_select(int16_t * cmd);
		void update_work_config(int16_t * cmd);
		void update_trans_switch(int16_t * cmd);
		void update_recv_switch(int16_t * cmd);
		void update_beam_pitch(int16_t * cmd);
		void update_threshold_control(int16_t * cmd);

		void update_terrain_detection(int16_t * cmd);
		void update_freq_point(int16_t * cmd);
		void update_frequency_agility(int16_t * cmd);
		void update_tas_beam(int16_t * cmd);
		void update_signal_process(int16_t * cmd);

		void update_data_save_config(int16_t * cmd);
	};
	
}



#endif