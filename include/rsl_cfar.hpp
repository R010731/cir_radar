#pragma once 

#ifndef __RSL_CFAR_HPP
#define __RSL_CFAR_HPP

#include <iostream>
#include <complex>
#include <map>
#include <queue>
#include <algorithm>
#include <numeric>
#include <fftw3.h> 
#include <math.h>
#include <Eigen/Dense>

#include "rsl_data_structure.hpp"
#include "rsl_radar_params_manager.hpp"
#include "debug_tools.hpp"
#include "rsl_win_func_manager.hpp"

#define CFAR_THREAD_NUM           THREADS_NUM_FOR_BEAM_0

using namespace::std;
using namespace::rsl_debug_tools;
using namespace::Eigen;

namespace radar_signal_process
{
	class rsl_cfar
	{
	private:
		/* data */
		float time_cfar = 0;

		int params_id;

		int azi_sector_num;

		//保护单元，平均单元
		int range_protect_units;
		int range_average_units;
		int velocity_protect_units;
		int velocity_average_units;

		float r_threshold;
		float v_threshold;
		float clutter_threshold;

		int velocity_max_snr_target_cnt;

		int clutter_suppression_enable;

		int zero_doppler_channel_number;
		int noise_average_units;

		float range_offset_value;

		float AD_clock;
		float Carrier_frequency;
		float Freq_point_interval;

		int wide_narrow_cohesion_enable;

		void params_update(int curr_params_id);

		void task_cfar(cpi_data_t& cpi_data, mtd_abs_t& mtd_abs_data, clutter_map_t& clutter_map_data, bool is_narrow_pulse, target_map_t& target_output);

		void task_cohesion(cpi_data_t& cpi_data, target_map_t& target_narrow, target_map_t& target_wide, int azi_beam_num, target_map_t& target_output);

	public:
		rsl_cfar(/* args */);
		~rsl_cfar();

		void cfar(cpi_data_t& cpi_data, mtd_abs_t &mtd_data, clutter_map_t& clutter_map_data, int azi_beam_num, target_map_t &target_output);

	};
	
	rsl_cfar::rsl_cfar(/* args */)
	{
		params_id = -1;
		params_update(0);
	}
	
	rsl_cfar::~rsl_cfar()
	{
	}

	inline void rsl_cfar::params_update(int curr_params_id)
	{
		if(params_id != curr_params_id){
			// range_protect_units = INT_PARAMS("Range_protect_units");
			// range_average_units = INT_PARAMS("Range_average_units");
			// velocity_protect_units = INT_PARAMS("Velocity_protect_units");
			// velocity_average_units = INT_PARAMS("Velocity_average_units");

			// r_threshold = powf(10, FLOAT_PARAMS("Range_threshold")/20);
			// v_threshold = powf(10, FLOAT_PARAMS("Velocity_threshold")/20);
			// clutter_threshold = powf(10, FLOAT_PARAMS("Clutter_map_threshold")/20);

			// velocity_max_snr_target_cnt = INT_PARAMS("Velocity_max_snr_target_cnt");

			// clutter_suppression_enable = INT_PARAMS("Clutter_suppression_enable");

			// zero_doppler_channel_number = INT_PARAMS("Zero_doppler_channel_number");
			// noise_average_units = INT_PARAMS("Noise_average_units");

			// range_offset_value = FLOAT_PARAMS("Range_offset_value");

			// AD_clock = FLOAT_PARAMS("AD_clock");
			// Carrier_frequency = FLOAT_PARAMS("Carrier_frequency");
			// Freq_point_interval = FLOAT_PARAMS("Freq_point_interval");

			// wide_narrow_cohesion_enable = INT_PARAMS("Wide_narrow_cohesion_enable");

			// params_id = rsl_radar_params_manager::getInstance().get_params_id();

			range_protect_units = 4;
			range_average_units = 32;
			velocity_protect_units = 4;
			velocity_average_units = 32;

			r_threshold = powf(10, 13.0/20);
			v_threshold = powf(10, 13.0/20);
			clutter_threshold = powf(10, 13.0/20);

			velocity_max_snr_target_cnt = 1;

			clutter_suppression_enable = 0;

			zero_doppler_channel_number = 1;
			noise_average_units = 64;

			range_offset_value = 38;

			AD_clock = 64e6;
			Carrier_frequency = 5.262e9;
			Freq_point_interval = 2e7;

			wide_narrow_cohesion_enable = 1;

			params_id = 0;
		}
	}

	// 先做距离维恒虚警（滑窗）
	// 再对距离维过门限的点进行距离维相邻波门凝聚
	// 再进行速度维相邻波门凝聚
	// 最后对剩下的点进行速度维恒虚警
	void rsl_cfar::task_cfar(cpi_data_t& cpi_data, mtd_abs_t& mtd_abs_data, clutter_map_t& clutter_map_data, bool is_narrow_pulse, target_map_t& target_output){
		//提取参数
		int v_total_units = 0;
		int r_total_units = 0;
		int r_half_pc_unit = 0;
		int prf_total_pulse = cpi_data.cmd_params->pulse_params.prf_total_pulse;

		//宽脉冲距离单元需要偏移一个窄脉冲宽度
		int r_unit_offset = 0;
		float** data_ptr = nullptr;

		if(is_narrow_pulse){
			v_total_units = rsl_win_func_manager::getInstance().nextpow2(prf_total_pulse);
			r_total_units = cpi_data.cmd_params->pulse_params.narrow_pulse_valid_point/8;
			r_half_pc_unit = cpi_data.cmd_params->PC_params.narrow_pulse_halfpc_points;
			r_unit_offset = 0;
			data_ptr = mtd_abs_data.narrow_sum;
		}else{
			v_total_units = rsl_win_func_manager::getInstance().nextpow2(prf_total_pulse);
			r_total_units = cpi_data.cmd_params->pulse_params.wide_pulse_valid_point/8;
			r_half_pc_unit = cpi_data.cmd_params->PC_params.wide_pulse_halfpc_points;
			r_unit_offset = cpi_data.cmd_params->PC_params.narrow_pulse_width;
			data_ptr = mtd_abs_data.wide_sum;
		}

		//分辨率
		float fd = (float)AD_clock / cpi_data.cmd_params->pulse_params.prf_period;
		float fc = Carrier_frequency + (cpi_data.cmd_params->work_mode.mode>>6 & 0x0000002f) * Freq_point_interval;
		float r_resolution = 18.75f;
		float v_resolution = fd * LIGHT_SPEED / 2 /fc / v_total_units;

		//距离维凝聚后目标
		target_map_t target_r_cohesion;
		//距离维速度维凝聚后目标
		target_map_t target_rv_cohesion;

		//距离维滑窗GO-CFAR加凝聚
		//跳过零速度左右各ZEROS_DOPPLER_CHN个通道
		//TODO: 对于高俯仰波束，速度0通道附近跳过点数是否应该减小？
		// #pragma omp parallel for num_threads(CFAR_THREAD_NUM) schedule(static, 256 / CFAR_THREAD_NUM)
		for (int v_index = zero_doppler_channel_number + 1; v_index < v_total_units - zero_doppler_channel_number; v_index++)
		{
			//初始化滑窗
			//       back_begin   (average)   back_end  (protect)   detect  (protect)  forward_begin  (average)  forward_end
			//            ↓                      ↓                     ↓                    ↓                        ↓
			// -----------|++++++++++++++++++++++|--------------------||--------------------|++++++++++++++++++++++++|------------------

			int detect_unit     = 0;
			int forward_begin   = r_half_pc_unit + range_protect_units + 1;
			int forward_end     = r_half_pc_unit + range_protect_units + range_average_units + 1;
			int back_begin      = 0;
			int back_end        = 0;
			float forward_sum = 0.0f;
			float back_sum = 0.0f;

			float noise_amp = 0.0f;
			float threshold = 0.0f;

			//保存未凝聚的距离维过门限点
			target_map_t target_r_tmp;

			//计算前部初始均值
			for (int index = forward_begin; index < forward_end; index++)
			{
				forward_sum += data_ptr[index][v_index];
			}
			
			//滑窗
			for (detect_unit = r_half_pc_unit; detect_unit < r_total_units; detect_unit++)
			{
				//判断当前点是否过门限
				//单元平均选大
				if(forward_sum == 0.0f){
					//前部边缘部分
					noise_amp = back_sum / range_average_units;
				}else if(back_sum == 0.0f){
					//后部边缘部分
					noise_amp = forward_sum / range_average_units;
				}else{
					//中间部分
					noise_amp = max(forward_sum, back_sum) / range_average_units;
				}
				//门限比较
				threshold = noise_amp * r_threshold;

				if(data_ptr[detect_unit][v_index] > threshold){
					//过门限的点
					target_t cfar_tmp;
					cfar_tmp.init_null();
					cfar_tmp.target_amp = data_ptr[detect_unit][v_index];
					cfar_tmp.r_unit = detect_unit;
					cfar_tmp.v_unit = v_index;
					target_r_tmp.insert(target_map_t::value_type(v_index<<16 | detect_unit & 0x0000ffff, cfar_tmp));
				}

				//移动滑窗并计算sum
				//前部计算
				if(forward_end == r_total_units){
					//前部滑到边缘，不参与平均选大
					forward_sum = 0.0f;
				}else{
					forward_sum += data_ptr[forward_end][v_index];
					forward_sum -= data_ptr[forward_begin][v_index];
					forward_end++;
					forward_begin++;
				}

				//后部计算
				if(detect_unit < r_half_pc_unit + range_protect_units + range_average_units + 1){
					//后部未滑出边缘，不参与平均选大
					back_sum = 0.0f;
				}else if(detect_unit == r_half_pc_unit + range_protect_units + range_average_units + 1){
					//后部刚好滑出边缘，计算后部初始均值
					back_begin = r_half_pc_unit;
					back_end = r_half_pc_unit + range_average_units + 1;
					for (int index = back_begin; index < back_end; index++)
					{
						back_sum += data_ptr[index][v_index];
					}
				}else{
					back_sum += data_ptr[back_end][v_index];
					back_sum -= data_ptr[back_begin][v_index];
					back_end++;
					back_begin++;
				}
			}

			//距离维凝聚
			if(target_r_tmp.size() > 0){
				
				int last_r_unit = 0;
				target_map_t::iterator iter = target_r_tmp.begin();
				if(iter != target_r_tmp.end()){
					last_r_unit = target_r_tmp.begin()->second.r_unit - 1;
				}

				target_t target_tmp;
				target_tmp.init_null();

				float amp_sum = 0.0f;
				float range_amp_product = 0.0f;
				bool cross_peak = false;
				float last_amp = iter->second.target_amp;
				float max_amp = 0.0f;
				int max_r_unit = 0;
				int max_v_unit = 0;

				for (iter = target_r_tmp.begin(); iter != target_r_tmp.end(); )
				{
					//寻找相邻的过门限点
					if(iter->second.r_unit - last_r_unit == 1){
						if(!cross_peak && iter->second.target_amp >= last_amp){
							//峰值起始，未越过峰值点，且幅度增加
							//记录峰值点位置和幅度
							max_amp = iter->second.target_amp;
							max_r_unit = iter->second.r_unit;
							max_v_unit = iter->second.v_unit;
							amp_sum += iter->second.target_amp;
							range_amp_product += iter->second.target_amp * r_resolution * (iter->second.r_unit + r_unit_offset);
						}else if(!cross_peak && iter->second.target_amp < last_amp){
							//峰值点，刚好越过峰值，幅度开始减小
							amp_sum += iter->second.target_amp;
							range_amp_product += iter->second.target_amp * r_resolution * (iter->second.r_unit + r_unit_offset);
							cross_peak = true;
						}else if(cross_peak && iter->second.target_amp <= last_amp){
							//越过峰值，幅度持续减小
							amp_sum += iter->second.target_amp;
							range_amp_product += iter->second.target_amp * r_resolution * (iter->second.r_unit + r_unit_offset);
						}else if(cross_peak && iter->second.target_amp > last_amp){
							//谷值点，新峰值的开始
							//将前一个目标凝聚
							target_tmp.target_amp = max_amp;
							target_tmp.r_unit = max_r_unit;
							target_tmp.v_unit = max_v_unit;
							target_tmp.r_cohesion_val = range_amp_product/amp_sum;
							#pragma omp critical
							target_r_cohesion.insert(target_map_t::value_type(target_tmp.r_unit<<16 | (target_tmp.v_unit & 0x0000ffff),target_tmp));

							//开始累加下一个峰值
							max_amp = iter->second.target_amp;
							max_r_unit = iter->second.r_unit;
							max_v_unit = iter->second.v_unit;
							amp_sum = 0.0f;
							amp_sum += iter->second.target_amp;
							range_amp_product = 0.0f;
							range_amp_product += iter->second.target_amp * r_resolution * (iter->second.r_unit + r_unit_offset);
							last_amp = iter->second.target_amp;
							cross_peak = false;
						}
						last_amp = iter->second.target_amp;
						last_r_unit = iter->second.r_unit;
					}else{
						//不相邻过门限点
						//将前一个目标凝聚
						target_tmp.target_amp = max_amp;
						target_tmp.r_unit = max_r_unit;
						target_tmp.v_unit = max_v_unit;
						target_tmp.r_cohesion_val = range_amp_product/amp_sum;
						#pragma omp critical
						target_r_cohesion.insert(target_map_t::value_type(target_tmp.r_unit<<16 | (target_tmp.v_unit & 0x0000ffff),target_tmp));

						//开始累加下一个峰值
						max_amp = iter->second.target_amp;
						max_r_unit = iter->second.r_unit;
						max_v_unit = iter->second.v_unit;
						amp_sum = 0.0f;
						amp_sum += iter->second.target_amp;
						range_amp_product = 0.0f;
						range_amp_product += iter->second.target_amp * r_resolution * (iter->second.r_unit + r_unit_offset);
						cross_peak = false;
						last_amp = iter->second.target_amp;
						last_r_unit = iter->second.r_unit;
					}
					iter++;
				}

				//将最后一个目标凝聚
				target_tmp.target_amp = max_amp;
				target_tmp.r_unit = max_r_unit;
				target_tmp.v_unit = max_v_unit;
				target_tmp.r_cohesion_val = range_amp_product/amp_sum;
				#pragma omp critical
				target_r_cohesion.insert(target_map_t::value_type(target_tmp.r_unit<<16 | (target_tmp.v_unit & 0x0000ffff),target_tmp));
			}
		}

		//速度维凝聚
		if(target_r_cohesion.size() > 0){
			
			int last_v_unit = 0;
			target_map_t::iterator iter = target_r_cohesion.begin();
			if(iter != target_r_cohesion.end()){
				last_v_unit = iter->second.v_unit - 1;
			}
			target_t target_tmp;
			target_tmp.init_null();

			float amp_sum = 0.0f;
			float velocity_amp_product = 0.0f;
			bool cross_peak = false;
			float last_amp = iter->second.target_amp;
			float max_amp = 0.0f;
			int max_r_unit = 0;
			int max_v_unit = 0;
			float max_r_cohesion = 0.0f;

			for (iter = target_r_cohesion.begin(); iter != target_r_cohesion.end(); )
			{
				//寻找相邻的过门限点
				if(iter->second.v_unit - last_v_unit == 1){
					if(!cross_peak && iter->second.target_amp >= last_amp){
						//峰值起始，未越过峰值点，且幅度增加
						//记录峰值点位置和幅度
						max_amp = iter->second.target_amp;
						max_r_unit = iter->second.r_unit;
						max_v_unit = iter->second.v_unit;
						max_r_cohesion = iter->second.r_cohesion_val;
						amp_sum += iter->second.target_amp;
						velocity_amp_product += iter->second.target_amp * v_resolution * iter->second.v_unit;
					}else if(!cross_peak && iter->second.target_amp < last_amp){
						//峰值点，刚好越过峰值，幅度开始减小
						amp_sum += iter->second.target_amp;
						velocity_amp_product += iter->second.target_amp * v_resolution * iter->second.v_unit;
						cross_peak = true;
					}else if(cross_peak && iter->second.target_amp <= last_amp){
						//越过峰值，幅度持续减小
						amp_sum += iter->second.target_amp;
						velocity_amp_product += iter->second.target_amp * v_resolution * iter->second.v_unit;
					}else if(cross_peak && iter->second.target_amp > last_amp){
						//谷值点，新峰值的开始
						//将前一个目标凝聚
						target_tmp.target_amp = max_amp;
						target_tmp.r_unit = max_r_unit;
						target_tmp.v_unit = max_v_unit;
						target_tmp.r_cohesion_val = max_r_cohesion;
						target_tmp.v_cohesion_val = velocity_amp_product/amp_sum;
						if(target_tmp.v_cohesion_val < v_total_units * v_resolution / 2){
							target_tmp.v_cohesion_val = -target_tmp.v_cohesion_val;
						}else{
							target_tmp.v_cohesion_val = v_total_units * v_resolution - target_tmp.v_cohesion_val;
						}
						target_rv_cohesion.insert(target_map_t::value_type(target_tmp.v_unit<<16 | (target_tmp.r_unit & 0x0000ffff),target_tmp));

						//开始累加下一个峰值
						max_amp = iter->second.target_amp;
						max_r_unit = iter->second.r_unit;
						max_v_unit = iter->second.v_unit;
						max_r_cohesion = iter->second.r_cohesion_val;
						amp_sum = 0.0f;
						amp_sum += iter->second.target_amp;
						velocity_amp_product = 0.0f;
						velocity_amp_product += iter->second.target_amp * v_resolution * iter->second.v_unit;
						last_amp = iter->second.target_amp;
						cross_peak = false;
					}
					last_amp = iter->second.target_amp;
					last_v_unit = iter->second.v_unit;
				}else{
					//不相邻过门限点
					//将前一个目标凝聚
					target_tmp.target_amp = max_amp;
					target_tmp.r_unit = max_r_unit;
					target_tmp.v_unit = max_v_unit;
					target_tmp.r_cohesion_val = max_r_cohesion;
					target_tmp.v_cohesion_val = velocity_amp_product/amp_sum;
					if(target_tmp.v_cohesion_val < v_total_units * v_resolution / 2){
						target_tmp.v_cohesion_val = -target_tmp.v_cohesion_val;
					}else{
						target_tmp.v_cohesion_val = v_total_units * v_resolution - target_tmp.v_cohesion_val;
					}
					target_rv_cohesion.insert(target_map_t::value_type(target_tmp.v_unit<<16 | (target_tmp.r_unit & 0x0000ffff),target_tmp));

					//开始累加下一个峰值
					max_amp = iter->second.target_amp;
					max_r_unit = iter->second.r_unit;
					max_v_unit = iter->second.v_unit;
					max_r_cohesion = iter->second.r_cohesion_val;
					amp_sum = 0.0f;
					amp_sum += iter->second.target_amp;
					velocity_amp_product = 0.0f;
					velocity_amp_product += iter->second.target_amp * v_resolution * iter->second.v_unit;
					cross_peak = false;
					last_amp = iter->second.target_amp;
					last_v_unit = iter->second.v_unit;
				}
				iter++;
			}

			//将最后一个目标凝聚
			target_tmp.target_amp = max_amp;
			target_tmp.r_unit = max_r_unit;
			target_tmp.v_unit = max_v_unit;
			target_tmp.r_cohesion_val = max_r_cohesion;
			target_tmp.v_cohesion_val = velocity_amp_product/amp_sum;
			if(target_tmp.v_cohesion_val < v_total_units * v_resolution / 2){
				target_tmp.v_cohesion_val = -target_tmp.v_cohesion_val;
			}else{
				target_tmp.v_cohesion_val = v_total_units * v_resolution - target_tmp.v_cohesion_val;
			}
			target_rv_cohesion.insert(target_map_t::value_type(target_tmp.v_unit<<16 | (target_tmp.r_unit & 0x0000ffff),target_tmp));
		}

		//按距离门保存目标，选取该距离门上信噪比最大的n个目标输出
		multimap<int, target_t> target_v_snr_max;
		//速度维恒虚警 SO-CFAR
		for (target_map_t::iterator iter = target_rv_cohesion.begin(); iter != target_rv_cohesion.end(); iter++)
		{
			int detect_unit = iter->second.v_unit;
			float forward_sum = 0.0f;
			float back_sum = 0.0f;
			float threshold = 0.0f;
			int forward_begin = 0;
			int forward_end = 0;
			int back_begin = 0;
			int back_end = 0;

			//计算参考单元门限
			if(detect_unit < velocity_protect_units + velocity_average_units + zero_doppler_channel_number){
				//后部处于边缘，只计算前部
				forward_begin = detect_unit + velocity_protect_units + 1;
				forward_end = detect_unit + velocity_protect_units + velocity_average_units + 1;
				for (int v_index = forward_begin; v_index < forward_end; v_index++)
				{
					forward_sum += data_ptr[iter->second.r_unit][v_index];
				}
				threshold = forward_sum / velocity_average_units * v_threshold;
			}else if(detect_unit < v_total_units - (range_protect_units + range_average_units + zero_doppler_channel_number)){
				//前部处于边缘，只计算后部
				back_begin = detect_unit - velocity_protect_units - velocity_average_units;
				back_end = detect_unit - velocity_protect_units;
				for (int v_index = back_begin; v_index < back_end; v_index++)
				{
					back_sum += data_ptr[iter->second.r_unit][v_index];
				}
				threshold = back_sum / velocity_average_units * v_threshold;
			}else{
				//中间部分，单元平均选小
				forward_begin = detect_unit + velocity_protect_units + 1;
				forward_end = detect_unit + velocity_protect_units + velocity_average_units + 1;
				for (int v_index = forward_begin; v_index < forward_end; v_index++)
				{
					forward_sum += data_ptr[iter->second.r_unit][v_index];
				}

				back_begin = detect_unit - velocity_protect_units - velocity_average_units;
				back_end = detect_unit - velocity_protect_units;
				for (int v_index = back_begin; v_index < back_end; v_index++)
				{
					back_sum += data_ptr[iter->second.r_unit][v_index];
				}

				threshold = min(forward_sum,back_sum) / velocity_average_units * v_threshold;
			}
			
			//杂波图门限
			if(clutter_map_data.clutter_map_valid && clutter_suppression_enable != 1){
				float clutter_amp = 0.0f;
				if(is_narrow_pulse){
					clutter_amp = clutter_map_data.narrow_sum[iter->second.r_unit][iter->second.v_unit];
				}else{
					clutter_amp = clutter_map_data.wide_sum[iter->second.r_unit][iter->second.v_unit];
				}
				
				threshold = max(threshold, clutter_amp * clutter_threshold);
			}
			
			//判断是否过门限
			if(data_ptr[iter->second.r_unit][iter->second.v_unit] > threshold){
				//过门限的点
				//计算噪声
				/// TODO: 此处应该去掉最大的一部分和最小的一部分，否则噪声估计不准
				int average_units = noise_average_units * 3 / 4;
				vector<float> tmp;
				for (int v_index = v_total_units/2 - noise_average_units/2; v_index < v_total_units/2 + noise_average_units/2; v_index++)
				{
					tmp.push_back(data_ptr[iter->second.r_unit][v_index]);
				}
				sort(tmp.begin(), tmp.end());
				iter->second.noise_amp = accumulate(tmp.begin(), next(tmp.begin(), average_units), 0.0f) / average_units;
				iter->second.snr = iter->second.target_amp / iter->second.noise_amp;
				iter->second.is_narrow_pulse = is_narrow_pulse;
				iter->second.r_total_unit = r_total_units;
				iter->second.v_total_unit = v_total_units;
#if STATUS_INFO_VALID
				iter->second.time_stamp = (unsigned long)(cpi_data.gnss_msg.time * 1000.0f);
#else
				iter->second.time_stamp = (unsigned long)(cpi_data.cmd_params->pulse_params.cpi_cnt / 4 * 281.5);
#endif
				target_v_snr_max.insert(make_pair(iter->second.r_unit, iter->second));
			}
		}

		//速度维选最大的n个目标输出
		multimap<float, target_t> snr_max_tmp;
		multimap<int, target_t>::iterator iter = target_v_snr_max.begin();
		int curr_r_unit = iter->first;
		int cnt = 0;
		for (; iter != target_v_snr_max.end(); iter++)
		{
			if(iter->first != curr_r_unit){
				for (multimap<float, target_t>::iterator iter2 = snr_max_tmp.begin(); iter2 != snr_max_tmp.end(); iter2++)
				{
					if(cnt < velocity_max_snr_target_cnt){
						target_output.insert(make_pair(curr_r_unit<<16 | (cnt), iter2->second));
						cnt++;
					}else{
						break;
					}
				}

				curr_r_unit = iter->first;
				cnt = 0;

				snr_max_tmp.clear();
			}
			snr_max_tmp.insert(make_pair(-iter->second.snr, iter->second));
		}

		for (multimap<float, target_t>::iterator iter2 = snr_max_tmp.begin(); iter2 != snr_max_tmp.end(); iter2++)
		{
			if(cnt < velocity_max_snr_target_cnt){
				target_output.insert(make_pair(curr_r_unit<<16 | (cnt), iter2->second));
				cnt++;
			}else{
				break;
			}
		}
	}

	// 宽窄脉冲目标凝聚
	void rsl_cfar::task_cohesion(cpi_data_t& cpi_data, target_map_t& target_narrow, target_map_t& target_wide, int azi_beam_num, target_map_t& target_output){
		//计算参数
		float r_cohesion_thd = 18.75f;
		float fd = (float)AD_clock / cpi_data.cmd_params->pulse_params.prf_period;
		float fc = Carrier_frequency + (cpi_data.cmd_params->work_mode.mode>>6 & 0x0000002f) * Freq_point_interval;
		float v_cohesion_thd = fd * LIGHT_SPEED / 2 /fc / rsl_win_func_manager::getInstance().nextpow2(cpi_data.cmd_params->pulse_params.prf_total_pulse);

		//速度凝聚门限
		// v_cohesion_thd = 2;

		// target_map_t target_nw_cohesion;

		//宽窄脉冲凝聚
		target_map_t::iterator iter;
		target_map_t target_map_tmp;
		for (iter = target_narrow.begin(); iter != target_narrow.end(); iter++)
		{
			target_map_tmp.insert(target_map_t::value_type(iter->first,iter->second));
		}
		for (iter = target_wide.begin(); iter != target_wide.end(); iter++)
		{
			//宽脉冲实际距离单元应该要加上窄脉冲宽度
			target_map_tmp.insert(target_map_t::value_type(iter->first + cpi_data.cmd_params->PC_params.narrow_pulse_width,iter->second));
		}

		if(target_map_tmp.empty()){
			return;
		}

		target_map_t::iterator iter_forward;
		target_map_t::iterator iter_back;

		//距离+速度凝聚
		if(wide_narrow_cohesion_enable == 1){
			for (iter_forward = target_map_tmp.begin(); iter_forward != target_map_tmp.end(); iter_forward++)
			{
				for (iter_back = iter_forward; ++iter_back != target_map_tmp.end(); ){
					//判断两个目标是否在凝聚范围
					if(fabsf(iter_forward->second.r_cohesion_val - iter_back->second.r_cohesion_val) < r_cohesion_thd \
					&& fabsf(iter_forward->second.v_cohesion_val - iter_back->second.v_cohesion_val) < v_cohesion_thd){
						//取信噪比大的目标
						if(iter_forward->second.target_amp / iter_forward->second.noise_amp < iter_back->second.target_amp / iter_back->second.noise_amp){
							iter_forward->second.flag = 1;
							iter_back->second.flag = 0;
						}else{
							iter_back->second.flag = 1;
							iter_forward->second.flag = 0;
						}
					}
				}
			}
		}
		
		for (iter = target_map_tmp.begin(); iter != target_map_tmp.end(); iter++){
			if(iter->second.flag == 0){
				iter->second.flag = 0;
				iter->second.azimuth_beam_num = cpi_data.cmd_params->work_mode.azimuth_beam_num[azi_beam_num];
				iter->second.pitch_beam_num = cpi_data.cmd_params->work_mode.pitch_beam_num;
				iter->second.cpi_index = cpi_data.cmd_params->pulse_params.cpi_cnt;
				iter->second.pri = cpi_data.cmd_params->pulse_params.prf_period / 64;
				iter->second.r_cohesion_val += range_offset_value;

				//计算RCS
				iter->second.rcs = 40 * log10f(iter->second.r_cohesion_val) + 20 * log10f(iter->second.target_amp);

				target_output.insert(target_map_t::value_type(iter->first,iter->second));
			}
		}
	}
	
	void rsl_cfar::cfar(cpi_data_t& cpi_data, mtd_abs_t &mtd_data, clutter_map_t& clutter_map_data, int azi_beam_num, target_map_t &target_output){
#if CFAR_DEBUG == 1
		rsl_debug_tools::debug_tools cfar_debug;
		cfar_debug.start();
#endif 
		params_update(cpi_data.cmd_params->work_mode.params_id);

		azi_sector_num = azi_beam_num;

		target_map_t target_narrow;
		target_map_t target_wide;
		//恒虚警
		task_cfar(cpi_data, mtd_data, clutter_map_data, true, target_narrow);
		task_cfar(cpi_data, mtd_data, clutter_map_data, false, target_wide);

		//宽窄脉冲凝聚+距离/速度维凝聚
		task_cohesion(cpi_data, target_narrow, target_wide, azi_beam_num, target_output);

#if CFAR_DEBUG == 1
		cfar_debug.stop();
		time_cfar += cfar_debug.get_interval_us();
		SPDLOG_LOGGER_INFO(cfar_debug._logger, " CFAR: time: {:.2f} ms, target cnt: {}", time_cfar/1000, target_narrow.size() + target_wide.size());
		// for (target_map_t::iterator iter = target_output.begin(); iter != target_output.end(); iter++)
		// {
		//     std::cout<<iter->second<<std::endl;
		// }

		// //导出MTD数据至matlab
		// int narrow_points = cpi_data.cmd_params->pulse_params.narrow_pulse_valid_point/8;
		// int wide_points = cpi_data.cmd_params->pulse_params.wide_pulse_valid_point/8;
		// int prf_total_pulse = cpi_data.cmd_params->pulse_params.prf_total_pulse;
		// int mtd_fft_points = rsl_win_func_manager::getInstance().nextpow2(prf_total_pulse);
		// int azi_beam = cpi_data.cmd_params->work_mode.azimuth_beam_num[azi_beam_num];
		// int pit_beam = cpi_data.cmd_params->work_mode.pitch_beam_num;

		// string file_path = "/home/hhky-centos/lzw/radar_signal_process/debug/MTD_narrow_sum_" + to_string(azi_beam_num) + ".dat";
		// debug_tools::debug_with_matlab(file_path,type_float,narrow_points,mtd_fft_points,(char*)mtd_data.narrow_sum[0]);

		// file_path = "/home/hhky-centos/lzw/radar_signal_process/debug/MTD_wide_sum_" + to_string(azi_beam_num) + ".dat";
		// debug_tools::debug_with_matlab(file_path,type_float,wide_points,mtd_fft_points,(char*)mtd_data.wide_sum[0]);

		// file_path = "/home/hhky-centos/lzw/radar_signal_process/debug/Target_azi" + to_string(azi_beam_num) + ".dat";
		// debug_tools::debug_with_matlab(file_path, target_output, azi_beam_num, 0);
#endif 

		

#if CLUTTER_MAP_DEBUG == 1

		//保存杂波图
		if(azi_beam == 51 && pit_beam == 0){
			string file_path = "/home/hhky-centos/lzw/radar_signal_process/debug/MTD_narrow_sum_" + to_string(azi_beam) + ".dat";
			debug_tools::debug_with_matlab(file_path,type_float,narrow_points,mtd_fft_points,(char*)mtd_data.narrow_sum[0]);

			file_path = "/home/hhky-centos/lzw/radar_signal_process/debug/MTD_wide_sum_" + to_string(azi_beam) + ".dat";
			debug_tools::debug_with_matlab(file_path,type_float,wide_points,mtd_fft_points,(char*)mtd_data.wide_sum[0]);

			file_path = "/home/hhky-centos/lzw/radar_signal_process/debug/Target_azi" + to_string(azi_beam) + ".dat";
			debug_tools::debug_with_matlab(file_path, target_output, azi_beam_num, 0);

			file_path = "/home/hhky-centos/lzw/radar_signal_process/debug/Clutter_map_narrow_" + to_string(azi_beam) + ".dat";
			debug_tools::debug_with_matlab(file_path,type_float,narrow_points,mtd_fft_points,(char*)clutter_map_data.narrow_sum[0]);

			file_path = "/home/hhky-centos/lzw/radar_signal_process/debug/Clutter_map_wide_" + to_string(azi_beam) + ".dat";
			debug_tools::debug_with_matlab(file_path,type_float,wide_points,mtd_fft_points,(char*)clutter_map_data.wide_sum[0]);
		}


#endif



	}
	
	
} // namespace radar_signal_process

#endif
