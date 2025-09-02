/**
 * @file      rsl_rsp_data_save.hpp
 * @author    lizhengwei (waiwaylee@foxmail.com)
 * @version   1.0
 * @date      2023-12-12
 * @brief     实现实时处理数据存储功能
 */
#pragma once

#ifndef __RSL_RSP_DATA_SAVE_HPP
#define __RSL_RSP_DATA_SAVE_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <mutex>
#include <atomic>
#include "rsl_radar_params_manager.hpp"
#include "thread_safety_queue.hpp"

namespace radar_signal_process{
	/**
	 * @brief	软件化处理数据实时存储类
	 * @details
	 * 	- 将软件化处理中各个关键步骤的输出保存到文件，目前包括DBF数据、脉压数据、MTD数据和目标信息等。
	 * 	- 数据按CPI序号和方位扇区号保存到不同的文件中，数据格式参考《软件化处理数据存储格式.xlsx》文档。
	 * 	- 使用matlab读取数据参考rsp_data_save项目，该项目提供了读取各类数据的matlab函数，具体使用方法参考该项目的readme文档。
	 */
	class rsl_rsp_data_save{
	private:
		#pragma pack(1)
		/**
		 * @brief     实时数据存储信息结构体
		 */
		typedef struct DATA_SAVE_INFO{
			int head;					///< 实时数据存储信息帧头
			int data_cate;				///< 数据类别，参考： @link cates cates @endlink
			int azimuth_sector_num;		///< 方位扇区号
			int azimuth_beam_num;		///< 方位波位号
			int pitch_beam_num;			///< 俯仰波位号
			int sum_diff;				///< 和差波束，参考： @link sum_diff sum_diff @endlink
			int narrow_wide;			///< 宽窄脉冲，参考： @link narrow_wide narrow_wide @endlink
			int row;					///< 数据行数
			int col;					///< 数据列数
			int data_type;				///< 数据类型，参考： @link rsl_debug_tools::debug_data_type_t debug_data_type_t @endlink
		}data_save_info_t;
		#pragma pack()

		/**
		 * @brief     存储数据类别枚举
		 */
		enum cates{
			DBF = 0x01,	///< DBF数据
			PC = 0x02,	///< 脉压数据
			MTD = 0x03	///< MTD数据
		};

		/**
		 * @brief     宽窄脉冲枚举
		 */
		enum narrow_wide{
			NARROW = 0x01,	///< 窄脉冲
			WIDE = 0x02		///< 宽脉冲
		};
		
		/**
		 * @brief     和差波束枚举
		 */
		enum sum_diff{
			SUM = 0x01,		///< 和波束
			DIFF = 0x02		///< 差波束
		};

		/**
		 * @brief     数据存储参数，包含文件路径、数据指针等信息
		 */
		struct data_save_params
		{
			string file_path;	///< 文件路径
			char * ram_ptr;		///< 数据指针
			size_t block_size;	///< 数据块大小
		};

		/// @brief 参数ID
		int params_id;

		/// @brief 线程运行标志 
		bool thread_running;

		/// @brief 方位扇区数
		int sector_beam_cnt;

		/// @brief 数据文件存储目录路径
		string data_path;
		/// @brief 目标文件存储目录路径
		string target_path;

		/// @brief 处理数据存储总使能，控制所有处理数据输出（目标信息除外）
		int process_data_save_enable;
		/// @brief DBF数据存储使能
		int DBF_data_save_enable;
		/// @brief 脉压数据存储使能
		int PC_data_save_enable;
		/// @brief MTD数据存储使能
		int MTD_data_save_enable;
		/// @brief 目标数据存储使能
		int target_data_save_enable;
		
		/// @brief 目标过滤参数，按距离、速度、方位、俯仰来过滤目标信息 //@{
		int target_filter_enable;
		float min_range;
		float max_range;
		float min_velocity;
		float max_velocity;
		float min_azimuth;
		float max_azimuth;
		float min_pitch;
		float max_pitch;
		//@} 
		
		/// @brief 目标信息输出文件流
		ofstream tf_outfile;
		/// @brief 目标计数 
		int target_cnt;

		/// @brief 	数据存储线程退出标志
		atomic<bool> exit_flag;

		/// @brief  需要保存的实时处理数据队列
		basic_componests::ts_queue<data_save_params> data_quque;

		/**
		 * @brief     构造函数
		 */
		rsl_rsp_data_save();

		/**
		 * @brief     参数更新函数，用于更新雷达参数
		 * @param     [in] curr_params_id 当前的参数ID
		 */
		void params_update(int curr_params_id);

		/**
		 * @brief     获取当前时间的字符串格式文本
		 * @return    string       当前时间的字符串  
		 */
		string get_time_string();
		
		/**
		 * @brief     根据实时处理数据存储信息结构体，计算存储所需的空间大小
		 * @param     [in] info      实时处理数据存储信息结构体
		 * @return    size_t     所需的空间大小    
		 */
		size_t get_bytes_length(data_save_info_t info);

		/**
		 * @brief     实时处理数据存储线程函数，将实时数据写入文件
		 */
		void fcn_save();

	public:
		/**
		 * @brief     获取实时数据存储类的唯一实例
		 * @return    rsl_rsp_data_save& 实时数据存储类唯一实例的引用
		 */
		static rsl_rsp_data_save& getInstance();

		rsl_rsp_data_save(const rsl_rsp_data_save&)=delete;
    	rsl_rsp_data_save& operator=(const rsl_rsp_data_save&)=delete;

		/**
		 * @brief     析构函数
		 */
		~rsl_rsp_data_save();

		/**
		 * @brief     保存雷达实时处理数据
		 * @param     [in] cpi_data  CPI数据结构体，保存了一个CPI的完整原始数据，参考 @link cpi_data_t cpi_data_t @endlink
		 * @param     [in] dbf_data  DBF输出结果数据，参考 @link dbf_t dbf_t @endlink
		 * @param     [in] pc_data   脉压输出结果数据，参考 @link pc_t pc_t @endlink
		 * @param     [in] mtd_data  MTD输出结果数据，参考 @link mtd_abs_t mtd_abs_t @endlink
		 * @param     [in] azimuth_beam_num 当前CPI的方位波位号
		 * @param     [in] pitch_beam_num 当前CPI的俯仰波位号
		 */
		void save(cpi_data_t &cpi_data, dbf_t &dbf_data, pc_t &pc_data, mtd_abs_t &mtd_data, int azimuth_beam_num, int pitch_beam_num);

		/**
		 * @brief     保存雷达输出的目标数据
		 * @param     [in] cpi_data  CPI数据结构体，保存了一个CPI的完整原始数据，参考 @link cpi_data_t cpi_data_t @endlink
		 * @param     [in] target    目标数据列表，参考 @link target_map_t target_map_t @endlink
		 */
		void save_target(cpi_data_t& cpi_data, target_map_t target);
		
		/**
		 * @brief     开启实时数据存储线程
		 */
		void start();

		/**
		 * @brief     停止实时数据存储线程
		 */
		void stop();
	};
}

#endif