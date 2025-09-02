#pragma once

#ifndef __RSL_DBF_BASE_HPP
#define __RSL_DBF_BASE_HPP

#include <complex>
#include <Eigen/Dense>

#include "omp.h"

#include "rsl_data_structure.hpp"
#include "rsl_radar_params_manager.hpp"
#include "rsl_win_func_manager.hpp"
#include "debug_tools.hpp"

using namespace::std;
using namespace::rsl_debug_tools;
using namespace::Eigen;

typedef Matrix<complex<float>, Dynamic, Dynamic, RowMajor>	MatrixDBF_beam;
typedef Matrix<float, 1, Dynamic, RowMajor>	MatrixSTC_coeff;
typedef Map<MatrixDBF_beam> MapDDC_beam;
typedef Map<MatrixSTC_coeff> MapSTC_coeff;

namespace radar_signal_process{

#define STC_COMPESATE_THREAD_NUM THREADS_NUM_FOR_BEAM_0

	/**
	 * @brief     DBF处理基类
	 */
    class rsl_dbf_base
    {
    private:
        /// @brief 参数ID，用于判断参数是否更新
        int params_id = 0;

		/// @brief 调试工具
		rsl_debug_tools::debug_tools debug_tool;

		/// @brief 时间计量
		static float time;

		/// @brief 处理计数
		static int cnt;

    public:
        /**
         * @brief     构造函数
         */
        rsl_dbf_base();
		
		/**
		 * @brief     析构函数
		 */
        ~rsl_dbf_base();

		/**
		 * @brief     DBF函数，用户调用函数，完成指定方位扇区的DBF处理和STC配平；
		 * @param     [in] cpi_data  CPI数据 
		 * @param     [in] azi_sector_num 方位扇区号，0~2
		 * @param     [in] dbf_output 输出DBF数据
		 */
		void dbf(cpi_data_t& cpi_data, int azi_sector_num, dbf_t& dbf_output);
	protected:
		// /// @brief STC开关标志
		// int stc;

        // /// @brief STC配平使能
        // int stc_compensate;

        // /// @brief STC配平系数
        // stc_compensate_value_t  stc_compensate_value;

		/**
		 * @brief     参数更新函数，由子类实现；基类控制参数何时更新，子类控制哪些参数需要更新；
		 */
		virtual void params_update() = 0;
		
		/**
		 * @brief     DBF处理函数，由子类实现具体算法
		 * @param     [in] cpi_data  CPI数据
		 * @param     [in] azi_sector_num 方位扇区号，0~2
		 * @param     [in] dbf_output 输出DBF数据
		 */
        virtual void process(cpi_data_t& cpi_data, int azi_sector_num, dbf_t& dbf_output) = 0;

		// /**
		//  * @brief     STC配平函数，在DBF后配平，已有默认实现，子类可以覆盖
		//  * @param     [in] cpi_data  CPI数据
		//  * @param     [in] dbf_output DBF数据
		//  */
		// virtual void stc_compensate_process(cpi_data_t &cpi_data, int azi_sector_num, dbf_t &dbf_output);
		
		/**
		 * @brief     Debug函数，供子类实现Debug功能
		 */
		virtual void debug();
    };
}

#endif