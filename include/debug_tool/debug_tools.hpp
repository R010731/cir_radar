/**
 * @file      debug_tools.hpp
 * @author    lizhengwei (waiwaylee@foxmail.com)
 * @version   1.0
 * @date      2022-10-18
 * @brief     实现运行计时、数据存储、日志等调试功能
 */
#pragma once

#ifndef __DEBUG_TOOLS_HPP
#define __DEBUG_TOOLS_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>

#include "rsl_data_structure.hpp"
#include "rsl_radar_params_manager.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

using namespace::std;

/**
 * @brief     调试工具命名空间，包括调试工具类、日志系统、数据存储系统等组件
 */
namespace rsl_debug_tools{

    /**
     * @brief     调试数据保存时的类型枚举
     */
    typedef enum DATA_TYPE{
        type_int = 1,               ///< 32位有符号整型
        type_long = 2,              ///< 64位有符号整型
        type_float = 3,             ///< 单精度浮点型
        type_double = 4,            ///< 双精度浮点型
        type_complex_float = 5,     ///< 单精度浮点复数型
        type_complex_double = 6,    ///< 双精度浮点复数型
        type_int16 = 7              ///< 16位有符号整型
    }debug_data_type_t;

    /**
     * @brief     调试工具类，包括日志记录、运行计时、信号处理数据保存、目标数据保存等功能；
     */
    class debug_tools
    {
    private:
        /**
         * @brief     计时起始时间戳
         */
        timeval m_start;

        /**
         * @brief     计时结束时间戳
         */
        timeval m_stop;

        /**
         * @brief     日志记录初始化状态
         */
        static std::atomic<bool> init_status;

    public:
        /**
         * @brief     构造函数
         */
        debug_tools(/* args */);

        /**
         * @brief     析构函数
         */
        ~debug_tools();

        /**
         * @brief     计时开始函数
         * @return    timeval     计时开始时刻的时间戳   
         */
        timeval start();

        /**
         * @brief     计时结束函数
         * @return    timeval      计时结束时刻的时间戳       
         */
        timeval stop();

        /**
         * @brief     获取计时开始时刻到计时结束时刻的时间间隔，以微秒为单位
         * @return    float          时间间隔值（us）
         */
        float get_interval_us();

        /**
         * @brief     保存处理数据到文件，从而联合matlab调试
         * @param     [in] file_name 数据保存文件名，需提供绝对路径
         * @param     [in] data_type 数据保存数据类型，参考 @link debug_data_type_t debug_data_type_t @endlink 
         * @param     [in] col       数据列数
         * @param     [in] row       数据行数
         * @param     [in] data      数据指针
         */
        static void debug_with_matlab(string file_name, debug_data_type_t data_type, int col, int row, char * data);

        /**
         * @brief     保存目标信息到文件，从而联合matlab调试
         * @param     [in] file_name 数据保存文件名，需提供绝对路径
         * @param     [in] target    目标数据结构体，参考 @link radar_signal_process::target_map_t target_map_t @endlink 
         * @param     [in] azi_beam_num 方位波位号
         * @param     [in] pit_beam_num 俯仰波位号
         */
        static void debug_with_matlab(string file_name, radar_signal_process::target_map_t target, int azi_beam_num, int pit_beam_num);

        /**
         * @brief     目标过滤函数，过滤目标列表中不满足指定条件的目标
         * @param     [in] target    目标数据结构体，参考 @link radar_signal_process::target_map_t target_map_t @endlink 
         * @param     [in] cmd_params 命令参数
         * @param     [in] mtd_abs   取模后的MTD数据
         */
        static void target_filter(radar_signal_process::target_map_t target, vector<radar_signal_process::cmd_params_t> cmd_params, vector<radar_signal_process::mtd_abs_t> mtd_abs);

        /**
         * @brief     crc校验函数
         * @param     [in] ucBuf 数据指针    
         * @param     [in] iLen  数据长度
         * @return    int   CRC校验值
         */
        static int rsl_crc_16_xmodem(const char *ucBuf, int iLen);

        /**
         * @brief     获取日志记录器实例
         * @return    std::shared_ptr<spdlog::logger>   spdlog日志记录器 
         */
        std::shared_ptr<spdlog::logger> logger();

        /**
         * @brief     spdlog日志记录器 
         */
        static std::shared_ptr<spdlog::logger> _logger;
    };
}

#endif
