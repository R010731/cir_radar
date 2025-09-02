/**
 * @file      debug_tools.cpp
 * @author    lizhengwei (waiwaylee@foxmail.com)
 * @version   1.0
 * @date      2023-12-12
 * @brief     调试工具类的实现
 */
#include <ctime>
#include "debug_tools.hpp"
#include "rsl_win_func_manager.hpp"

namespace rsl_debug_tools{

	std::shared_ptr<spdlog::logger> debug_tools::_logger;
	std::atomic<bool> debug_tools::init_status;

	debug_tools::debug_tools(/* args */)
	{
		try
		{
			if(debug_tools::init_status == false){
				debug_tools::init_status = true;

				// spdlog::set_async_mode(8192);

				// 基于当前系统的当前日期/时间
   				time_t now = time(0);
				tm *gmtm = gmtime(&now);

				std::string file_path = "logs/rsl_log-" + to_string(gmtm->tm_year + 1900) + "-" + to_string(gmtm->tm_mon + 1) + "-" + to_string(gmtm->tm_mday) + " " + to_string(gmtm->tm_hour + 8) + "-" + to_string(gmtm->tm_min) + "-" + to_string(gmtm->tm_sec) + ".txt";

				auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
				console_sink->set_level(spdlog::level::debug);
				console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%l][%t](%s:%#): %v");
				
				// auto daily_sink = spdlog::create_async<spdlog::sinks::basic_file_sink_mt>("async_file_logger", "logs/async_log.txt");
				auto daily_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_path, true);
				// auto daily_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/rsl_log.txt", 23, 59);
				daily_sink->set_level(spdlog::level::info);
				daily_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%l][%t](%s:%#): %v");

				spdlog::sinks_init_list sink_list = { daily_sink, console_sink };

				spdlog::logger logger("rsp_logger", sink_list.begin(), sink_list.end());

				logger.flush_on(spdlog::level::err);

				// or you can even set rsp_logger logger as default logger
				spdlog::set_default_logger(std::make_shared<spdlog::logger>("rsp_logger", spdlog::sinks_init_list({console_sink, daily_sink})));

				debug_tools::_logger = spdlog::get("rsp_logger");

			}
		}
		catch (const spdlog::spdlog_ex& ex)
		{
			std::cout << "Log initialization failed: " << ex.what() << std::endl;
		}
	}
		
	debug_tools::~debug_tools()
	{
		
	}

	timeval debug_tools::start(){
		gettimeofday(&m_start, NULL);
		return m_start;
	}

	timeval debug_tools::stop(){
		gettimeofday(&m_stop, NULL);
		return m_stop;
	}

	float debug_tools::get_interval_us(){
		return 1000000.0f * (m_stop.tv_sec - m_start.tv_sec) + m_stop.tv_usec - m_start.tv_usec;
	}

	void debug_tools::debug_with_matlab(string file_name, debug_data_type_t data_type, int row, int col, char * data ){
		
		//打开数据文件
		ofstream out_file(file_name, ios::binary|ios::out|ios::trunc);
		if(!out_file.is_open()){
			cout<<"Debug_with_matlab: Error: Failed to open file!";
		}

		//写入头部信息
		int head[4] = {data_type, row, col, 0};
		out_file.write((char *)head,16);

		//写入数据
		int len = col*row;
		switch (data_type)
		{
		case type_int16:
			len*=2;
			break;
		case type_int:
		case type_float:
			len*=4;
			break;
		case type_long:
		case type_double:
		case type_complex_float:
			len*=8;
			break;
		case type_complex_double:
			len*=16;
			break;
		default:
			break;
		}
		out_file.write(data,len);

		out_file.close();

		return;
	}

	void debug_tools::debug_with_matlab(string file_name, radar_signal_process::target_map_t target, int azi_beam_num, int pit_beam_num){
		//打开数据文件
		ofstream out_file(file_name, ios::binary|ios::out|ios::trunc);
		if(!out_file.is_open()){
			cout<<"Debug_with_matlab: Error: Failed to open file!";
		}

		//写入头部信息
		int head[4] = {(int)target.size(), azi_beam_num, pit_beam_num, 0};
		out_file.write((char *)head,16);
		
		radar_signal_process::target_map_t::iterator iter;
		for ( iter = target.begin(); iter != target.end(); iter++)
		{
			out_file.write((char *)&iter->second,sizeof(radar_signal_process::target_t));
		}

		out_file.close();

		return;
	}

	// 目标过滤条件
	float min_range = 0;
	float max_range = 30000;
	float min_velocity = 9.7;
	float max_velocity = 11.3;
	float min_azimuth = 283;
	float max_azimuth = 293;
	float min_pitch = -30;
	float max_pitch = 60;
	ofstream tf_outfile;
	int target_cnt = 0;

	void debug_tools::target_filter(radar_signal_process::target_map_t target, vector<radar_signal_process::cmd_params_t> cmd_params, vector<radar_signal_process::mtd_abs_t> mtd_abs){

		min_velocity = FLOAT_PARAMS("Simulator_target_velocity") - 0.3f;
		max_velocity = FLOAT_PARAMS("Simulator_target_velocity") + 0.3f;
		min_azimuth = FLOAT_PARAMS("Simulator_target_azimuth") - 5.0f;
		max_azimuth = FLOAT_PARAMS("Simulator_target_azimuth") + 5.0f;

		if(!tf_outfile.is_open()){
			tf_outfile.open("/home/hhky-centos/lzw/radar_signal_process/debug/target_fliter_output/target_fliter_output.dat", ios::binary|ios::out|ios::trunc);
			//写入头部信息
			tf_outfile.seekp(0,tf_outfile.beg);
			int head[4] = {target_cnt, 0, 0, 0};
			tf_outfile.write((char *)head,16);
		}
		
		tf_outfile.seekp(sizeof(radar_signal_process::target_t) * target_cnt + 16, tf_outfile.beg);
		radar_signal_process::target_map_t::iterator iter = target.begin();
		for (;iter != target.end();iter ++)
		{
			if(	iter->second.r_cohesion_val > min_range && iter->second.r_cohesion_val < max_range &&
				iter->second.v_cohesion_val > min_velocity && iter->second.v_cohesion_val < max_velocity && iter->second.pitch > min_pitch && iter->second.pitch < max_pitch
			){
				float azi_tmp1 = iter->second.azimuth - min_azimuth;
				float azi_tmp2 = iter->second.azimuth - max_azimuth;

				if( (fabsf(azi_tmp1) < fabsf(azi_tmp2) && azi_tmp1 > 0) || (fabsf(azi_tmp2) < fabsf(azi_tmp1) && azi_tmp2 < 0) ){
					//保存目标信息
					tf_outfile.write((char *)&iter->second,sizeof(radar_signal_process::target_t));
					//保存距离一维像

					if(iter->second.is_narrow_pulse){
						debug_with_matlab(
						"/home/hhky-centos/lzw/radar_signal_process/debug/target_fliter_output/mtd_abs_" + to_string(target_cnt), \
						type_float, \
						cmd_params[iter->second.pitch_beam_num].pulse_params.narrow_pulse_valid_point/8, \
						radar_signal_process::rsl_win_func_manager::getInstance().nextpow2(cmd_params[iter->second.pitch_beam_num].pulse_params.prf_total_pulse), \
						(char*)mtd_abs[iter->second.pitch_beam_num].narrow_sum[0] \
						);
					}else{
						debug_with_matlab(
						"/home/hhky-centos/lzw/radar_signal_process/debug/target_fliter_output/mtd_abs_" + to_string(target_cnt), \
						type_float, \
						cmd_params[iter->second.pitch_beam_num].pulse_params.wide_pulse_valid_point/8, \
						radar_signal_process::rsl_win_func_manager::getInstance().nextpow2(cmd_params[iter->second.pitch_beam_num].pulse_params.prf_total_pulse), \
						(char*)mtd_abs[iter->second.pitch_beam_num].wide_sum[0] \
						);
					}

					//保存下面波束MTD
					if(iter->second.pitch_beam_num != 0){
						debug_with_matlab(
						"/home/hhky-centos/lzw/radar_signal_process/debug/target_fliter_output/mtd_abs_down_wide_" + to_string(target_cnt), \
						type_float, \
						cmd_params[iter->second.pitch_beam_num - 1].pulse_params.wide_pulse_valid_point/8, \
						radar_signal_process::rsl_win_func_manager::getInstance().nextpow2(cmd_params[iter->second.pitch_beam_num - 1].pulse_params.prf_total_pulse), \
						(char*)mtd_abs[iter->second.pitch_beam_num - 1].wide_sum[0] \
						);

						debug_with_matlab(
						"/home/hhky-centos/lzw/radar_signal_process/debug/target_fliter_output/mtd_abs_down_narrow_" + to_string(target_cnt), \
						type_float, \
						cmd_params[iter->second.pitch_beam_num - 1].pulse_params.narrow_pulse_valid_point/8, \
						radar_signal_process::rsl_win_func_manager::getInstance().nextpow2(cmd_params[iter->second.pitch_beam_num - 1].pulse_params.prf_total_pulse), \
						(char*)mtd_abs[iter->second.pitch_beam_num - 1].narrow_sum[0] \
						);
					}
					//保存上面波束MTD
					if(iter->second.pitch_beam_num != FLOAT_PARAMS("Pitch_total_beam_number") - 1){
						debug_with_matlab(
						"/home/hhky-centos/lzw/radar_signal_process/debug/target_fliter_output/mtd_abs_up_wide_" + to_string(target_cnt), \
						type_float, \
						cmd_params[iter->second.pitch_beam_num + 1].pulse_params.wide_pulse_valid_point/8, \
						radar_signal_process::rsl_win_func_manager::getInstance().nextpow2(cmd_params[iter->second.pitch_beam_num + 1].pulse_params.prf_total_pulse), \
						(char*)mtd_abs[iter->second.pitch_beam_num + 1].wide_sum[0] \
						);

						debug_with_matlab(
						"/home/hhky-centos/lzw/radar_signal_process/debug/target_fliter_output/mtd_abs_up_narrow_" + to_string(target_cnt), \
						type_float, \
						cmd_params[iter->second.pitch_beam_num + 1].pulse_params.narrow_pulse_valid_point/8, \
						radar_signal_process::rsl_win_func_manager::getInstance().nextpow2(cmd_params[iter->second.pitch_beam_num + 1].pulse_params.prf_total_pulse), \
						(char*)mtd_abs[iter->second.pitch_beam_num + 1].narrow_sum[0] \
						);
					}

					target_cnt++;
				}
			}
		}
		
		//写入头部信息
		tf_outfile.seekp(0,tf_outfile.beg);
		int head[4] = {target_cnt, 0, 0, 0};
		tf_outfile.write((char *)head,16);
		tf_outfile.flush();
		// tf_outfile.close();
	}

	int debug_tools::rsl_crc_16_xmodem(const char *ucBuf, int iLen){
		unsigned int byte;
		unsigned char k;
		unsigned short ACC,TOPBIT;
		//初始值0000
		unsigned short iRemainder = 0x0000;
		TOPBIT = 0x8000;
		for (byte = 0; byte < iLen; ++byte)
		{
			ACC = ucBuf[byte];
			iRemainder ^= (ACC << 8);
			for (k = 8; k > 0; --k)
			{
				if (iRemainder & TOPBIT)
				{
					//CRC-16/XMODEM多项式0x1021
					iRemainder = (iRemainder << 1) ^ 0x1021;
				}
				else
				{
					iRemainder = (iRemainder << 1);
				}
			}
		}
		//结果异或值0000
		iRemainder = iRemainder ^ 0x0000;
		return iRemainder;
	}

	std::shared_ptr<spdlog::logger> debug_tools::logger(){
		return debug_tools::_logger;
	}
}

