/*** 
 * @Author       : lizhengwei waiwaylee@foxmail.com
 * @Date         : 2022-10-20 10:12:43
 * @LastEditors  : lizhengwei
 * @FilePath     : /radar_signal_process/source/rsl_radar_params_manager.cpp
 * @Description  : 
 */


#include "rsl_radar_params_manager.hpp"


namespace radar_signal_process{

	std::mutex rsl_radar_params_manager::fftw_plan_mtx;

	/**
	 * @brief     参数管理器构造函数，从配置文件中读取雷达工作参数，从相位/配平/修正数据中读取数据
	 */
	rsl_radar_params_manager::rsl_radar_params_manager(/* args */)
	{
		params_id = 1;
		load_radar_params();
		SPDLOG_LOGGER_INFO(rsl_debug_tools::debug_tools::_logger, "Load radar configuration: Success");
		load_calibration_data();
		SPDLOG_LOGGER_INFO(rsl_debug_tools::debug_tools::_logger, "Load calibration data: Success");
	}

	rsl_radar_params_manager::~rsl_radar_params_manager()
	{
		if(recv_phase.phase_data != nullptr){
			delete[] recv_phase.phase_data;
		}
		if(trans_phase.phase_data != nullptr){
			delete[] trans_phase.phase_data;
		}

		if(recv_amp_balancing_value.balancing_value != nullptr){
			delete[] recv_amp_balancing_value.balancing_value;
		}
		if(recv_phase_balancing_value.balancing_value != nullptr){
			delete[] recv_phase_balancing_value.balancing_value;
		}

		if(recv_dbf_calibration_value.calibration_value != nullptr){
			delete[] recv_dbf_calibration_value.calibration_value;
		}
		if(recv_target_calibration_value.calibration_value != nullptr){
			delete[] recv_target_calibration_value.calibration_value;
		}
		if(trans_beam_calibration_value.calibration_value != nullptr){
			delete[] trans_beam_calibration_value.calibration_value;
		}

		if(azimuth_angle_coeff.coeff != nullptr){
			delete[] azimuth_angle_coeff.coeff;
		}
		if(pitch_angle_coeff.coeff != nullptr){
			delete[] pitch_angle_coeff.coeff;
		}

		for(auto coeff : stc_compensate_value.narrow_coeff){
			delete[] coeff;
		}
		stc_compensate_value.narrow_coeff.clear();

		for(auto coeff : stc_compensate_value.wide_coeff){
			delete[] coeff;
		}
		stc_compensate_value.wide_coeff.clear();
	}

	/**
	 * @brief     获取雷达参数管理器唯一实例
	 * @return    rsl_radar_params_manager& 
	 */
	rsl_radar_params_manager& rsl_radar_params_manager::getInstance(){
		static rsl_radar_params_manager instance;
        return instance;
	}

	void rsl_radar_params_manager::update_mode_select(int16_t * cmd){
		ini.SetValue("Mode", "Default_work_mode", to_string(cmd[3]).c_str());
		ini.SaveFile(file_path.c_str(), false);
		load_radar_params();
		params_id++;
	}

	void rsl_radar_params_manager::update_work_config(int16_t * cmd){
		vector<string> range_mode_list = {"Short_range","Mid_range","Test_range"};

		bool range_mode_changed = false;
		if(STRING_PARAMS("Default_range_mode") != range_mode_list[cmd[3]]){
			ini.SetValue("Mode", "Default_range_mode", range_mode_list[cmd[3]].c_str());
			range_mode_changed = true;
		}

		ini.SetValue("Scan_mode", "Scan_method", to_string(cmd[4]).c_str());
		ini.SetValue("Scan_mode", "Sector_scan_begin_azimuth", to_string(cmd[5]).c_str());
		ini.SetValue("Scan_mode", "Sector_scan_end_azimuth", to_string(cmd[6]).c_str());

		ini.SetValue("Net", "En_target_package", to_string((cmd[7] >> 0) && 0x0001).c_str());
		ini.SetValue("Net", "En_Status_package", to_string((cmd[7] >> 1) && 0x0001).c_str());
		ini.SetValue("Net", "En_error_bit_package", to_string((cmd[7] >> 2) && 0x0001).c_str());
		ini.SetValue("Net", "En_control_answer_package", to_string((cmd[7] >> 3) && 0x0001).c_str());
		ini.SetValue("Net", "En_jamming_package", to_string((cmd[7] >> 4) && 0x0001).c_str());
		ini.SetValue("Net", "En_noise_package", to_string((cmd[7] >> 5) && 0x0001).c_str());
		ini.SetValue("Net", "En_terrain_package", to_string((cmd[7] >> 6) && 0x0001).c_str());
		ini.SetValue("Net", "En_selftest_package", to_string((cmd[7] >> 7) && 0x0001).c_str());

		ini.SetValue("Signal_process", "DBF_channel_number", to_string(cmd[8] * 2 + 12).c_str());

		ini.SetValue("Source", "Signal_source", to_string(cmd[9]).c_str());
		ini.SetValue("Source", "Internal_source_target_range", to_string(cmd[10]).c_str());
		ini.SetValue("Source", "Internal_source_target_velocity", to_string(cmd[11] * 0.1f).c_str());
		ini.SetValue("Source", "Internal_source_target_azimuth", to_string(cmd[12] * 0.1f).c_str());

		ini.SaveFile(file_path.c_str(), false);
		load_radar_params();

		if(range_mode_changed){
			load_calibration_data();
		}

		params_id++;
		return;
	}

	void rsl_radar_params_manager::update_trans_switch(int16_t * cmd){
		int board_total_num = INT_PARAMS("Antenna_board_total_num");
		int channel_total_num = INT_PARAMS("Channel_num_of_Antenna_board");
		for (int board_index = 0; board_index < board_total_num; board_index++)
		{
			for (int chn_index = 0; chn_index < channel_total_num; chn_index++)
			{
				*t_switch(board_index, chn_index) = (cmd[3+board_index] >> chn_index) && 0x0001;
			}
		}
		params_id++;
		return;
	}

	void rsl_radar_params_manager::update_recv_switch(int16_t * cmd){
		int board_total_num = INT_PARAMS("Antenna_board_total_num");
		int channel_total_num = INT_PARAMS("Channel_num_of_Antenna_board");
		for (int board_index = 0; board_index < board_total_num; board_index++)
		{
			for (int chn_index = 0; chn_index < channel_total_num; chn_index++)
			{
				*r_switch(board_index, chn_index) = (cmd[3+board_index] >> chn_index) && 0x0001;
			}
		}

		params_id++;

		return;
	}
	
	void rsl_radar_params_manager::update_beam_pitch(int16_t * cmd){
		string range_mode = STRING_PARAMS("Default_range_mode");
		int pit_beam_num = INT_PARAMS("Pitch_total_beam_number");

		ini.SetValue(range_mode.c_str(), "Pitch_beam_angle", to_string(cmd[3]).c_str());
		// for (int pit_index = 0; pit_index < pit_beam_num; pit_index++)
		// {
		// 	ini.SetValue(range_mode.c_str(), ("Pitch_beam_offset_deg_" + to_string(pit_index)).c_str(), to_string(cmd[3 + pit_index] - cmd[3]).c_str());
		// }

		ini.SaveFile(file_path.c_str(), false);
		load_radar_params();
		params_id++;
		return;
	}
	
	void rsl_radar_params_manager::update_threshold_control(int16_t * cmd){

		ini.SetValue("Signal_process", "Range_threshold", to_string(cmd[5]*0.1f).c_str());
		ini.SetValue("Signal_process", "Velocity_threshold", to_string(cmd[4]*0.1f).c_str());
		ini.SetValue("Signal_process", "Clutter_map_threshold", to_string(cmd[6]*0.1f).c_str());
		ini.SetValue("Signal_process", "Noise_threshold", to_string(cmd[7]*0.1f).c_str());

		ini.SaveFile(file_path.c_str(), false);
		load_radar_params();
		params_id++;
		return;
	}

	void rsl_radar_params_manager::update_terrain_detection(int16_t * cmd){

		// ini.SetValue("Signal_process", "Range_threshold", to_string(cmd[5]*0.1f).c_str());
		// ini.SetValue("Signal_process", "Velocity_threshold", to_string(cmd[4]*0.1f).c_str());
		// ini.SetValue("Signal_process", "Clutter_map_threshold", to_string(cmd[6]*0.1f).c_str());
		// ini.SetValue("Signal_process", "Noise_threshold", to_string(cmd[7]*0.1f).c_str());

		// ini.SaveFile(file_path.c_str(), false);
		// load_radar_params();
		// params_id++;
		return;
	}

	void rsl_radar_params_manager::update_freq_point(int16_t * cmd){

		ini.SetValue("Waveform", "Freq_point", to_string(cmd[3]).c_str());
	
		ini.SaveFile(file_path.c_str(), false);
		load_radar_params();
		params_id++;
		return;
	}

	void rsl_radar_params_manager::update_frequency_agility(int16_t * cmd){

		// ini.SetValue("Waveform", "Freq_point", to_string(cmd[3]*0.1f).c_str());
	
		// ini.SaveFile(file_path.c_str(), false);
		// load_radar_params();
		// params_id ++;
		return;
	}

	void rsl_radar_params_manager::update_tas_beam(int16_t * cmd){
		

		return;
	}

	void rsl_radar_params_manager::update_signal_process(int16_t * cmd){
		
		if(cmd[3] == 0){
			ini.SetValue("Signal_process", "Band_width", "5e6");
		}else if(cmd[3] == 1){
			ini.SetValue("Signal_process", "Band_width", "2.5e6");
		}

		// ini.SetValue("Signal_process", "MTD_win_func", to_string(cmd[4]).c_str());
		// ini.SetValue("Signal_process", "MTI_method", to_string(cmd[5]).c_str());

		// ini.SetValue("Signal_process", "Range_protect_units", to_string(cmd[10]).c_str());
		// ini.SetValue("Signal_process", "Range_average_units", to_string(cmd[8]).c_str());
		// ini.SetValue("Signal_process", "Velocity_protect_units", to_string(cmd[11]).c_str());
		// ini.SetValue("Signal_process", "Velocity_average_units", to_string(cmd[9]).c_str());

		ini.SetValue("Signal_process", "Azimuth_beam_compress_enable", to_string(cmd[13]).c_str());
		
		ini.SaveFile(file_path.c_str(), false);
		load_radar_params();
		
		params_id++;

		return;
	}

	void rsl_radar_params_manager::update_data_save_config(int16_t *cmd)
	{
		rt_data_save_config.origin_data_save_enable = cmd[3];
		rt_data_save_config.origin_data_save_size_GB = cmd[4];
		rt_data_save_config.origin_data_circular_storage = cmd[5];

		rt_data_save_config.process_data_save_enable = cmd[6];
		rt_data_save_config.DBF_data_save_enable = cmd[7];
		rt_data_save_config.PC_data_save_enable = cmd[8];
		rt_data_save_config.MTD_data_save_enable = cmd[9];

		rt_data_save_config.target_data_save_enable = cmd[10];
		rt_data_save_config.target_filter_enable = cmd[11];
		rt_data_save_config.min_velocity = cmd[12] / 10.0f;
		rt_data_save_config.max_velocity = cmd[13] / 10.0f;
		rt_data_save_config.min_range = cmd[14] / 10.0f;
		rt_data_save_config.max_range = cmd[15] / 10.0f;
		rt_data_save_config.min_azimuth = cmd[16] / 10.0f;
		rt_data_save_config.max_azimuth = cmd[17] / 10.0f;
		rt_data_save_config.min_pitch = cmd[18] / 10.0f;
		rt_data_save_config.max_azimuth = cmd[19] / 10.0f;

		params_id++;

		return;
	}

	/**
	 * @brief     获取当前参数的id，用于判断参数是否已经更新
	 * @return    int	当前参数id            
	 */
	int rsl_radar_params_manager::get_params_id(){
		return params_id;
	}

	/**
	 * @brief     从配置文件中加载雷达工作参数
	 */
	void rsl_radar_params_manager::load_radar_params(){
		ini.SetUnicode();
		//清空当前参数，重新加载
		params_map.clear();
		ini.Reset();
		ini.LoadFile("config/radar_params.ini");

		// get all sections
		CSimpleIniA::TNamesDepend sections;
		ini.GetAllSections(sections);

		for (CSimpleIniA::TNamesDepend::iterator s_iter = sections.begin(); s_iter != sections.end(); s_iter++)
		{
			// 加载所有非工作模式参数
			if(strcmp("Short-range", s_iter->pItem)==0 || strcmp("Mid-range", s_iter->pItem)==0 || strcmp("Long-range", s_iter->pItem)==0){
				continue;
			}

			CSimpleIniA::TNamesDepend keys;
			ini.GetAllKeys(s_iter->pItem, keys);
			
			for (CSimpleIniA::TNamesDepend::iterator k_iter = keys.begin(); k_iter != keys.end(); k_iter++)
			{
				params_map.erase(k_iter->pItem);
				params_map.insert(std::pair<std::string,std::string>(k_iter->pItem, ini.GetValue(s_iter->pItem,k_iter->pItem)));
			}
		}

		// 加载默认工作模式参数
		CSimpleIniA::TNamesDepend keys;
		ini.GetAllKeys(params_map.find("Default_range_mode")->second.c_str(), keys);
		for (CSimpleIniA::TNamesDepend::iterator k_iter = keys.begin(); k_iter != keys.end(); k_iter++)
		{
			params_map.erase(k_iter->pItem);
			params_map.insert(std::pair<std::string,std::string>(k_iter->pItem, ini.GetValue(params_map.find("Default_range_mode")->second.c_str(),k_iter->pItem)));
		}
	}

	void rsl_radar_params_manager::load_recv_phase_data(){
		if(recv_phase.phase_data != nullptr){
			delete[] recv_phase.phase_data;
		}

		std::string file_path = "";
		std::string current_path(getcwd(NULL, 0));
		current_path += "/config/" + params_map.find("Radar_name")->second + "/Recv_phase_data";
		std::string line;
		std::string value;

		//接收相位校正数据
		recv_phase.total_board = atoi(params_map.find("Antenna_board_total_num")->second.c_str());
		recv_phase.total_channel = atoi(params_map.find("Channel_num_of_Antenna_board")->second.c_str());
		recv_phase.total_freq_point = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		recv_phase.phase_data = new float[recv_phase.total_freq_point * recv_phase.total_board * recv_phase.total_channel];
		memset(recv_phase.phase_data, 0, recv_phase.total_freq_point * recv_phase.total_board * recv_phase.total_channel * 4);

		for (int freq_index = 0; freq_index < recv_phase.total_freq_point; freq_index++)
		{
			file_path = current_path + "/Freq" + std::to_string(freq_index) + ".csv";
			if(access(file_path.c_str(), F_OK ) == -1){
				SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
				continue;
			}
			ifstream infile_1(file_path, ios::in);

			getline(infile_1, line);
			
			for (int board_index = 0; board_index < recv_phase.total_board; board_index++)
			{
				getline(infile_1, line);
				stringstream ss(line);
				for (int chn_index = 0; chn_index < recv_phase.total_channel; chn_index++)
				{
					getline(ss, value, ',');
					*recv_phase(freq_index,board_index,chn_index) = (float)atof(value.c_str());
				}
			}
			infile_1.close();
		}

		return;
	}

	void rsl_radar_params_manager::load_trans_phase_data(){
		if(trans_phase.phase_data != nullptr){
			delete[] trans_phase.phase_data;
		}

		std::string file_path = "";
		std::string current_path(getcwd(NULL, 0));
		current_path += "/config/" + params_map.find("Radar_name")->second + "/Trans_phase_data";
		std::string line;
		std::string value;

		//发送相位校正数据
		trans_phase.total_board = atoi(params_map.find("Antenna_board_total_num")->second.c_str());
		trans_phase.total_channel = atoi(params_map.find("Channel_num_of_Antenna_board")->second.c_str());
		trans_phase.total_freq_point = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		trans_phase.phase_data = new float[trans_phase.total_freq_point * trans_phase.total_board * trans_phase.total_channel];
		memset(trans_phase.phase_data, 0, trans_phase.total_freq_point * trans_phase.total_board * trans_phase.total_channel * 4);

		for (int freq_index = 0; freq_index < trans_phase.total_freq_point; freq_index++)
		{
			file_path = current_path + "/Freq" + std::to_string(freq_index) + ".csv";
			if(access(file_path.c_str(), F_OK ) == -1){
				SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
				continue;
			}

			ifstream infile_4(file_path, ios::in);

			getline(infile_4, line);
			
			for (int board_index = 0; board_index < trans_phase.total_board; board_index++)
			{
				getline(infile_4, line);
				stringstream ss(line);
				for (int chn_index = 0; chn_index < trans_phase.total_channel; chn_index++)
				{
					getline(ss, value, ',');
					*trans_phase(freq_index,board_index,chn_index) = (float)atof(value.c_str());
				}
			}
			infile_4.close();
			// infile.clear();
		}

		return;
	}

	void rsl_radar_params_manager::load_recv_compensate_data(){
		if(recv_amp_balancing_value.balancing_value != nullptr){
			delete[] recv_amp_balancing_value.balancing_value;
		}
		if(recv_phase_balancing_value.balancing_value != nullptr){
			delete[] recv_phase_balancing_value.balancing_value;
		}

		std::string file_path = "";
		std::string current_path(getcwd(NULL, 0));
		current_path += "/config/" + params_map.find("Radar_name")->second + "/Recv_amp_compensate_data";
		std::string line;
		std::string value;

		//接收幅相配平数据
		recv_amp_balancing_value.total_freq_point = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		recv_amp_balancing_value.total_board = atoi(params_map.find("Antenna_board_total_num")->second.c_str());
		recv_amp_balancing_value.balancing_value = new float[recv_amp_balancing_value.total_freq_point * recv_amp_balancing_value.total_board];
		memset(recv_amp_balancing_value.balancing_value, 0, recv_amp_balancing_value.total_freq_point * recv_amp_balancing_value.total_board * 4);
		file_path = current_path + "/Data.csv";
		if(access(file_path.c_str(), F_OK ) == -1){
			SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		}

		ifstream infile_2(file_path, ios::in);

		getline(infile_2, line);
		
		for (int board_index = 0; board_index < recv_amp_balancing_value.total_board; board_index++)
		{
			getline(infile_2, line);
			stringstream ss(line);
			for (int freq_index = 0; freq_index < recv_amp_balancing_value.total_freq_point; freq_index++)
			{
				getline(ss, value, ',');
				*recv_amp_balancing_value(freq_index,board_index) = (float)atof(value.c_str());
			}
		}
		infile_2.close();

		current_path = getcwd(NULL, 0);
		current_path += "/config/" + params_map.find("Radar_name")->second + "/Recv_phase_compensate_data";

		recv_phase_balancing_value.total_freq_point = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		recv_phase_balancing_value.total_board = atoi(params_map.find("Antenna_board_total_num")->second.c_str());
		recv_phase_balancing_value.balancing_value = new float[recv_phase_balancing_value.total_freq_point * recv_phase_balancing_value.total_board];
		memset(recv_phase_balancing_value.balancing_value, 0, recv_phase_balancing_value.total_freq_point * recv_phase_balancing_value.total_board * 4);
		file_path = current_path + "/Data.csv";
		if(access(file_path.c_str(), F_OK ) == -1){
			SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		}

		ifstream infile_3(file_path, ios::in);

		getline(infile_3, line);
		
		for (int board_index = 0; board_index < recv_phase_balancing_value.total_board; board_index++)
		{
			getline(infile_3, line);
			stringstream ss(line);
			for (int freq_index = 0; freq_index < recv_phase_balancing_value.total_freq_point; freq_index++)
			{
				getline(ss, value, ',');
				*recv_phase_balancing_value(freq_index,board_index) = (float)atof(value.c_str());
			}
		}
		infile_3.close();

		return;
	}

	void rsl_radar_params_manager::load_recv_beam_calibrate_data(){
		if(recv_dbf_calibration_value.calibration_value != nullptr){
			delete[] recv_dbf_calibration_value.calibration_value;
		}
		if(recv_target_calibration_value.calibration_value != nullptr){
			delete[] recv_target_calibration_value.calibration_value;
		}

		std::string file_path = "";
		std::string current_path(getcwd(NULL, 0));
		current_path += "/config/" + params_map.find("Radar_name")->second + "/Recv_beam_calibrate_data";
		std::string line;
		std::string value;

		//接收指向修正数据
		recv_dbf_calibration_value.total_beams = atoi(params_map.find("Azimuth_total_beam_number")->second.c_str());
		recv_dbf_calibration_value.total_freq_point = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		recv_dbf_calibration_value.calibration_value = new float[recv_dbf_calibration_value.total_beams * recv_dbf_calibration_value.total_freq_point];
		memset(recv_dbf_calibration_value.calibration_value, 0, recv_dbf_calibration_value.total_freq_point * recv_dbf_calibration_value.total_beams * 4);

		file_path = current_path + "/DBF.csv";
		if(access(file_path.c_str(), F_OK ) == -1){
			SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		}

		ifstream infile_5(file_path, ios::in);

		getline(infile_5, line);

		for (int beam_index = 0; beam_index < recv_dbf_calibration_value.total_beams; beam_index++)
		{
			getline(infile_5, line);
			stringstream ss(line);
			for (int freq_index = 0; freq_index < recv_dbf_calibration_value.total_freq_point; freq_index++)
			{
				getline(ss, value, ',');
				*recv_dbf_calibration_value(freq_index,beam_index) = (float)atof(value.c_str());
			}
		}
		infile_5.close();

		recv_target_calibration_value.total_beams = atoi(params_map.find("Azimuth_total_beam_number")->second.c_str());
		recv_target_calibration_value.total_freq_point = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		recv_target_calibration_value.calibration_value = new float[recv_target_calibration_value.total_beams * recv_target_calibration_value.total_freq_point];
		memset(recv_target_calibration_value.calibration_value, 0, recv_target_calibration_value.total_freq_point * recv_target_calibration_value.total_beams * 4);

		file_path = current_path + "/Data.csv";
		if(access(file_path.c_str(), F_OK ) == -1){
			SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		}

		ifstream infile_6(file_path, ios::in);

		getline(infile_6, line);

		for (int beam_index = 0; beam_index < recv_target_calibration_value.total_beams; beam_index++)
		{
			getline(infile_6, line);
			stringstream ss(line);
			for (int freq_index = 0; freq_index < recv_target_calibration_value.total_freq_point; freq_index++)
			{
				getline(ss, value, ',');
				*recv_target_calibration_value(freq_index,beam_index) = (float)atof(value.c_str());
			}
		}
		infile_6.close();

		return;
	}

	void rsl_radar_params_manager::load_trans_beam_calibrate_data(){
		
		if(trans_beam_calibration_value.calibration_value != nullptr){
			delete[] trans_beam_calibration_value.calibration_value;
		}

		std::string file_path = "";
		std::string current_path(getcwd(NULL, 0));
		current_path += "/config/" + params_map.find("Radar_name")->second + "/Trans_beam_calibrate_data";
		std::string line;
		std::string value;

		//发射指向修正数据
		trans_beam_calibration_value.total_beams = atoi(params_map.find("Azimuth_total_beam_number")->second.c_str());
		trans_beam_calibration_value.total_freq_point = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		trans_beam_calibration_value.calibration_value = new float[trans_beam_calibration_value.total_beams * trans_beam_calibration_value.total_freq_point];
		memset(trans_beam_calibration_value.calibration_value, 0, trans_beam_calibration_value.total_freq_point * trans_beam_calibration_value.total_beams * 4);

		file_path = current_path + "/Data.csv";
		if(access(file_path.c_str(), F_OK ) == -1){
			SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		}

		ifstream infile_7(file_path, ios::in);

		getline(infile_7, line);

		for (int beam_index = 0; beam_index < trans_beam_calibration_value.total_beams; beam_index++)
		{
			getline(infile_7, line);
			stringstream ss(line);
			for (int freq_index = 0; freq_index < trans_beam_calibration_value.total_freq_point; freq_index++)
			{
				getline(ss, value, ',');
				*trans_beam_calibration_value(freq_index,beam_index) = (float)atof(value.c_str());
			}
		}
		infile_7.close();

		return;
	}

	void rsl_radar_params_manager::load_azimuth_angle_coeff(){
		
		if(azimuth_angle_coeff.coeff != nullptr){
			delete[] azimuth_angle_coeff.coeff;
		}

		std::string file_path = "";
		std::string current_path(getcwd(NULL, 0));
		current_path += "/config/" + params_map.find("Radar_name")->second + "/Azimuth_angle_coeff";
		std::string line;
		std::string value;

		//方位测角系数
#if AZI_COEFF_ORDER_3 == 0
		azimuth_angle_coeff.total_beams = atoi(params_map.find("Azimuth_total_beam_number")->second.c_str());
		azimuth_angle_coeff.freq_point_num = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		azimuth_angle_coeff.coeff = new float[azimuth_angle_coeff.total_beams * azimuth_angle_coeff.freq_point_num];
		memset(azimuth_angle_coeff.coeff, 0, azimuth_angle_coeff.total_beams * azimuth_angle_coeff.freq_point_num * 4);

		file_path = current_path + "/Data.csv";
		if(access(file_path.c_str(), F_OK ) == -1){
			SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		}

		ifstream infile_8(file_path, ios::in);
		
		getline(infile_8, line);

		for (int beam_index = 0; beam_index < azimuth_angle_coeff.total_beams; beam_index++)
		{
			getline(infile_8, line);
			stringstream ss(line);
			for (int freq_index = 0; freq_index < azimuth_angle_coeff.freq_point_num; freq_index++)
			{
				getline(ss, value, ',');
				*azimuth_angle_coeff(beam_index, freq_index) = - (float)atof(value.c_str());
			}
		}

		infile_8.close();
#else
		azimuth_angle_coeff.total_beams = atoi(params_map.find("Azimuth_total_beam_number")->second.c_str());
		azimuth_angle_coeff.freq_point_num = atoi(params_map.find("Freq_point_total_num")->second.c_str());
		azimuth_angle_coeff.coeff = new float[azimuth_angle_coeff.total_beams * azimuth_angle_coeff.freq_point_num * 3];
		memset(azimuth_angle_coeff.coeff, 0, azimuth_angle_coeff.total_beams * azimuth_angle_coeff.freq_point_num * 3 * 4);

		file_path = current_path + "/Data_order3.csv";
		if(access(file_path.c_str(), F_OK ) == -1){
			SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		}

		ifstream infile_8(file_path, ios::in);

		for (int beam_index = 0; beam_index < azimuth_angle_coeff.total_beams; beam_index++)
		{
			getline(infile_8, line);
			stringstream ss(line);
			for (int freq_index = 0; freq_index < azimuth_angle_coeff.freq_point_num; freq_index++)
			{
				float * tmp = azimuth_angle_coeff(beam_index, freq_index);
				getline(ss, value, ',');
				tmp[0] = - (float)atof(value.c_str());
				getline(ss, value, ',');
				tmp[1] = - (float)atof(value.c_str());
				getline(ss, value, ',');
				tmp[2] = - (float)atof(value.c_str());
			}
		}

		infile_8.close();
#endif

		return;
	}

	void rsl_radar_params_manager::load_pitch_angle_coeff(){
		
		if(pitch_angle_coeff.coeff != nullptr){
			delete[] pitch_angle_coeff.coeff;
		}

		std::string file_path = "";
		std::string current_path(getcwd(NULL, 0));
		current_path += "/config/" + params_map.find("Radar_name")->second + "/Pitch_angle_coeff";
		std::string line;
		std::string value;

		//俯仰测角系数
		pitch_angle_coeff.total_pitch_beams = atoi(params_map.find("Max_pitch_beam_number")->second.c_str()) - 1;
		pitch_angle_coeff.angle_range = atoi(params_map.find("Max_pitch_angle")->second.c_str()) - atoi(params_map.find("Min_pitch_angle")->second.c_str()) + 1;
		pitch_angle_coeff.max_order = 4;
		pitch_angle_coeff.min_angle = atoi(params_map.find("Min_pitch_angle")->second.c_str());
		
		pitch_angle_coeff.coeff = new float[pitch_angle_coeff.total_pitch_beams * pitch_angle_coeff.angle_range * pitch_angle_coeff.max_order];
		memset(pitch_angle_coeff.coeff, 0, pitch_angle_coeff.total_pitch_beams * pitch_angle_coeff.angle_range * pitch_angle_coeff.max_order * 4);

		for (int beam_index = 0; beam_index < pitch_angle_coeff.total_pitch_beams; beam_index++)
		{
			file_path = current_path + "/pit_angle_coeff_" + to_string(beam_index) + to_string(beam_index + 1) + ".csv";
			if(access(file_path.c_str(), F_OK ) == -1){
				SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
				continue;
			}
			
			ifstream infile_9(file_path, ios::in);

			for (int angle_index = 0; angle_index < pitch_angle_coeff.angle_range; angle_index++)
			{
				getline(infile_9, line);
				stringstream ss(line);
				for (int order_index = 0; order_index < pitch_angle_coeff.max_order; order_index++)
				{
					getline(ss, value, ',');
					*pitch_angle_coeff(beam_index,angle_index + pitch_angle_coeff.min_angle, order_index) = (float)atof(value.c_str());
				}
			}
			if(infile_9.is_open())
				infile_9.close();

		}

		return;
	}
	
	void rsl_radar_params_manager::load_STC_compensate_data(){
		
		for(auto coeff : stc_compensate_value.narrow_coeff){
			delete[] coeff;
		}
		stc_compensate_value.narrow_coeff.clear();

		for(auto coeff : stc_compensate_value.wide_coeff){
			delete[] coeff;
		}
		stc_compensate_value.wide_coeff.clear();

		std::string file_path = "";
		std::string current_path(getcwd(NULL, 0));
		current_path += "/config/" + params_map.find("Radar_name")->second + "/STC_compensate_data";
		std::string line;
		std::string value;

		//STC曲线补偿值
		//STC曲线补偿两种方案：1、目标点补偿；2、dbf后补偿
		//采用目标点补偿方案
		file_path = current_path + "/" + params_map.find("Default_range_mode")->second + "/STC_compensate_mtd.csv";
		if(access(file_path.c_str(), F_OK ) == -1){
			SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		}else{
			int pit_total_beam_num = atoi(params_map.find("Pitch_total_beam_number")->second.c_str());
			float dds_sample_rate = atof(params_map.find("DDC_sample_rate")->second.c_str());
			int point_index = 0;

			ifstream infile_10(file_path, ios::in);

			for (int pit_index = 0; pit_index < pit_total_beam_num; pit_index++)
			{
				int sample_point = atof(params_map.find("Pulse_repeat_period_" + to_string(pit_index))->second.c_str()) * dds_sample_rate;

				float * tmp = new float[sample_point];
				memset(tmp, 0, sample_point * sizeof(float));

				getline(infile_10, line);
				stringstream ss(line);
				for (point_index = 0; point_index < sample_point; point_index++)
				{
					value = "";
					getline(ss, value, ',');
					if(value == "")
						break;
					tmp[point_index] = (float)atof(value.c_str());
				}
				stc_compensate_value.wide_coeff.push_back(tmp);
			}
			
			for (int pit_index = 0; pit_index < pit_total_beam_num; pit_index++)
			{
				int sample_point = atof(params_map.find("Pulse_repeat_period_" + to_string(pit_index))->second.c_str()) * dds_sample_rate;

				float * tmp = new float[sample_point];
				memset(tmp, 0, sample_point * sizeof(float));

				getline(infile_10, line);
				stringstream ss(line);
				for (point_index = 0; point_index < sample_point; point_index++)
				{
					value = "";
					getline(ss, value, ',');
					if(value == "")
						break;
					tmp[point_index] = (float)atof(value.c_str());
				}
				stc_compensate_value.narrow_coeff.push_back(tmp);
			}

			infile_10.close();
		}

		// file_path = current_path + "/config/STC曲线补偿值/" + params_map.find("Default_range_mode")->second + "/STC_compensate_beam.csv";
		// if(access(file_path.c_str(), F_OK ) == -1){
		// 	SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		// }else{
		// 	int azi_total_beam_num = atoi(params_map.find("Azimuth_total_beam_number")->second.c_str());
		// 	int pit_total_beam_num = atoi(params_map.find("Pitch_total_beam_number")->second.c_str());
			
		// 	float stc_tmp[256];
		// 	int point_index = 0;

		// 	infile.open(file_path, ios::in);

		// 	for (int azi_index = 0; azi_index < azi_total_beam_num; azi_index++)
		// 	{
		// 		for (int pit_index = 0; pit_index < pit_total_beam_num; pit_index++)
		// 		{
		// 			if(stc_compensate_value.coeff == nullptr){
		// 				getline(infile, line);
		// 				stringstream ss(line);
		// 				for (point_index = 0; point_index < 256; point_index++)
		// 				{
		// 					value = "";
		// 					getline(ss, value, ',');
		// 					if(value == ""){
		// 						break;
		// 					}
		// 				}

		// 				stc_compensate_value.coeff = new float[azi_total_beam_num * pit_total_beam_num * point_index];
		// 				stc_compensate_value.azimuth_total_beam_num = azi_total_beam_num;
		// 				stc_compensate_value.pitch_total_beam_num = pit_total_beam_num;
		// 				stc_compensate_value.max_length = point_index;
		// 				infile.seekg(ios::beg);
		// 			}
		// 			getline(infile, line);
		// 			stringstream ss(line);
		// 			for (point_index = 0; point_index < stc_compensate_value.max_length; point_index++)
		// 			{
		// 				getline(ss, value, ',');
		// 				*stc_compensate_value(azi_index,pit_index,point_index) = (float)atof(value.c_str());
		// 			}
		// 		}
		// 	}
		// 	infile.close();
		// 	infile.clear();
		// }

		// float stc_tmp[256];
		// int point_index = 0;

		// file_path = current_path + "/config/STC曲线补偿值/STC预补偿/STC_compensate.csv";
		// if(access(file_path.c_str(), F_OK ) == -1){
		// 	SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "File doesn't exist!: {}", file_path);
		// }else{
		// 	infile.open(file_path, ios::in);
		// 	getline(infile, line);
		// 	stringstream ss(line);
		// 	for (; point_index < 256; point_index++)
		// 	{
		// 		getline(ss, value, ',');
		// 		if(value == ""){
		// 			break;
		// 		}else{
		// 			stc_tmp[point_index] = (float)atof(value.c_str());
		// 		}
		// 	}
		// 	stc_compensate_value.max_length = point_index;
		// 	stc_compensate_value.coeff = new float[point_index];
		// 	memcpy(stc_compensate_value.coeff, stc_tmp, point_index * 4);
		// 	infile.close();
		// 	infile.clear();
		// }

		return;
	}

	/**
	 * @brief     从校正数据文件中加载雷达校正数据
	 */
	void rsl_radar_params_manager::load_calibration_data(){

		load_recv_phase_data();
		load_trans_phase_data();
		load_recv_compensate_data();
		load_recv_beam_calibrate_data();
		load_trans_beam_calibrate_data();
		load_azimuth_angle_coeff();
		load_pitch_angle_coeff();
		load_STC_compensate_data();
		
		//初始化收发开关
		int board_num = atoi(params_map.find("Antenna_board_total_num")->second.c_str());
		int chn_num = atoi(params_map.find("Channel_num_of_Antenna_board")->second.c_str());

		t_switch.ant_switch = new uint16_t[board_num * chn_num];
		t_switch.channel_num_of_antenna_board = chn_num;

		r_switch.ant_switch = new uint16_t[board_num * chn_num];
		r_switch.channel_num_of_antenna_board = chn_num;

		if(atoi(params_map.find("Recv_switch")->second.c_str()) == 1){
			for (int board_index = 0; board_index < board_num; board_index++)
			{
				for (int chn_index = 0; chn_index < chn_num; chn_index++)
				{
					*r_switch(board_index,chn_index) = 1;
				}
			}
		}else{
			for (int board_index = 0; board_index < board_num; board_index++)
			{
				for (int chn_index = 0; chn_index < chn_num; chn_index++)
				{
					*r_switch(board_index, chn_index) = 0;
				}
			}
		}

		if(atoi(params_map.find("Trans_switch")->second.c_str()) == 1){
			for (int board_index = 0; board_index < board_num; board_index++)
			{
				for (int chn_index = 0; chn_index < chn_num; chn_index++)
				{
					*t_switch(board_index,chn_index) = 1;
				}
			}
		}else{
			for (int board_index = 0; board_index < board_num; board_index++)
			{
				for (int chn_index = 0; chn_index < chn_num; chn_index++)
				{
					*t_switch(board_index,chn_index) = 0;
				}  
			}
		}
	}

}