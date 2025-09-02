#include "rsl_rsp_data_save.hpp"

radar_signal_process::rsl_rsp_data_save::rsl_rsp_data_save()
{
	params_id = -1;
    params_update(0);

	thread_running = false;
}

void radar_signal_process::rsl_rsp_data_save::params_update(int curr_params_id)
{
	if(params_id != curr_params_id){
		rt_data_save_config_t config_tmp = rsl_radar_params_manager::getInstance().rt_data_save_config;
		process_data_save_enable = config_tmp.process_data_save_enable;
		DBF_data_save_enable = config_tmp.DBF_data_save_enable;
		PC_data_save_enable = config_tmp.PC_data_save_enable;
		MTD_data_save_enable = config_tmp.MTD_data_save_enable;
		target_data_save_enable = config_tmp.target_data_save_enable;

		target_filter_enable = config_tmp.target_filter_enable;
		min_range = config_tmp.min_range;
		max_range = config_tmp.max_range;
		min_velocity = config_tmp.min_velocity;
		max_velocity = config_tmp.max_velocity;
		min_azimuth = config_tmp.min_azimuth;
		max_azimuth = config_tmp.max_azimuth;
		min_pitch = config_tmp.min_pitch;
		max_pitch = config_tmp.max_pitch;
		int target_cnt = 0;

		sector_beam_cnt = INT_PARAMS("Azimuth_total_beam_number") / INT_PARAMS("Azimuth_beam_number");

		if(process_data_save_enable == 1 || target_data_save_enable == 1){
			string time_str = STRING_PARAMS("Data_save_file_path") + "/" + get_time_string();

			mkdir(time_str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

			data_path = time_str + "/Data";
			target_path = time_str + "/Target";

			mkdir(data_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			mkdir(target_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		}

		if(target_data_save_enable != 1 && tf_outfile.is_open()){
			tf_outfile.close();
			target_cnt = 0;
		}

		if(process_data_save_enable == 1){
			start();
		}

		params_id = curr_params_id;
	}
}

radar_signal_process::rsl_rsp_data_save &radar_signal_process::rsl_rsp_data_save::getInstance()
{
	static rsl_rsp_data_save instance;
    return instance;
}

radar_signal_process::rsl_rsp_data_save::~rsl_rsp_data_save()
{
	exit_flag = true;
	if(tf_outfile.is_open()){
		tf_outfile.close();
	}
}

void radar_signal_process::rsl_rsp_data_save::start()
{
	if(!thread_running){
		exit_flag = false;
		std::thread thread_data_save(&rsl_rsp_data_save::fcn_save, this);
		thread_data_save.detach();
		thread_running = true;
	}
}

void radar_signal_process::rsl_rsp_data_save::stop()
{
	if(thread_running){
		exit_flag = true;
		thread_running = false;
	}
}

void radar_signal_process::rsl_rsp_data_save::fcn_save()
{
	while(!exit_flag){

		data_save_params params = data_quque.pop();

		int fd = -1;
		if(access(params.file_path.c_str(), F_OK) != 0){
			fd = open(params.file_path.c_str(), O_RDWR | O_DIRECT | O_CREAT, 0666);
		}else{
			fd = open(params.file_path.c_str(), O_RDWR | O_DIRECT | O_CREAT, 0666);
		}
		if (fd == -1)
		{
			printf("Error: Failed to open file, %d\n",fd);
			free(params.ram_ptr);
			return;
		}

		int ret = ftruncate(fd, params.block_size);
		ssize_t ret_size = pwrite64(fd, params.ram_ptr, params.block_size, 0);

		close(fd);
		free(params.ram_ptr);
	}
	
	while(!data_quque.empty()){
		data_save_params params = data_quque.pop();
		free(params.ram_ptr);
	}

	exit_flag = false;

	return;
}

void radar_signal_process::rsl_rsp_data_save::save(cpi_data_t &cpi_data, dbf_t &dbf_data, pc_t &pc_data, mtd_abs_t &mtd_data, int azimuth_beam_num, int pitch_beam_num)
{
	params_update(cpi_data.cmd_params->work_mode.params_id);

	if(process_data_save_enable != 1){
		return;
	}

	//参数
	int prf_total_pulse = cpi_data.cmd_params->pulse_params.prf_total_pulse;
	int narrow_points = cpi_data.cmd_params->pulse_params.narrow_pulse_valid_point/8;
	int wide_points = cpi_data.cmd_params->pulse_params.wide_pulse_valid_point/8;

	int mtd_narrow_points = mtd_data.narrow_points;
	int mtd_wide_points = mtd_data.wide_points;
	int mtd_prf_total_pulse = mtd_data.prf_total_pulse;

	//计算数据总量
	size_t block_size = 0;
	block_size += CMD_PARAMS_LEN;
	if(DBF_data_save_enable == 1){
		block_size += sizeof(data_save_info_t);
		block_size += prf_total_pulse * (narrow_points + wide_points) * 2 * 8;
	}

	if(PC_data_save_enable == 1){
		block_size += sizeof(data_save_info_t);
		block_size += prf_total_pulse * (narrow_points + wide_points) * 2 * 8;
	}
	
	if(MTD_data_save_enable == 1){
		block_size += sizeof(data_save_info_t);
		block_size += mtd_prf_total_pulse * (mtd_narrow_points + mtd_wide_points) * 4;
	}
	block_size = ((block_size / 4096) + 1) * 4096;

	//初始化RAM空间
	char *ram_ptr = nullptr;
	int retval = posix_memalign((void **)&ram_ptr, 4096, block_size);
	if(retval < 0){
		SPDLOG_LOGGER_ERROR(rsl_debug_tools::debug_tools::_logger, "posix_memalign: Invalid pointer %d",retval);
		return;
	}
	char * tmp = ram_ptr;

	//生成文件名
	string file_name = to_string(cpi_data.cmd_params->pulse_params.cpi_cnt) + "_" + to_string(azimuth_beam_num / sector_beam_cnt);
	string file_path = data_path + "/" + file_name;

	//保存命令参数
	memcpy(tmp, (void *)cpi_data.cmd_params, CMD_PARAMS_LEN);
	tmp += CMD_PARAMS_LEN;

	//保存DBF数据
	if(DBF_data_save_enable == 1){
		data_save_info_t info{
			.head = (int)0xf010f010,
			.data_cate = DBF,
			.azimuth_sector_num = azimuth_beam_num / sector_beam_cnt,
			.azimuth_beam_num = azimuth_beam_num,
			.pitch_beam_num = pitch_beam_num,
			.sum_diff = SUM,
			.narrow_wide = NARROW,
			.row = prf_total_pulse,
			.col = narrow_points,
			.data_type = rsl_debug_tools::type_complex_float
		};
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)dbf_data.narrow_sum[0], get_bytes_length(info));
		tmp += get_bytes_length(info);
		
		info.sum_diff = DIFF;
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)dbf_data.narrow_diff[0], get_bytes_length(info));
		tmp += get_bytes_length(info);

		info.narrow_wide = WIDE;
		info.col = wide_points;
		info.sum_diff = SUM;
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)dbf_data.wide_sum[0], get_bytes_length(info));
		tmp += get_bytes_length(info);

		info.sum_diff = DIFF;
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)dbf_data.wide_diff[0], get_bytes_length(info));
		tmp += get_bytes_length(info);
	}
	
	if(PC_data_save_enable == 1){
		data_save_info_t info{
			.head = (int)0xf010f010,
			.data_cate = PC,
			.azimuth_sector_num = azimuth_beam_num / sector_beam_cnt,
			.azimuth_beam_num = azimuth_beam_num,
			.pitch_beam_num = pitch_beam_num,
			.sum_diff = SUM,
			.narrow_wide = NARROW,
			.row = narrow_points,
			.col = prf_total_pulse,
			.data_type = rsl_debug_tools::type_complex_float
		};
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)pc_data.narrow_sum[0], get_bytes_length(info));
		tmp += get_bytes_length(info);
		
		info.sum_diff = DIFF;
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)pc_data.narrow_diff[0], get_bytes_length(info));
		tmp += get_bytes_length(info);

		info.narrow_wide = WIDE;
		info.row = wide_points;
		info.sum_diff = SUM;
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)pc_data.wide_sum[0], get_bytes_length(info));
		tmp += get_bytes_length(info);

		info.sum_diff = DIFF;
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)pc_data.wide_diff[0], get_bytes_length(info));
		tmp += get_bytes_length(info);
	}

	if(MTD_data_save_enable == 1){
		data_save_info_t info{
			.head = (int)0xf010f010,
			.data_cate = MTD,
			.azimuth_sector_num = azimuth_beam_num / sector_beam_cnt,
			.azimuth_beam_num = azimuth_beam_num,
			.pitch_beam_num = pitch_beam_num,
			.sum_diff = SUM,
			.narrow_wide = NARROW,
			.row = mtd_narrow_points,
			.col = mtd_prf_total_pulse,
			.data_type = rsl_debug_tools::type_float
		};
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)mtd_data.narrow_sum[0], get_bytes_length(info));
		tmp += get_bytes_length(info);

		info.narrow_wide = WIDE;
		info.row = mtd_wide_points;
		info.sum_diff = SUM;
		memcpy(tmp, (void *)&info, sizeof(data_save_info_t));
		tmp += sizeof(data_save_info_t);
		memcpy(tmp, (void *)mtd_data.wide_sum[0], get_bytes_length(info));
		tmp += get_bytes_length(info);
	}

	data_save_params params_tmp;
	params_tmp.block_size = block_size;
	params_tmp.file_path = file_path;
	params_tmp.ram_ptr = ram_ptr;

	data_quque.push(params_tmp);

	//SPDLOG_LOGGER_INFO(rsl_debug_tools::debug_tools::_logger, "data_quque size: {}", data_quque.size());

	return;
}

void radar_signal_process::rsl_rsp_data_save::save_target(cpi_data_t &cpi_data, target_map_t target)
{
	if(target_data_save_enable != 1){
		return;
	}

	if(!tf_outfile.is_open()){
		tf_outfile.open(target_path + "/target", ios::binary|ios::out|ios::trunc);
		//写入头部信息
		target_cnt = 0;
		tf_outfile.seekp(0,tf_outfile.beg);
		int head[4] = {target_cnt, sizeof(target_t::reserved), 0, 0};
		tf_outfile.write((char *)head,16);
	}

	tf_outfile.seekp(sizeof(radar_signal_process::target_t) * target_cnt + 16, tf_outfile.beg);
	radar_signal_process::target_map_t::iterator iter = target.begin();
	for (;iter != target.end();iter ++)
	{
		if(target_filter_enable == 1){
			if(	iter->second.r_cohesion_val > min_range && \
				iter->second.r_cohesion_val < max_range && \
				iter->second.v_cohesion_val > min_velocity && \
				iter->second.v_cohesion_val < max_velocity && \
				iter->second.pitch > min_pitch && \
				iter->second.pitch < max_pitch){

				float azi_tmp1 = iter->second.azimuth - min_azimuth;
				float azi_tmp2 = iter->second.azimuth - max_azimuth;

				if( (fabsf(azi_tmp1) < fabsf(azi_tmp2) && azi_tmp1 > 0) || (fabsf(azi_tmp2) < fabsf(azi_tmp1) && azi_tmp2 < 0) ){
					//保存目标信息
					tf_outfile.write((char *)&iter->second,sizeof(radar_signal_process::target_t));
					target_cnt++;
				}
			}
		}else{
			//保存目标信息
			tf_outfile.write((char *)&iter->second,sizeof(radar_signal_process::target_t));
			target_cnt++;
		}
	}
	
	//写入头部信息
	tf_outfile.seekp(0,tf_outfile.beg);
	int head[4] = {target_cnt, sizeof(target_t::reserved), 0, 0};
	tf_outfile.write((char *)head,16);
	tf_outfile.flush();
}

size_t radar_signal_process::rsl_rsp_data_save::get_bytes_length(data_save_info_t info)
{
	size_t len = info.col*info.row;
	switch (info.data_type)
	{
	case rsl_debug_tools::type_int16:
		len*=2;
		break;
	case rsl_debug_tools::type_int:
	case rsl_debug_tools::type_float:
		len*=4;
		break;
	case rsl_debug_tools::type_long:
	case rsl_debug_tools::type_double:
	case rsl_debug_tools::type_complex_float:
		len*=8;
		break;
	case rsl_debug_tools::type_complex_double:
		len*=16;
		break;
	default:
		break;
	}
	return len;
}

string radar_signal_process::rsl_rsp_data_save::get_time_string()
{
	time_t t = time(0); 
	char tmp[32]={'\0'};
	strftime(tmp, sizeof(tmp), "%Y%m%d_%H%M",localtime(&t)); 
	return string(tmp);
}


