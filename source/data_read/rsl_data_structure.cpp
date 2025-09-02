#include <iostream>
#include <string>
#include <fstream>
#include "rsl_data_structure.hpp"

namespace radar_signal_process{
	cpi_data_t::cpi_data_t()
	{
        this->cpi_data_ptr = nullptr;
        this->cmd_params = nullptr;
        this->ddc_data = nullptr;
        this->jamming_data = nullptr;
        this->noise_data = nullptr;
        this->self_test = nullptr;
        this->status_info = nullptr;
	}

	cpi_data_t::cpi_data_t(const cpi_data_t &data)
	{
        this->cpi_data_ptr = data.cpi_data_ptr;
        this->cpis_ram = data.cpis_ram;
        this->cmd_params = data.cmd_params;
        this->ddc_data = data.ddc_data;
        this->jamming_data = data.jamming_data;
        this->noise_data = data.noise_data;
        this->self_test = data.self_test;
        this->status_info = data.status_info;
        this->gnss_msg = data.gnss_msg;
	}

	cpi_data_t::cpi_data_t(uint32_t* data_ptr)
    {
        cpi_data_ptr = data_ptr;

        // printf("%x\n", ram_ptr);

        //构建CPI数据
        uint32_t * ram_ptr_tmp;
        ram_ptr_tmp = cpi_data_ptr;
        cmd_params = (cmd_params_t *)ram_ptr_tmp;
        ram_ptr_tmp += CMD_PARAMS_LEN / 4;

#if STATUS_INFO_VALID == 1
        status_info = (status_info_t *)ram_ptr_tmp;
        ram_ptr_tmp += STATUS_INFO_LEN / 4;
#endif

        self_test = new self_test_t;
        self_test->data = ram_ptr_tmp;
        ram_ptr_tmp += SELF_TEST_LEN / 4;

        ddc_data = new ddc_data_t;
        for (int prf_index = 0; prf_index < cmd_params->pulse_params.prf_total_pulse; prf_index++)
        {

            for (int chn_index = 0; chn_index < 36; chn_index++)
            {
                ddc_data->prf[prf_index].narrow.chn[chn_index] = (complex<float> *)ram_ptr_tmp;
                ram_ptr_tmp += cmd_params->pulse_params.narrow_pulse_valid_point / 8 * 2;
            }
            
            for (int chn_index = 0; chn_index < 36; chn_index++)
            {
                ddc_data->prf[prf_index].wide.chn[chn_index] = (complex<float> *)ram_ptr_tmp;
                ram_ptr_tmp += cmd_params->pulse_params.wide_pulse_valid_point / 8 * 2;
            }
        }

        jamming_data = new jamming_data_t;
        for (int chn_index = 0; chn_index < 36; chn_index++)
        {
            jamming_data->narrow.chn[chn_index] = (complex<float> *)ram_ptr_tmp;
            ram_ptr_tmp += cmd_params->pulse_params.narrow_pulse_valid_point / 8 * 2;
        }
        for (int chn_index = 0; chn_index < 36; chn_index++)
        {
            jamming_data->wide.chn[chn_index] = (complex<float> *)ram_ptr_tmp;
            ram_ptr_tmp += cmd_params->pulse_params.wide_pulse_valid_point / 8 * 2;
        }

        noise_data = new noise_data_t;
        noise_data->data = ram_ptr_tmp;

    }
    
    cpi_data_t::~cpi_data_t()
    {

    }

    void cpi_data_t::dispose(){
        // if(cpi_data_ptr != nullptr){
        //     free(cpi_data_ptr);
        // }

        if(self_test != nullptr){
            delete self_test;
        }

        if(ddc_data != nullptr){
            delete ddc_data;
        }

        if(jamming_data != nullptr){
            delete jamming_data;
        }

        if(noise_data != nullptr){
            delete noise_data;
        }
    }

	cpis_ram_t::cpis_ram_t()
	{
	}

	cpis_ram_t::cpis_ram_t(char* ptr, size_t s)
	{
        ram_ptr = ptr;
        size = s;
	}

	cpis_ram_t::~cpis_ram_t()
	{
        free(ram_ptr);
	}
}