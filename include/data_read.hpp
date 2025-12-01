#pragma once

#include <common.cuh>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <string.h>
#include "rsl_data_structure.hpp"
#include "rsl_radar_params_manager.hpp"
#include "cuda_dbf.cuh"
#include "cufft.h"
using namespace::radar_signal_process;



//==================函数声明===============================//
int data_read(streampos &current_pos , uint32_t* cpi_data_begin);
uint32_t * data_read(const std::string &filename, streampos &current_pos);
void ddc_data_save_to_csv(ddc_data_t* ddc_data, const std::string& filename);
void writeDBFDataToCSV(const cmd_params_t& cmdParams, const std::string& filename);
void beam_saveToCSV(const MapDDC_beam& beam_narrow_sum, int prf_total_pulse, int points, const std::string& filename);
void cufftComplex_save_to_csv(const std::string& filename, cufftComplex* d_data, size_t size, int row, int col);
void float_save_to_csv(const std::string& filename, float* d_data, size_t size, int row, int col);
