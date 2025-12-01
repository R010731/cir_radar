#include "data_read.hpp"

//=================函数实现===============================//
int data_read(streampos &current_pos, uint32_t* cpi_data_begin){
    // 1. 打开文件
    string file_path = "/media/ryh/新加卷/ryh/D/0";
    std::ifstream in_file(file_path, std::ios::binary);
    if (!in_file) {
        cerr << "文件打开失败" << endl;
        return -1;
    }
    
    //2.读取一个CPI数据
    if(!in_file.eof()){
        
        //读取命令参数帧头
        uint32_t tmp = 0;
        int error_cnt = -1;
        while(tmp != 0xa5a5a5a5){
            in_file.read((char *)&tmp,4);
            error_cnt++;
        }

        //读取命令参数包
        cmd_params_t cmd_params_tmp;
        in_file.seekg(streampos(in_file.tellg() - streamoff(4)));
        in_file.read((char *)&cmd_params_tmp, CMD_PARAMS_LEN);

        //计算CPI数据量，分配内存空间
        int cpi_data_len = CPI_DATA_LEN(cmd_params_tmp.pulse_params.prf_total_pulse,cmd_params_tmp.pulse_params.narrow_pulse_valid_point / 8,cmd_params_tmp.pulse_params.wide_pulse_valid_point / 8);
        // int retval = posix_memalign((void **)&ram_ptr, 64, cpi_data_len);
        ErrorCheck(cudaMallocHost((void **)&cpi_data_begin, cpi_data_len), __FILE__, __LINE__);

        //读取一个CPI的数据
        in_file.seekg(streampos(in_file.tellg() - streamoff(CMD_PARAMS_LEN)));
        in_file.read((char *)cpi_data_begin, cpi_data_len);

        return 1;
    }
    // 关闭文件
    else{
        in_file.close();
        // 关闭文件
        return 0;
    }
}

uint32_t * data_read(const std::string &filename, streampos &current_pos){
    // 1. 打开文件
    std::ifstream in_file(filename, std::ios::binary);
    if (!in_file) {
        cerr << "文件打开失败" << endl;
        return nullptr;
    }
    

    //2.读取一个CPI数据
    if(!in_file.eof()){
        // 将文件指针移动到之前保存的位置
        in_file.seekg(current_pos);
        //读取命令参数帧头
        uint32_t tmp = 0;
        int error_cnt = -1;
        while(tmp != 0xa5a5a5a5){
            in_file.read((char *)&tmp,4);
            error_cnt++;
        }

        //读取命令参数包
        cmd_params_t cmd_params_tmp;
        in_file.seekg(streampos(in_file.tellg() - streamoff(4)));
        in_file.read((char *)&cmd_params_tmp, CMD_PARAMS_LEN);

        //计算CPI数据量，分配内存空间
        int cpi_data_len = CPI_DATA_LEN(cmd_params_tmp.pulse_params.prf_total_pulse,cmd_params_tmp.pulse_params.narrow_pulse_valid_point / 8,cmd_params_tmp.pulse_params.wide_pulse_valid_point / 8);
        uint32_t *ram_ptr;
        // int retval = posix_memalign((void **)&ram_ptr, 64, cpi_data_len);
        ErrorCheck(cudaMallocHost((void **)&ram_ptr, cpi_data_len), __FILE__, __LINE__);

        //读取一个CPI的数据
        in_file.seekg(streampos(in_file.tellg() - streamoff(CMD_PARAMS_LEN)));
        in_file.read((char *)ram_ptr, cpi_data_len);
        
        // 记录这次读取文件的偏移量
        current_pos = in_file.tellg();
        return ram_ptr;
    }
    // 关闭文件
    else{
        in_file.close();
        // 关闭文件
        return nullptr;
    }
}



// ddc数据保存
void ddc_data_save_to_csv(ddc_data_t* ddc_data, const std::string& filename) {
    std::ofstream out_file(filename);
    
    // Check if file is open
    if (!out_file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }

    // Write CSV header
    out_file << "prf_index,pulse_type,chn_index,real_part,imaginary_part\n";

    // Iterate through prf data
    for (int i = 0; i < MAX_PRF_PULSE; ++i) {
        // Iterate through narrow and wide pulse types
        for (int pulse_type = 0; pulse_type < 2; ++pulse_type) {
            pulse_data_t* pulse = (pulse_type == 0) ? &ddc_data->prf[i].narrow : &ddc_data->prf[i].wide;
            
            // Iterate through channels (chn[36])
            for (int chn_index = 0; chn_index < 36; ++chn_index) {
                if (pulse->chn[chn_index] != nullptr) {
                    // Iterate through complex numbers in the channel (assuming complex numbers are stored as an array)
                    for (int j = 0; j < 36; ++j) {
                        std::complex<float> value = pulse->chn[chn_index][j];
                        // Write each entry in the CSV
                        out_file << i << ","
                                 << (pulse_type == 0 ? "narrow" : "wide") << ","
                                 << chn_index << ","
                                 << value.real() << ","
                                 << value.imag() << "\n";
                    }
                }
            }
        }
    }

    out_file.close();
    std::cout << "Data saved to " << filename << std::endl;
}

// DBF数据保存
void writeDBFDataToCSV(cpi_data_t& cpi_data, const std::string& filename) {
    // 创建并打开 CSV 文件
    std::ofstream file(filename);

    if (file.is_open()) {
        // 写入 DBF_group 数据
        file << "DBF_group Data:\n";
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 20; j++) {
                file << (int)cpi_data.cmd_params->DBF_group[i][j];
                if (j < 19) file << ", "; // 添加逗号分隔符
            }
            file << "\n";
        }

        file << "\nDBF_coeff Data:\n";
        // 写入 DBF_coeff 数据
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 36; k++) {
                    file << cpi_data.cmd_params->DBF_coeff[i][j][k];
                    if (k < 35) file << ", "; // 添加逗号分隔符
                }
                file << "\n";
            }
        }

        file.close();  // 关闭文件
        std::cout << "Data has been written to " << filename << "\n";
    } else {
        std::cerr << "Unable to open the file.\n";
    }
}

void beam_saveToCSV(const MapDDC_beam& beam_narrow_sum, int prf_total_pulse, int points, const std::string& filename) {
    std::ofstream file(filename);

    // 检查文件是否打开成功
    if (!file.is_open()) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return;
    }

    // 遍历矩阵，将数据写入文件
    for (int i = 0; i < prf_total_pulse; ++i) {
        for (int j = 0; j < points; ++j) {
            // 获取复数的实部和虚部
            auto data = beam_narrow_sum(i, j);
            file << data.real() << "," << data.imag();  // 写入实部和虚部
            if (j != points - 1) {
                file << ",";  // 如果不是最后一列，添加逗号
            }
        }
        file << "\n";  // 每行结束
    }

    file.close();
    std::cout << "数据已保存到 " << filename << std::endl;
}

void cufftComplex_save_to_csv(const std::string& filename, cufftComplex* d_data, size_t size, int row, int col) {
    // 1. 从设备复制数据到主机
    cufftComplex* h_data = (cufftComplex*)malloc(size * sizeof(cufftComplex));
    cudaError_t err = cudaMemcpy(h_data, d_data, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        free(h_data);
        return;
    }

    // 2. 打开CSV文件以写入
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        free(h_data);
        return;
    }

    // 3. 遍历数据，将其写入CSV文件
    int count = 0;
    for (size_t i = 0; i < size; ++i) {
        // 获取复数的实部和虚部
        file << h_data[i].x << "," << h_data[i].y;  // 写入实部和虚部

        count++;
        
        // 每行写入 col 个数据
        if (count % col == 0) {
            file << "\n";  // 换行
        } else {
            file << ",";  // 添加逗号分隔下一个数据
        }

        // 如果到达数据的最后一项并且没有换行，补充换行符
        if (i == size - 1 && count % col != 0) {
            file << "\n";
        }
    }

    // 4. 关闭文件
    file.close();
    std::cout << "数据已保存到 " << filename << std::endl;

    // 5. 释放主机内存
    free(h_data);
}

void float_save_to_csv(const std::string& filename, float* d_data, size_t size, int row, int col) {
    // 1. 从设备复制数据到主机
    float* h_data = (float*)malloc(size * sizeof(float));
    cudaError_t err = cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        free(h_data);
        return;
    }

    // 2. 打开CSV文件以写入
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        free(h_data);
        return;
    }

    // 3. 遍历数据，将其写入CSV文件
    int count = 0;
    for (size_t i = 0; i < size; ++i) {
        file << h_data[i];  // 写入 float 数据

        count++;
        
        // 每行写入 col 个数据
        if (count % col == 0) {
            file << "\n";  // 换行
        } else {
            file << ",";  // 添加逗号分隔下一个数据
        }

        // 如果到达数据的最后一项并且没有换行，补充换行符
        if (i == size - 1 && count % col != 0) {
            file << "\n";
        }
    }

    // 4. 关闭文件
    file.close();
    std::cout << "数据已保存到 " << filename << std::endl;

    // 5. 释放主机内存
    free(h_data);
}
