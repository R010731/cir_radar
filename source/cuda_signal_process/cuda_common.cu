#include <stdlib.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <time.h>
#include "common.cuh"
#include <iostream>

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
            error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

// 错误检查函数
void checkCufftError(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT error: ");
        switch (result) {
        case CUFFT_INVALID_PLAN:
            fprintf(stderr, "Invalid plan\n");
            break;
        case CUFFT_ALLOC_FAILED:
            fprintf(stderr, "Memory allocation failed\n");
            break;
        case CUFFT_INVALID_TYPE:
            fprintf(stderr, "Invalid type\n");
            break;
        case CUFFT_INVALID_VALUE:
            fprintf(stderr, "Invalid value\n");
            break;
        case CUFFT_INTERNAL_ERROR:
            fprintf(stderr, "Internal error\n");
            break;
        case CUFFT_EXEC_FAILED:
            fprintf(stderr, "Execution failed\n");
            break;
        case CUFFT_SETUP_FAILED:
            fprintf(stderr, "Setup failed\n");
            break;
        case CUFFT_INVALID_SIZE:
            fprintf(stderr, "Invalid size\n");
            break;
        default:
            fprintf(stderr, "Unknown error code: %d\n", result);
            break;
        }
        // exit(EXIT_FAILURE);  // 出现错误时终止程序
    }
}



void setGPU()
{
    // 检测计算机GPU数量
    int iDeviceCount = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount), __FILE__, __LINE__);

    if (error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else
    {
        //printf("Number of CUDA devices: %d\n", iDeviceCount);
        cudaDeviceProp device_prop;
        // 获取指定设备的属性
        for (int i = 0; i < iDeviceCount; i++){
            cudaGetDeviceProperties(&device_prop, i);
            // 输出 GPU 型号
            printf("Name: %s ", device_prop.name);
            // printf("Total global memory: %lu MB\n", device_prop.totalGlobalMem / (1024 * 1024));
            // printf("Memory clock rate: %d MHz\n", device_prop.memoryClockRate / 1000);
            // printf("Memory bus width: %d bits\n", device_prop.memoryBusWidth);
            // printf("Shared memory per block: %lu KB\n", device_prop.sharedMemPerBlock / 1024);
            // printf("Max threads per block: %d\n", device_prop.maxThreadsPerBlock);
            // printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        }
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("free_mem:%ld MB,total_mem:%ld MB\n",free_mem/1024/1024,total_mem/1024/1024);
    }

    // 设置执行
    int iDev = 0;
    error = ErrorCheck(cudaSetDevice(iDev), __FILE__, __LINE__);
    if (error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for computing.\n");
    }

}



float pi = 3.14159;
// CUDA核函数，用于执行滤波操作
__global__ void filter_kernel(cufftComplex* input, int input_length, float* filter, int filter_length, cufftComplex* output) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;  // 当前线程处理的索引

    // 确保线程不越界
    if (n < input_length) {
        cufftComplex result;
        result.x = 0.00000f;
        result.y = 0.00000f;
        // 对输入数据应用滤波器
        for (int k = 0; k < filter_length; k++) {
            if (n - k >= 0) { // 确保不越界
                result.x += filter[k] * input[n - k].x;
                result.y += filter[k] * input[n - k].y;
            }
        }
        output[n].x = result.x;
        output[n].y = result.y;
    }
}


// 抽取（降采样）

__global__ void decimate_kernel(cufftComplex* input, cufftComplex* output, int input_len, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程的全局索引
    // 每个线程处理一个采样点
    if (idx * M < input_len) {
        int input_idx = idx * M;
        output[idx].x = input[input_idx].x;
        output[idx].y = input[input_idx].y;
    }
}


// GPU复数乘法
// 因为CUDA中IFFT没有像matlab中一样归一化，所以要除以SampleNumber（一个脉冲周期的采样点数）
__global__ void complexMultiply(const cuFloatComplex* a, const cuFloatComplex* b, cuFloatComplex* result, int numSegments, int SampleNumber) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSegments) {
        result[idx] = cuCmulf(a[idx], b[idx % SampleNumber]);
        result[idx].x = result[idx].x/SampleNumber;
        result[idx].y = result[idx].y/SampleNumber;          
    }
}

__global__ void complexMultiplyfloat(const cuFloatComplex* a, const float* b, cuFloatComplex* result, int numSegments, int SampleNumber) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSegments) {
       int bIndex = idx / SampleNumber;  // 计算访问b数组的索引
        result[idx].x = a[idx].x * b[bIndex];
        result[idx].y = a[idx].y * b[bIndex];
    }
}

// gpu内核复数相加, N为操作的数量
__global__ void complexAdd_Kernel(cuComplex* A, const cuComplex* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < N) {
        cuComplex temp = cuCaddf(A[idx], B[idx]);  // 复数加法
        A[idx] = temp;  // 将结果写回 A
    }
}

// GPU DDC float乘法
// a:输入波形 b:本地I路信号 c:本地Q路信号 d:DDC后复数信号
__global__ void multiplyArrays(float* a, float* b, float* c, cufftComplex* d, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // 计算每个线程的索引
    if (idx < size) {  // 确保索引不越界
        d[idx].x = a[idx] * b[idx];  // 计算元素乘积
        d[idx].y = a[idx] * c[idx];
    }
}

// GPU 复数取平方
__global__ void abs_kernel(cufftComplex* idata1, float* odata, int num)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num) {
        //odata[tid] = double(sqrt(pow(idata1[tid].x, 2) + pow(idata1[tid].y, 2)));
        odata[tid] = cuCabsf(idata1[tid]);
    }
}

// CA_CFAR
__global__ void CA_CFAR_kernel(float* Sig, int Range_gate_length, int Velocity_gate_length,
    int Num_range_gate_ref, int Num_velocity_gate_ref,
    int Num_range_gate_protect, int Num_velocity_gate_protect,
    float Threshold, float* CFAR_matrix, int* Flag_matrix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < Velocity_gate_length && j < Range_gate_length) {
        // Initialize CFAR matrix
        CFAR_matrix[i * Range_gate_length + j] = 0;
        Flag_matrix[i * Range_gate_length + j] = 0;
        // Range dimension CFAR
        if (j < Num_range_gate_ref / 2 + Num_range_gate_protect) {
            double ref_sum = 0;
            for (int k = j + Num_range_gate_protect + 1; k < j + Num_range_gate_protect + Num_range_gate_ref / 2; k++) {
                ref_sum += Sig[i * Range_gate_length + k];
            }
            if (Sig[i * Range_gate_length + j] > Threshold * (ref_sum / (Num_range_gate_ref / 2))) {
                CFAR_matrix[i * Range_gate_length + j] = Sig[i * Range_gate_length + j];
                Flag_matrix[i * Range_gate_length + j] = 1;
            }
        }
        else if (j >= Num_range_gate_ref / 2 + Num_range_gate_protect &&
            j < Range_gate_length - Num_range_gate_ref / 2 - Num_range_gate_protect) {
            double ref_sum = 0;
            for (int k = j - Num_range_gate_ref / 2 - Num_range_gate_protect; k < j - Num_range_gate_protect; k++) {
                ref_sum += Sig[i * Range_gate_length + k];
            }
            for (int k = j + Num_range_gate_protect + 1; k < j + Num_range_gate_ref / 2 + Num_range_gate_protect; k++) {
                ref_sum += Sig[i * Range_gate_length + k];
            }
            if (Sig[i * Range_gate_length + j] > Threshold * (ref_sum / Num_range_gate_ref)) {
                CFAR_matrix[i * Range_gate_length + j] = Sig[i * Range_gate_length + j];
                Flag_matrix[i * Range_gate_length + j] = 1;
            }
        }
        else {
            double ref_sum = 0;
            for (int k = j - Num_range_gate_ref / 2 - Num_range_gate_protect; k < j - Num_range_gate_protect; k++) {
                ref_sum += Sig[i * Range_gate_length + k];
            }
            if (Sig[i * Range_gate_length + j] > Threshold * (ref_sum / (Num_range_gate_ref / 2))) {
                CFAR_matrix[i * Range_gate_length + j] = Sig[i * Range_gate_length + j];
                Flag_matrix[i * Range_gate_length + j] = 1;
            }
        }

        // Velocity dimension CFAR
        if (i < Num_velocity_gate_ref / 2 + Num_velocity_gate_protect) {
            double ref_sum = 0;
            for (int k = i + Num_velocity_gate_protect + 1; k < i + Num_velocity_gate_ref / 2 + Num_velocity_gate_protect; k++) {
                ref_sum += Sig[k * Range_gate_length + j];
            }
            if (Sig[i * Range_gate_length + j] > Threshold * (ref_sum / (Num_velocity_gate_ref / 2))) {
                CFAR_matrix[i * Range_gate_length + j] = Sig[i * Range_gate_length + j];
                Flag_matrix[i * Range_gate_length + j] = 1;
            }
        }
        else if (i >= Num_velocity_gate_ref / 2 + Num_velocity_gate_protect &&
            i < Velocity_gate_length - Num_velocity_gate_ref / 2 - Num_velocity_gate_protect) {
            double ref_sum = 0;
            for (int k = i - Num_velocity_gate_ref / 2 - Num_velocity_gate_protect; k < i - Num_velocity_gate_protect; k++) {
                ref_sum += Sig[k * Range_gate_length + j];
            }
            for (int k = i + Num_velocity_gate_protect + 1; k < i + Num_velocity_gate_ref / 2 + Num_velocity_gate_protect; k++) {
                ref_sum += Sig[k * Range_gate_length + j];
            }
            if (Sig[i * Range_gate_length + j] > Threshold * (ref_sum / Num_velocity_gate_ref)) {
                CFAR_matrix[i * Range_gate_length + j] = Sig[i * Range_gate_length + j];
                Flag_matrix[i * Range_gate_length + j] = 1;
            }
        }
        else {
            double ref_sum = 0;
            for (int k = i - Num_velocity_gate_ref / 2 - Num_velocity_gate_protect; k < i - Num_velocity_gate_protect; k++) {
                ref_sum += Sig[k * Range_gate_length + j];
            }
            if (Sig[i * Range_gate_length + j] > Threshold * (ref_sum / (Num_velocity_gate_ref / 2))) {
                CFAR_matrix[i * Range_gate_length + j] = Sig[i * Range_gate_length + j];
                Flag_matrix[i * Range_gate_length + j] = 1;
            }
        }
    }
}

// MTI 两脉冲对消
// a:输入信号 输入numRow：行数（CPI）numClos:单周期采样点数
// b:输出信号
// 线程数为(CPI-1)*SampleNumber
__global__ void subtractMti(cufftComplex* a, cufftComplex* b, int numRows, int numCols) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < (numRows - 1) * numCols) {
        b[index] = { a[index + numCols].x - a[index].x, a[index + numCols].y - a[index + numCols].y };
    }
}

// 读取GPU里的数据并保存
void writeComplexToFile(const char* filename, cufftComplex* d_data, size_t size) {
    // 1. 分配主机内存
    cufftComplex* h_data = (cufftComplex*)malloc(size * sizeof(cufftComplex));
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return;
    }

    // 2. 从设备复制数据到主机
    cudaError_t err = cudaMemcpy(h_data, d_data, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return;
    }

    // 3. 打开文本文件以写入
    FILE* outfile = fopen(filename, "w");
    if (!outfile) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        free(h_data);
        return;
    }

    // 4. 写入数据
    for (size_t i = 0; i < size; i++) {
        // 输出实部
        fprintf(outfile, "%.6f", h_data[i].x);
        // 输出虚部，考虑正负号
        if (h_data[i].y >= 0) {
            fprintf(outfile, "+%.6fi ", h_data[i].y);
        }
        else {
            fprintf(outfile, "-%.6fi ", -h_data[i].y);
        }
    }

    // 5. 关闭文件
    fclose(outfile);

    // 6. 释放主机内存
    free(h_data);
}

// 目标凝聚：找邻域内局部最大值
__global__ void TargetsCohensionKernel(const float* CFAR_matrix, float* Cohension_matrix,
    int Velocity_gate_length, int Range_gate_length) {
    // 计算每个线程的二维索引
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程在矩阵内部
    if (i > 0 && i < Velocity_gate_length - 1 && j > 0 && j < Range_gate_length - 1) {
        // 获取当前点的值
        float current_value = CFAR_matrix[i * Range_gate_length + j];

        // 获取周围 8 个邻居的值
        bool is_local_max = true;
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                if (di == 0 && dj == 0) continue;
                int ni = i + di;
                int nj = j + dj;
                if (CFAR_matrix[ni * Range_gate_length + nj] >= current_value) {
                    is_local_max = false;
                    break;
                }
            }
            if (!is_local_max) break;
        }

        // 如果是局部极大值，则赋值给 Cohension_matrix
        if (is_local_max) {
            Cohension_matrix[i * Range_gate_length + j] = current_value;
        }
        else {
            Cohension_matrix[i * Range_gate_length + j] = 0.0f;
        }
    }
}

// 读取GPU里的double数据并保存
void writeDoubleToFile(const char* filename, double* d_data, size_t size) {
    // 1. 分配主机内存
    double* h_data = (double*)malloc(size * sizeof(double));
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return;
    }

    // 2. 从设备复制数据到主机
    cudaError_t err = cudaMemcpy(h_data, d_data, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return;
    }

    // 3. 打开文本文件以写入
    FILE* outfile = fopen(filename, "w");
    if (!outfile) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        free(h_data);
        return;
    }

    // 4. 写入数据
    for (size_t i = 0; i < size; i++) {
        // 输出实数值
        fprintf(outfile, "%.6lf ", h_data[i]);
    }

    // 5. 关闭文件
    fclose(outfile);

    // 6. 释放主机内存
    free(h_data);
}


void writeIntToFile(const char* filename, int* d_data, size_t size) {
    // 1. 分配主机内存
    int* h_data = (int*)malloc(size * sizeof(int));
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return;
    }

    // 2. 从设备复制数据到主机
    cudaError_t err = cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return;
    }

    // 3. 打开文本文件以写入
    FILE* outfile = fopen(filename, "w");
    if (!outfile) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        free(h_data);
        return;
    }

    // 4. 写入数据
    for (size_t i = 0; i < size; i++) {
        // 输出整数值
        fprintf(outfile, "%d ", h_data[i]);
    }

    // 5. 关闭文件
    fclose(outfile);

    // 6. 释放主机内存
    free(h_data);
}


void writeFloatToFile(const char* filename, float* d_data, size_t size) {
    // 1. 分配主机内存
    float* h_data = (float*)malloc(size * sizeof(float));
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return;
    }

    // 2. 从设备复制数据到主机
    cudaError_t err = cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return;
    }

    // 3. 打开文本文件以写入
    FILE* outfile = fopen(filename, "w");
    if (!outfile) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        free(h_data);
        return;
    }

    // 4. 写入数据
    for (size_t i = 0; i < size; i++) {
        // 输出实数值，使用 %.6f 格式化输出 float
        fprintf(outfile, "%.6f ", h_data[i]);
    }

    // 5. 关闭文件
    fclose(outfile);

    // 6. 释放主机内存
    free(h_data);
}


// 获取系统当前时间
long long getTimeInMicroseconds() {
    struct timespec ts;

    // 获取当前时间，单位是纳秒
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime failed");
        return -1;
    }

    // 将纳秒转换为微秒
    return (long long)(ts.tv_sec) * 1000000 + (ts.tv_nsec / 1000);
}







/************************* 未包含错误检测代码 ******************************************************************************
void setGPU()
{
    // 检测计算机GPU数量
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

    if (error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else
    {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }
    // 设置执行
    int iDev = 0;
    error = cudaSetDevice(iDev);
    if (error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for computing.\n");
    }
}
*********************************************************************************************************************/
