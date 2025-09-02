nsys profile --stats=true -o cir_radar ./cir_radar 生成CUDA分析报告
nsys profile --trace=cuda,osrt,opengl -o cir_radar ./cir_radar
nsys profile -o report_name ./your_program

其中，report_name是生成的报告文件名，./your_program是要分析的可执行程序。该命令会生成一个.nsys-rep格式的报告文件，包含详细的性能数据。
常用命令行参数:
    --stats=true：在分析后打印统计信息。
    --trace=cuda,osrt,opengl：分析CUDA、操作系统调度和OpenGL相关的性能数据。
    --capture-range=cudaProfilerApi：分析特定的CUDA内核。


release编译 :/usr/bin/cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-12 --no-warn-unused-cli -S /home/ryh/ryh/cir_radar -B /home/ryh/ryh/cir_radar/build -G Ninja

GPU 圆阵雷达信号处理
1.时间和内存
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)  Name                                            --------------------------------------------------------------------------------------------------------                                               
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     86.0       22,632,464      8,192      2,762.8      2,720.0      2,592     10,880        398.7  void gemv2N_kernel 
      9.8        2,579,046          1  2,579,046.0  2,579,046.0  2,579,046  2,579,046          0.0  complexMultiply                            
      2.0          513,633          2    256,816.5    256,816.5    252,737    260,896      5,769.3  void vector_fft
      1.3          354,848          1    354,848.0    354,848.0    354,848    354,848          0.0  void regular_fft_factor
      0.9          242,817          1    242,817.0    242,817.0    242,817    242,817          0.0  void regular_fft_factor


 Time (%)  Total Time (ns)  Count   Avg (ns)  Med (ns)  Min (ns)  Max (ns)   StdDev (ns)           Operation          
 --------  ---------------  ------  --------  --------  --------  ---------  -----------  ----------------------------
     58.3       29,206,864  16,390   1,782.0   1,440.0       192    898,338      9,675.1  [CUDA memcpy Host-to-Device]
     41.7       20,904,060   8,196   2,550.5   1,504.0       704  3,671,144     57,200.5  [CUDA memcpy Device-to-Host]


 Total (MB)  Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  ------  --------  --------  --------  --------  -----------  ----------------------------
    198.430  16,390     0.012     0.002     0.000     8.389        0.085  [CUDA memcpy Host-to-Device](CPU到GPU)
     45.613   8,196     0.006     0.003     0.001     8.389        0.171  [CUDA memcpy Device-to-Host](GPU到CPU)

DBF矩阵乘法核函数：gemv2N_kernel占用了 86% 的时间，总执行时间为 22,632,464 纳秒，约23ms，执行了 8,192(2048*4) 次。每次执行的平均时间为 2,762.8 纳秒。（2048*2次[1*12]*[12*320]矩阵乘法，2048*2次[1*12]*[12*144]矩阵乘法，实现一个扇区的DBF，得到宽脉冲和信号，宽脉冲差信号，窄脉冲和信号，窄脉冲差信号四路信号）
之后的信号处理目前只实现了脉冲和信号的PC，MTD
PC的频域相乘核函数：complexMultiply总时间为 2,579,046 纳秒，约为2ms。（2048*512次复数乘法）
PC的信号FFT和IFFT核函数：void vector_fft执行两次，总用时513,633ns，约513us。（2048次512点的FFT和IFFT）
MTD核函数：两个点数不同的void regular_fft_factor实现，总用时597,665ns,597us。（320次2048点的FFT）。



2. GPU 接口速率和带宽：https://blog.csdn.net/u013431916/article/details/81912266