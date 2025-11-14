cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(cir_radar)

# 强制设置构建类型为 Release 
# set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
##### include路径查询函数 ###############################################################
macro(FIND_INCLUDE_DIR result curdir)									#定义函数,2个参数:存放结果result；指定路径curdir；
	file(GLOB_RECURSE children "${curdir}/*.hpp" "${curdir}/*.h" "${curdir}/*.cuh")		#遍历获取{curdir}中*.hpp,*.h和*.cuh文件列表
	set(dirlist "")														#定义dirlist中间变量，并初始化
	foreach(child ${children})											#for循环
		string(REGEX REPLACE "(.*)/.*" "\\1" LIB_NAME ${child})			#字符串替换,用/前的字符替换/*h
		if(IS_DIRECTORY ${LIB_NAME})									#判断是否为路径
			list (FIND dirlist ${LIB_NAME} list_index)					#去重，查找dirlist中是否有${LIB_NAME}指定的值，可以区分字符串相同数子后缀不同的路径：例如/app/test_1和/app/test_2
			if(${list_index} LESS 0)									#若没找到则代表列表中没有该路径
				LIST(APPEND dirlist ${LIB_NAME})						#将合法的路径加入dirlist变量中  
			endif()														#结束判断
		endif()															
	endforeach()														#结束for循环
	set(${result} ${dirlist})											#dirlist结果放入result变量中
endmacro()
########################################################################################

#### 查找自定义源文件
# 查找source目录下的所有*.cpp,*.c*.cu源文件,并将文件列表保存到 DIR_LIB_SRCS 变量
file(GLOB_RECURSE USER_SOURCE_LIST "./source/*.cpp" "./source/*.c" "./source/*.cu")		#遍历获取source/路径下所有的*.cpp和*.c文件列表

#### 添加自定义头文件搜索路径
#查找include目录下的所有*.hpp,*.h头文件,并路径列表保存到 INCLUDE_DIR_LIST 变量中
FIND_INCLUDE_DIR(USER_INCLUDE_DIR_LIST ./include)
#将INCLUDE_DIR_LIST中路径列表加入工程		
include_directories(${USER_INCLUDE_DIR_LIST})  
## 添加CUDA的库 
include_directories(/usr/local/cuda/include)
########################################################################################

#### 添加库文件搜索路径
link_directories(/usr/local/bin)
link_directories(/usr/local/lib)
link_directories(/usr/local/lib64)
link_directories(/usr/local/cuda/lib64)

#### 添加编译参数
#### 开启优化，debug模式下注释
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CUDA)


	
#### 设置输出文件路径
# .so输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/lib")
#执行程序、dll动态库、pdb调试文件
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")
# 静态库 .lib 动态库.lib地址文件 linux静态库.a 静态库pdb调试文件
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/lib")
#### 生成可执行文件
add_executable(${PROJECT_NAME} ${USER_SOURCE_LIST})


#### 查找OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

#### 查找FFTW3
target_link_libraries(${PROJECT_NAME} PUBLIC fftw3f)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lfftw3f -lpthread")


#### 查找CUDA
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

## cuda的nvcc开启优化，debug模式下注释
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_80 -O3 -Xcompiler -O3")

# P1000 61 A4000 80
# 设置目标架构为 8.0（针对 RTX A4000）
set(CMAKE_CUDA_ARCHITECTURES "80")
set_target_properties(cir_radar PROPERTIES CUDA_ARCHITECTURES "80")

# 设置 CUDA 编译选项以静态链接
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSTATIC_LINKING")
target_link_libraries(${PROJECT_NAME} PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
)


#### 添加libtorch路径
list(APPEND CMAKE_PREFIX_PATH /home/ryh/ryh/cir_radar/lib/libtorch)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ../lib/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#### 添加链接库
target_link_libraries(cir_radar PUBLIC pthread)
target_link_libraries(cir_radar PUBLIC spdlog)
target_link_libraries(cir_radar PUBLIC hv)
target_link_libraries(cir_radar PUBLIC "${TORCH_LIBRARIES}")
set_property(TARGET cir_radar PROPERTY CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)