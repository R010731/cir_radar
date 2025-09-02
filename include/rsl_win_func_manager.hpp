/**
 * @file      rsl_win_func_manager.hpp
 * @author    lizhengwei (waiwaylee@foxmail.com)
 * @version   1.0
 * @date      2022-11-10
 * @brief     
 */

#pragma once

#ifndef __RSL_WIN_FUNC_MANAGER
#define __RSL_WIN_FUNC_MANAGER

#include <iostream>
#include <utility>
#include <map>
#include <math.h>

typedef std::map<std::pair<int,float>,float*> win_map_t;

namespace radar_signal_process{

	typedef enum{
        Taylor = 0,
        Chebyshev = 1,
        Hamming = 2
    }win_type_t;

	class rsl_win_func_manager
	{
	private:

		//Ì©ÀÕ´°
        win_map_t taylor_win_map;
        //ÇÐ±ÈÑ©·ò´°
        win_map_t cheb_win_map;
        //ººÃ÷´°
        win_map_t hamming_win_map;

		rsl_win_func_manager(/* args */);

	public:

		static rsl_win_func_manager& getInstance();

		rsl_win_func_manager(const rsl_win_func_manager&)=delete;
    	rsl_win_func_manager& operator=(const rsl_win_func_manager&)=delete;
		
		~rsl_win_func_manager();

		int nextpow2(int len);

		float * get_win_func(win_type_t win_type, int len, float sll);

		int rsl_taylorWin(unsigned int N, unsigned int nbar,float sll,float *w);
		int rsl_chebWin(unsigned int N,float r,float *w);
		int rsl_hammingWin(unsigned int N, float *w);

		int rsl_circal_array_taylorwin(int R, unsigned int N, unsigned int nbar,float sll,float *w);
	};
}






#endif