

#include "rsl_win_func_manager.hpp"


namespace radar_signal_process{
	rsl_win_func_manager::rsl_win_func_manager(/* args */)
	{
		win_map_t::const_iterator map_it = taylor_win_map.begin();
        while (map_it != taylor_win_map.end()) {
            delete[] map_it->second;
            ++map_it;
        }
        map_it = cheb_win_map.begin();
        while (map_it != cheb_win_map.end()) {
            delete[] map_it->second;
            ++map_it;
        }
        map_it = hamming_win_map.begin();
        while (map_it != hamming_win_map.end()) {
            delete[] map_it->second;
            ++map_it;
        }
	}
	
	rsl_win_func_manager::~rsl_win_func_manager()
	{
	}

	/**
	 * @brief     获取窗函数管理器唯一实例
	 * @return    rsl_win_func_manager& 
	 */
	rsl_win_func_manager& rsl_win_func_manager::getInstance(){
		static rsl_win_func_manager instance;
        return instance;
	}
	
	/**
	 * @brief     返回len的下一个二次幂值
	 * @param     [in] len 输入数据值      
	 * @return    int      输入数据值的下一个二次幂值
	 */
	int rsl_win_func_manager::nextpow2(int len){
		int fftn;

		if(len > 65535 && len <= 131072)
            fftn = 131072;
		else if(len > 32768 && len <= 65535)
            fftn = 65535;
		else if(len > 16384 && len <= 32768)
            fftn = 32768;
		else if(len > 8192 && len <= 16384)
            fftn = 16384;
		else if(len > 4096 && len <= 8192)
            fftn = 8192;
		else if(len > 2048 && len <= 4096)
            fftn = 4096;
        else if(len > 1024 && len <= 2048)
            fftn = 2048;
        else if(len > 512 && len <= 1024)
            fftn = 1024;
        else if(len > 256 && len <= 512)
            fftn = 512;
        else if(len > 128 && len <= 256)
            fftn = 256;
        else if(len > 64 && len <= 128)
            fftn = 128;
        else if(len > 32 && len <= 64)
            fftn = 64;
        else if(len > 16 && len <= 32)
            fftn = 32;
        else
            fftn = 32;

        return fftn;
	}

	/**
	 * @brief     获取指定类型、长度、副瓣相对衰减的窗函数
	 * @param     [in] win_type  窗函数类型
	 * @param     [in] len       窗长度
	 * @param     [in] sll       副瓣相对衰减，取绝对值
	 * @return    float*         窗函数数组指针
	 */
	float * rsl_win_func_manager::get_win_func(win_type_t win_type, int len, float sll){
		if(len < 0)
            return nullptr;

        win_map_t::iterator iter;
        win_map_t* win_map_tmp;

        switch (win_type)
        {
        case Taylor:
            win_map_tmp = &taylor_win_map;
            break;
        case Chebyshev:
            win_map_tmp = &cheb_win_map;
            break;
        case Hamming:
            win_map_tmp = &hamming_win_map;
            break;
        
        default:
            break;
        }

        iter = win_map_tmp->find(std::pair<int,float>(len,sll));
        if(iter != win_map_tmp->end()){
            return iter->second;
        }else{
            float* new_win = new float[len];
            switch (win_type)
            {
            case Taylor:
                rsl_taylorWin(len,4,-sll,new_win);
                break;
            case Chebyshev:
                rsl_chebWin(len,sll,new_win);
                break;
            case Hamming:
                rsl_hammingWin(len,new_win);
                break;
            default:
                break;
            }
            win_map_tmp->insert(win_map_t::value_type(std::pair<int,float>(len,sll),new_win));
            return new_win;
        }
	}

	/* ==========================================================
	* 泰勒窗
	* 使用方法和MATLAB类似
	* w = tayloyWin(n,nbar,sll)returns an n-point Taylor window
	* with a maximum sidelobe level of sll dB relative to the
	* mainlobe peak. sll must be negative.
	* 【algorithm refer to taylorwin.m from Matlab】
	* N > 1 , nbar > 1 , sll < 0
	* example:
	* ret = rsl_taylorWin(64,4,-35,&w);
	* ==========================================================*/
	int rsl_win_func_manager::rsl_taylorWin(unsigned int N, unsigned int nbar,float sll,float *w)
	{
		int m,n,k;
		float Num,Den,sp2,A,pi,xi,summation;

		pi = 3.1415926;
		A = acosh(pow(10 , (-sll/20))) / pi;
		//Taylor pulse widening (dilation) factor.
		sp2 = pow(nbar,2) / (pow(A,2) + pow((nbar - 0.5),2));

		float Fm[10];

		//calculate Fm
		for(m = 1; m < nbar; m++)
		{
			Num = 1.0;
			Den = 1.0;
			for(n = 1;n < nbar; n++)
			{
				Num *= (1 - (pow(m,2) / sp2) / (pow(A,2)+ pow((n - 0.5),2)));
				if(n != m)
					Den *= (1 - pow(m,2) / pow(n,2));
			}
			Fm[m - 1] = pow((-1),(m+1)) * Num / (2 * Den);
		}

		for(k = 0 ; k < N; k ++)
		{
			summation = 0;
			for(m = 1; m < nbar; m++)
			{
				xi = (k - 0.5 * N + 0.5) / N;
				summation += Fm[m - 1] * cos(2 * pi * m * xi);
			}
			w[k] = 1.0 + 2 * summation;
		}
		return 1;
	}
	/* ==========================================================
	* 切比雪夫窗函数
	* 使用方法和MATLAB类似
	* N > 1 , r > 0
	* example:
	* ret = rsl_chebWin(64,35,&w);
	* ==========================================================*/
	int rsl_win_func_manager::rsl_chebWin(unsigned int N,float r,float *w)
	{
		int n,index;
		float *ret;
		float x, alpha, beta, theta, gama;

		ret = w;


		/*10^(r/20)*/
		theta = pow((float)10, (float)(r/20.0));
		beta = pow(cosh(acosh(theta)/(N - 1)),2);
		alpha = 1 - (float)1 / beta;

		if((N % 2) == 1)
		{
			/*计算一半的区间*/
			for( n = 1; n < ( N + 1 ) / 2; n++ )
			{
				gama = 1;
				for(index = 1; index < n; index++)
				{
					x = index * (float)( N - 1 - 2 * n + index) /(( n - index ) * (n + 1 -index));
					gama = gama * alpha * x + 1;
				}
				*(ret + n) = (N - 1) * alpha * gama;
			}

			theta = *( ret + (N - 1)/2 );
			*ret = 1;

			for(n = 0; n < ( N + 1 ) / 2; n++ )
			{
				*(ret + n) = (float)(*(ret + n)) / theta;
			}

			/*填充另一半*/
			for(; n < N; n++)
			{
				*(ret + n) = ret[N - n - 1];
			}
		}
		else
		{
			/*计算一半的区间*/
			for( n = 1; n < ( N + 1 ) / 2; n++ )
			{
				gama = 1;
				for(index = 1; index < n; index++)
				{
					x = index * (float)( N - 1 - 2 * n + index) /(( n - index ) * (n + 1 -index));
					gama = gama * alpha * x + 1;
				}
				*(ret + n) = (N - 1) * alpha * gama;
			}

			theta = *( ret + (N/2) - 1);
			*ret = 1;

			for(n = 0; n < ( N + 1 ) / 2; n++ )
			{
				*(ret + n) = (float)(*(ret + n)) / theta;
			}

			/*填充另一半*/
			for(; n < N; n++)
			{
				*(ret + n) = ret[N - n - 1];
			}
		}

		return 1;
	}
	/* ==========================================================
	* 汉明窗
	* 使用方法和MATLAB类似
	* N > 1
	* example:
	* ret = rsl_hammingWin(64,&w);
	* ==========================================================*/
	int rsl_win_func_manager::rsl_hammingWin(unsigned int N, float *w)
	{
		unsigned int n;
		float *ret;
		float pi;

		pi = 3.1415926;
		ret = w;

		for(n = 0; n < N; n++)
		{
			*(ret + n) = 0.54 - 0.46 * cos (2 * pi *  ( float )n / ( N - 1 ) );
		}

		return 1;
	}

	/**
	 * @brief     获取圆阵泰勒窗系数
	 * @param     [in] R         阵元半径，单位为mm
	 * @param     [in] N         阵元数，即泰勒窗系数
	 * @param     [in] nbar      NBAR nearly constant-level sidelobes adjacent to the mainlobe. 
	 * @param     [in] sll       最大旁瓣电平，相对于主瓣
	 * @param     [in] w         窗系数数组指针
	 * @return    int            
	 */
	int rsl_win_func_manager::rsl_circal_array_taylorwin(int R, unsigned int N, unsigned int nbar,float sll,float *w){
		float pi = 3.1415926;

		//将圆阵阵元投影平面，获得各阵元在平面投影的横坐标
		float x_pos[N];
		for (int index = 0; index < N; index++)
		{
			x_pos[index] = roundf(R*sinf((index * 10 - 55)* pi /180));
		}

		for (int index = N - 1; index >= 0; index--)
		{
			//将坐标取非负值
			x_pos[index] += -x_pos[0];
		}
		

		//以1mm为间隔产生泰勒系数
		float taylorwin_tmp[(unsigned int)x_pos[N - 1] + 1];
		rsl_taylorWin((unsigned int)x_pos[N - 1] + 1, nbar, -sll, taylorwin_tmp); 

		//抽取对应阵元的系数并归一化
		float max_taylor = 0;
		for (int index = 0; index < N; index++)
		{
			w[index] = taylorwin_tmp[(int)x_pos[index]];
			if(max_taylor < w[index]){
				max_taylor = w[index];
			}
		}
		for (int index = 0; index < N; index++)
		{
			w[index] /= max_taylor;
		}

		return 0;
	}


}