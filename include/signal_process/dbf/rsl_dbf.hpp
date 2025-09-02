#pragma once

#ifndef __RSL_DBF_HPP
#define __RSL_DBF_HPP

#include "rsl_dbf_base.hpp"

#define DBF_THREAD_NUM              THREADS_NUM_FOR_BEAM_0

using namespace::std;
using namespace::rsl_debug_tools;
using namespace::Eigen;

typedef Matrix<complex<float>, 1,           Dynamic,        RowMajor>       MatrixDDC_chn;
typedef Matrix<complex<float>, Dynamic,     Dynamic,        RowMajor>       MatrixDDC_sector;
typedef Matrix<complex<float>, 1,           Dynamic,        RowMajor>       MatrixDBF_coeff;
typedef Matrix<complex<float>, Dynamic,     Dynamic,        RowMajor>       MatrixDBF_beam;
typedef Map<MatrixDDC_chn> MapDDC_chn;
typedef Map<MatrixDBF_beam> MapDDC_beam;

namespace radar_signal_process{

    class rsl_dbf : public rsl_dbf_base
    {
    private:
        int dbf_channel_number;
        int dbf_narrow_weighting_boundry;
        int dbf_wide_weighting_boundry;
        float radius_of_attenna_array;

        void params_update();

        void process(cpi_data_t& cpi_data, int azi_sector_num, dbf_t& dbf_output);

    public:
        rsl_dbf();
        ~rsl_dbf();
    };
}

#endif