// Copyright 2018 municHMotorsport e.V. <info@munichmotorsport.de>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <blaze/Math.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <memory>
#include "catch.h"
#include "csv.h"

#include "../library/kafi.h"

#define UNUSED(x) (void)(x)

TEST_CASE("kalman filter examples", "[kafi]") {

    SECTION("temperature test, N = 1, M = 2") {
        const size_t N = 1UL;
        const size_t M = 2UL;

        using mx1_vector = typename kafi::jacobian_function<N,M>::mx1_vector;
        using nx1_vector = typename kafi::jacobian_function<N,M>::nx1_vector;
        using mxm_matrix = typename kafi::jacobian_function<N,M>::mxm_matrix;
        using nxn_matrix = typename kafi::jacobian_function<N,M>::nxn_matrix;
        using return_t   = typename kafi::kafi<N,M>::return_t;

        // state transition
        kafi::jacobian_function<N,N> f(
            std::move(kafi::util::create_identity_jacobian<N,N>()));

        // prediction scaling (state -> observations)
        kafi::jacobian_function<N,M> h(
            std::move(kafi::util::create_identity_jacobian<N,M>()));

        // given by our example, read as "the real world temperature changes are 0.22° (0.22^2 =~ 0.05)"
        nxn_matrix process_noise( { { 0.05 } } );
        // given by our example, read as "both temperature sensors fluctuate by 0.8° (0.8^2 = 0.64)"
        mxm_matrix sensor_noise( { { 0.64, 0    }
                                 , { 0,    0.64 } });
        // given by our example, read as "first time we measured temperature, we got these values"
        std::shared_ptr< mx1_vector > first_observation = std::make_shared< mx1_vector >(
                              mx1_vector({ { 18.625 } 
                                         , { 20     } }));

        /* given by our example, the first observation is used to approximate the current state
         *     * sensor_1 = 20.33°
         *     * sensor_2 = 20.94°
         * mean_state: (20.33 + 20.94) / 2 = 20.64
         */
        nx1_vector starting_state( { { 20.64 } } );

        // init kalman filter
        kafi::kafi<N,M> kafi(std::move(f)
                           , std::move(h)
                           , starting_state
                           , process_noise
                           , sensor_noise);
        // update the observation
        kafi.set_current_observation(first_observation);
        // run the estimation
        return_t   result          = kafi.step();
        nx1_vector estimated_state = std::get<0>(result);
        // given by another kalman implementation in python which was validated by this implementation https://home.wlu.edu/~levys/kalman_tutorial/
        double ground_truth = 19.62;
        // because of rounding between the different implementations, this may be the +/- difference to the ground truth
        double eps = 0.01; 

        REQUIRE(ground_truth == Approx(estimated_state(0,0)).epsilon(eps));
    }

    SECTION("acceleration / correvit test, N = 7, M = 5") {
                              // 0  1   2   3   4   5   6
        const size_t N = 7UL; // x, y, ax, ay, vx, vy, phi
        const size_t M = 5UL; //     , ax, ay, vx, vy, phi
                              //        0   1   2   3   4

        using mx1_vector = typename kafi::jacobian_function<N,M>::mx1_vector;
        using nx1_vector = typename kafi::jacobian_function<N,M>::nx1_vector;
        using mxm_matrix = typename kafi::jacobian_function<N,M>::mxm_matrix;
        using nxn_matrix = typename kafi::jacobian_function<N,M>::nxn_matrix;
        using return_t   = typename kafi::kafi<N,M>::return_t;

        using f_func          = std::function<void(nx1_vector &, nx1_vector &)>;
        using h_func          = std::function<void(nx1_vector &, mx1_vector &)>;
        using par_jacobi_func = std::function<double(const nx1_vector &)>;
        using f_jacobi_func   = kafi::jacobian_function<N,N>::jacobi_func;
        using h_jacobi_func   = kafi::jacobian_function<N,M>::jacobi_func;

        // we have a sample rate of 0.001 second, or 1 millisecond, or 1000Hz
        const double t  = 0.001;
        // t squared precomputed
        const double t2 = 0.000001;

        // state transition model with updates to x,y from the a(x,y), v(x,y) and phi
        const f_func _f =
             [t,t2](nx1_vector & input, nx1_vector & output){

                double x   = input(0, 0);
                double y   = input(1, 0);
                double ax  = input(2, 0);
                double ay  = input(3, 0);
                double vx  = input(4, 0);
                double vy  = input(5, 0);
                double phi = input(6, 0);
                
                // x update
                output(0, 0) = (0.5 * ax * t2 + vx*t) * std::cos(phi) \
                             + (0.5 * ay * t2 + vy*t) * std::sin(phi) \
                             + x;
                // y update
                output(1, 0) = - (0.5 * ax * t2 + vx*t) * std::sin(phi) \
                             + (0.5 * ay * t2 + vy*t) * std::cos(phi) \
                             + y;
                // vx update
                output(4, 0) = vx + ax*t;
                // vy update
                output(5, 0) = vy + ay*t;

        };
        // jacobian of `f`
        
        // first row of the jacobian, derivative of f0 (the output(0,0), x update)
        const par_jacobi_func df0_dx  = [](const nx1_vector & in){ UNUSED(in); return 1; };
        const par_jacobi_func df0_dy  = [](const nx1_vector & in){ UNUSED(in); return 0; };
        const par_jacobi_func df0_dax = [t2](const nx1_vector & in)
        {
             double phi = in(6, 0);
             return 0.5*t2*std::cos(phi);
        };
        const par_jacobi_func df0_day = [t2](const nx1_vector & in)
        {
            double phi = in(6, 0);
            return 0.5*t2*std::sin(phi);
        };
        const par_jacobi_func df0_dvx = [t](const nx1_vector & in)
        {
            double phi = in(6, 0);
            return t*std::cos(phi);
        };
        const par_jacobi_func df0_dvy = [t](const nx1_vector & in)
        {
            double phi = in(6, 0);
            return t*std::sin(phi);
        };
        const par_jacobi_func df0_dphi = [t,t2](const nx1_vector & in)
        {
            double ax  = in(2, 0);
            double ay  = in(3, 0);
            double vx  = in(4, 0);
            double vy  = in(5, 0);
            double phi = in(6, 0);
            return std::cos(phi)*(0.5*ay*t2 + vy*t) - std::sin(phi)*(0.5*ax*t2+vx*t);
        };

        // second row of the jacobian, derivative of f1 (the output(1,0), y update)
        const par_jacobi_func df1_dx  = [](const nx1_vector & in){ UNUSED(in); return 0; };
        const par_jacobi_func df1_dy  = [](const nx1_vector & in){ UNUSED(in); return 1; };
        const par_jacobi_func df1_dax = [t2](const nx1_vector & in)
        {
            double phi = in(6, 0);
            return -0.5*t2*std::sin(phi);
        };
        const par_jacobi_func df1_day = [t2](const nx1_vector & in)
        {
            double phi = in(6, 0);
            return 0.5*t2*std::cos(phi);
        };
        const par_jacobi_func df1_dvx = [t](const nx1_vector & in)
        {
            double phi = in(6, 0);
            return -t*std::sin(phi);
        };
        const par_jacobi_func df1_dvy = [t](const nx1_vector & in)
        {
            double phi = in(6, 0);
            return t*std::cos(phi);
        };
        const par_jacobi_func df1_dphi = [t,t2](const nx1_vector & in)
        {
            double ax  = in(2, 0);
            double ay  = in(3, 0);
            double vx  = in(4, 0);
            double vy  = in(5, 0);
            double phi = in(6, 0);
            return -std::cos(phi)*(0.5*ax*t2 + vx*t) - std::sin(phi)*(0.5*ay*t2+vy*t);
        };

        // fifth row of the jacobian, derivative of f4 (the output(4,0), vx update)
        const par_jacobi_func df4_ax = [t](const nx1_vector & in){ UNUSED(in); return t; };
        // sixth row of the jacobian, derivative of f5 (the output(5,0), vy update)
        const par_jacobi_func df5_ay = [t](const nx1_vector & in){ UNUSED(in); return t; };

        const par_jacobi_func df_one  = kafi::util::identity_derivative<N>(1);
        const par_jacobi_func df_zero = kafi::util::identity_derivative<N>(0);

        const f_jacobi_func _F
        {  //      x        y        ax       ay       vx       vy       phi
/*f0*/     { df0_dx,  df0_dy,  df0_dax, df0_day, df0_dvx, df0_dvy, df0_dphi } 
/*f1*/   , { df1_dx,  df1_dy,  df1_dax, df1_day, df1_dvx, df1_dvy, df1_dphi } 
/*f2*/   , { df_zero, df_zero, df_one,  df_zero, df_zero, df_zero, df_zero  } 
/*f3*/   , { df_zero, df_zero, df_zero, df_one,  df_zero, df_zero, df_zero  } 
/*f4*/   , { df_zero, df_zero, df4_ax,  df_zero, df_one,  df_zero, df_zero  } 
/*f5*/   , { df_zero, df_zero, df_zero, df5_ay,  df_zero, df_one,  df_zero  } 
/*f6*/   , { df_zero, df_zero, df_zero, df_zero, df_zero, df_zero, df_one   } 
        };

        kafi::jacobian_function<N,N> f(_f, _F);

        // cut the `x` and `y` from the state vector
        const h_func _h = [](nx1_vector & in, mx1_vector & out)
        {
            out(0,0) = in(2,0);
            out(1,0) = in(3,0);
            out(2,0) = in(4,0);
            out(3,0) = in(5,0);
            out(4,0) = in(6,0);
        };

        const par_jacobi_func dh_one  = kafi::util::identity_derivative<N>(1);
        const par_jacobi_func dh_zero = kafi::util::identity_derivative<N>(0);

        const h_jacobi_func _H
        {
            { dh_zero, dh_zero, dh_one,  dh_zero, dh_zero, dh_zero, dh_zero }
         ,  { dh_zero, dh_zero, dh_zero, dh_one,  dh_zero, dh_zero, dh_zero }
         ,  { dh_zero, dh_zero, dh_zero, dh_zero, dh_one,  dh_zero, dh_zero }
         ,  { dh_zero, dh_zero, dh_zero, dh_zero, dh_zero, dh_one,  dh_zero }
         ,  { dh_zero, dh_zero, dh_zero, dh_zero, dh_zero, dh_zero, dh_one  }
        };

        kafi::jacobian_function<N,M> h(_h, _H);

        nxn_matrix process_noise(
        {
            { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 }
          , { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 }
          , { 0.00000, 0.00000, 0.10000, 0.00000, 0.00000, 0.00000, 0.00000 }
          , { 0.00000, 0.00000, 0.00000, 0.10000, 0.00000, 0.00000, 0.00000 }
          , { 0.00000, 0.00000, 0.00000, 0.00000, 0.10000, 0.00000, 0.00000 }
          , { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.10000, 0.00000 }
          , { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.10000 }
        } );

        mxm_matrix sensor_noise( { { 0.7,   0,    0,    0,    0  }
                                 , { 0,   0.7,    0,    0,    0  }
                                 , { 0,     0, 0.45,    0,    0  }
                                 , { 0,     0,    0, 0.45,    0  }
                                 , { 0,     0,    0,    0, 0.001 }
                                 });


        // Time[s], ax[m/s^2], ay[m/s^2], vx[m/s], vy[m/s], psi[rad]
        std::string csv_path = "test-data/2017-01-01-sensordata-wemding-alle-runden.csv";

        io::CSVReader<5> in(csv_path);
        in.read_header(io::ignore_extra_column, "ax[m/s^2]", "ay[m/s^2]", "vx[m/s]", "vy[m/s]", "psi[rad]");
        double ax_, ay_, vx_, vy_, phi_;
        in.read_row(ax_, ay_, vx_, vy_, phi_);

        std::shared_ptr< mx1_vector > first_observation = std::make_shared< mx1_vector >( 
                          mx1_vector( { { ax_ } 
                                      , { ay_ }
                                      , { vx_ }
                                      , { vy_ }
                                      , { phi_} }));

        nx1_vector starting_state( { { 0 }
                                   , { 0 }
                                   , { ax_ } 
                                   , { ay_ }
                                   , { vx_ }
                                   , { vy_ }
                                   , { phi_} });


        // init kalman filter
        kafi::kafi<N,M> kafi(std::move(f)
                           , std::move(h)
                           , starting_state
                           , process_noise
                           , sensor_noise);

        kafi.set_current_observation(first_observation);
        kafi.step();

        while(in.read_row(ax_, ay_, vx_, vy_, phi_))
        {   
            std::shared_ptr< mx1_vector > second_observation = std::make_shared< mx1_vector >(
                                mx1_vector({ { ax_ } 
                                           , { ay_ }
                                           , { vx_ }
                                           , { vy_ }
                                           , { phi_} }));
            // update the observation
            kafi.set_current_observation(second_observation);

            // run the estimation
            return_t   result          = kafi.step();
            nx1_vector estimated_state = std::get<0>(result);
            UNUSED(estimated_state);
            // nxn_matrix predicton_error = std::get<1>(result);
            // std::cout << predicton_error << '\n';
            // std::cout << estimated_state(0,0) << ", " << estimated_state(1,0) << '\n';
            // std::cout << estimated_state(4,0) - vx_ << ", " << estimated_state(5,0) - vy_ << '\n';
            // std::cout << estimated_state(2,0) - ax_ << ", " << estimated_state(3,0) - ay_ << '\n';
        }
        // to plot the output:
        // ```
        // cd build && make
        // build> ./tests/kafi_test > temp.csv
        // build> gnuplot
        // gnuplot> plot 'temp.csv'
        //

        REQUIRE(true);
    }
}
