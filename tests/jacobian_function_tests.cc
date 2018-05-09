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
#include "catch.h"

#include "../library/jacobian_function.h"
#include "../library/util.h"

#define UNUSED(x) (void)(x)


template< size_t N
        , size_t M >
void test_create_identity_jacobian()
{

    std::string description = "N = ";
    description.append(std::to_string(N));
    description.append(", M = ");
    description.append(std::to_string(M));
    SECTION(description){

    using nx1_vector = typename kafi::jacobian_function<N,M>::nx1_vector;
    using mx1_vector = typename kafi::jacobian_function<N,M>::mx1_vector;
    using mxn_matrix = typename kafi::jacobian_function<N,M>::mxn_matrix;

    kafi::jacobian_function<N,M> prediction_scaling(
        std::move(kafi::util::create_identity_jacobian<N,M>()));


    double x = 1.5;
    // test
    nx1_vector input(x);
    // preallocating resources for inner computation
    mx1_vector h_result(0);
    mxn_matrix H_result(0);

    prediction_scaling(input, h_result);
    prediction_scaling.jacobian(input, H_result);

    mx1_vector h_ground_truth(0);
    for (size_t row = 0; row < M; ++row)
    {
        h_ground_truth(row, 0) = x;
    }

    mxn_matrix H_ground_truth(0);
    for (size_t row = 0; row < M; ++row)
    {
        H_ground_truth(row, 0) = 1;
    }

    REQUIRE((h_result == h_ground_truth));
    REQUIRE((H_result == H_ground_truth));
}
}


TEST_CASE("basic functionality of jacobian", "[jacobian]") {

    SECTION("jacobian N = 1, M = 2") {
        const size_t N = 1; // single state
        const size_t M = 2; // two sensors

        using nx1_vector = kafi::jacobian_function<N,M>::nx1_vector;
        using mx1_vector = kafi::jacobian_function<N,M>::mx1_vector;
        using mxn_matrix = kafi::jacobian_function<N,M>::mxn_matrix;

        using func            = kafi::jacobian_function<N,M>::func;
        using par_jacobi_func = kafi::jacobian_function<N,M>::par_jacobi_func;
        using jacobi_func     = kafi::jacobian_function<N,M>::jacobi_func;

        // copy the 1x1 state to both rows of 2x1
        const func h =
             [&](nx1_vector & input, mx1_vector & output){
                output(0, 0) = input(0,0);
                output(1, 0) = input(0,0);
        };

        // partial derivative of [ s_0;  s_0 ] -> [ 1; 1 ]
        const par_jacobi_func h_0_0 =
             [&](const nx1_vector & input){
                UNUSED(input);
                return 1;
        };
        const par_jacobi_func h_1_0 =
            [&](const nx1_vector & input){
                UNUSED(input);
                return 1;
        };

        // combining them to the matrix "view" of jacobians
        const jacobi_func H { { std::move(h_0_0) },
                              { std::move(h_1_0) } };

        // creation of jacobian function object with the corresponding functions
        kafi::jacobian_function<N,M> prediction_scaling(h, H);

        // test
        nx1_vector input({ 1.5 });
        // preallocating resources for inner computation
        mx1_vector h_result(0);
        mxn_matrix H_result(0);

        prediction_scaling(input, h_result);
        prediction_scaling.jacobian(input, H_result);

        mx1_vector h_ground_truth({ { 1.5 }
                                  , { 1.5 } });

        mxn_matrix H_ground_truth({ { 1 }
                                  , { 1 } });

        REQUIRE((h_result == h_ground_truth));
        REQUIRE((H_result == H_ground_truth));
    }

    SECTION("jacobian with different N / Ms") {
        test_create_identity_jacobian<1,4>();
        test_create_identity_jacobian<2,4>();
        test_create_identity_jacobian<5,5>();
        test_create_identity_jacobian<2,10>();
        test_create_identity_jacobian<1,100>();
        test_create_identity_jacobian<20,50>();
        test_create_identity_jacobian<50,4>();
        test_create_identity_jacobian<100,100>();
        // test_create_identity_jacobian<200,300>(); // SIGSEV (probably because of StaticMatrix size)
        // test_create_identity_jacobian<400,350>(); // SIGSEV (probably because of StaticMatrix size)
    }
}