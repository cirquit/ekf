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
#include <random>
#include "catch.h"

#include "../library/util.h"

#define UNUSED(x) (void)(x)

template<size_t N>
void test_create_identity()
{
    std::string description = "N = ";
    description.append(std::to_string(N));
    SECTION(description){
        auto matrix(kafi::util::create_identity<N, blaze::rowMajor>());
        REQUIRE(blaze::isIdentity(matrix));
        REQUIRE(blaze::isDiagonal(matrix));
    }
}

TEST_CASE("util.h", "[util]") {

    SECTION("testing create_identity", "N = 1,2,5,11,24,40,50,100,250,1000") {
        test_create_identity<1>();
        test_create_identity<2>();
        test_create_identity<5>();
        test_create_identity<11>();
        test_create_identity<24>();
        test_create_identity<40>();
        test_create_identity<50>();
        test_create_identity<100>();
        test_create_identity<250>();
        test_create_identity<1000>();
    }

    SECTION("testin identity_broadcast_function") {
        const size_t N = 1;
        const size_t M = 4;

        using nx1_vector = blaze::StaticMatrix<double, N, 1UL, blaze::rowMajor>;
        using mx1_vector = blaze::StaticMatrix<double, M, 1UL, blaze::rowMajor>;
        using func       = std::function<void(nx1_vector &, mx1_vector &)>;

        nx1_vector input( { { 1 } } );
        mx1_vector output(0);
        mx1_vector ground_truth({ { 1 }
                                , { 1 }
                                , { 1 }
                                , { 1 } });

        const func f(kafi::util::identity_broadcast_function<N,M>());

        f(input, output);

        REQUIRE(output == ground_truth);
    }

    SECTION("testing identity_derivative"){
        const size_t N = 1;

        using nx1_vector = blaze::StaticMatrix<double, N, 1UL, blaze::rowMajor>;
        using par_jacobi_func = std::function<double(const nx1_vector &)>;

        const nx1_vector input({ { 0 } });

        const par_jacobi_func f_one = kafi::util::identity_derivative<N>(1.0);
        double result_one = f_one(input);

        REQUIRE(result_one == 1.0);

        const par_jacobi_func f_zero = kafi::util::identity_derivative<N>(0.0);
        double result_zero = f_zero(input);

        REQUIRE(result_zero == 0.0);
    }
}
