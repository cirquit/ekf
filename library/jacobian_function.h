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

#ifndef JACOBIAN_FUNCTION_H
#define JACOBIAN_FUNCTION_H

#include <functional>
#include <blaze/Math.h>

namespace kafi {

/**
 * \brief A wrapper function that stores a function and its jacobian.
 * 
 * This is currently working with references of the return values so this class doesn't have any ownership over the stored data
 *
 * Template arguments:
 * * `N`  = state dimensions
 * * `M`  = sensor dimensions
 *  
 * For examples, see [tests/jacobian_function_tests.cc](../../tests/jacobian_function_tests.cc)
 */
template< size_t N  // input dimensions  (N x 1)
        , size_t M> // output dimensions (M x 1)
class jacobian_function {

    // typenames
    public:
        //! self type for conciseness
        using self_t     = jacobian_function<N,M>;

        /** `N` rows, `1` column `(N x 1)` */
        using nx1_vector = blaze::StaticMatrix<double, N, 1UL, blaze::rowMajor>; 
        /** `M` rows, `1` column `(M x 1)` */
        using mx1_vector = blaze::StaticMatrix<double, M, 1UL, blaze::rowMajor>;
        /** Return type of applied partial derivatives with `M` rows, `N` columns */
        using mxn_matrix = blaze::StaticMatrix<double, M,   N, blaze::rowMajor>;
        /** defining this type here to have a single point of access */
        using nxm_matrix = blaze::StaticMatrix<double, N,   M, blaze::rowMajor>;
        /** defining this type here to have a single point of access */
        using mxm_matrix = blaze::StaticMatrix<double, M,   M, blaze::rowMajor>;
        /** defining this type here to have a single point of access */
        using nxn_matrix = blaze::StaticMatrix<double, N,   N, blaze::rowMajor>;
        /** function that takes `(N x 1)` and returns `(M x 1)` */
        using func            = std::function<void(nx1_vector &, mx1_vector &)>;
        /** partial derivative of fun for a single dimension of `N` */
        using par_jacobi_func = std::function<double(const nx1_vector &)>;
        /** full `M x N` matrix of partial derivatives of jacobian_function::func */
        using jacobi_func     = blaze::StaticMatrix<par_jacobi_func, M, N, blaze::rowMajor>;

    // constructors
    public:
        //! Default constructor with the normal function `f` and its derivative `F`
        constexpr jacobian_function(func f, jacobi_func F)
        : _f(f)
        , _F(F){ }

        //! copy constructor is deleted, because we want to disallow copying of matrices (which may be added to this class ownership)
        constexpr jacobian_function(const self_t & other) = delete;
        //! move constructor
        constexpr jacobian_function(const self_t && other)
        : _f(std::move(other._f))
        , _F(std::move(other._F)) { }

    // methods
    public:
        /**
         * \brief Forwarding to the 'f' function
         * * `state` ~= input, **IMPORTANT**: state might be equal to output, so use a deep copy if you need the values
         * * `output` is the return type, because we want a constant memory footprint
         *    - therefore the return type is preallocated and reused
         * 
         * \todo make a functional version of this call with `mx1_vector` as return type and test the performance decrease
         */
        constexpr void operator()(nx1_vector & state, mx1_vector & output) const
        {
            return _f(state, output);
        }

        /**
         * \brief Iterates over all partial derivatives of `_f` in `_F` and runs them with `state`
         * 
         * Saving the results in 'jacobi_temp' matrix to be fully functional and parallelizable

         * \todo make a functional version of this call with `mxn_matrix` as return type and test the performance decrease
         */
        constexpr mxn_matrix & jacobian(const nx1_vector & state, mxn_matrix & jacobi_temp) const
        {
            for(size_t row = 0UL; row < _F.rows(); ++row)
            {
                for(size_t col = 0UL; col < _F.columns(); ++col)
                {
                    const par_jacobi_func & j_func = _F(row, col);
                    jacobi_temp(row, col) = j_func(state);
                }
            }
            return jacobi_temp; 
        }

    // member
    public:

    // member
    private:
        //! normal function `_f :: nx1_vector -> mx1_vector`
        const func        _f;
        //! jacobian function `_F :: nx1_vector -> mxn_matrix`
        const jacobi_func _F;
};

} // namespace jacobian_function

#endif // JACOBIAN_FUNCTION_H