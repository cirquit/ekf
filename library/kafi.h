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

#ifndef KAFI_H
#define KAFI_H

#include <functional>
#include <iostream>
#include <memory>
#include "jacobian_function.h"
#include "util.h"
#include "autogen-KAFI-macros.h"

/*!
 *  \addtogroup kafi
 *  @{
 */

/** \brief KaFi namespace, which is the main wrapper of the kalman filter library
 *
 */
namespace kafi {
 
/** \brief A templated EKF class with static matrix sizes
 * 
 * Template arguments:
 * * `N`  = state dimensions
 * * `M`  = sensor dimensions
 * 
 * See examples at [tests/kafi_tests.cc](../../tests/kafi_tests.cc)
 */
template<size_t N  // state  dimensions (N x 1)
       , size_t M> // sensor dimensions (M x 1)
class kafi {

    // typenames
    public:
        //! self type for conciseness
        using self_t     = kafi<N,M>;
        //! copied typename for conciseness
        using nx1_vector = typename jacobian_function<N,M>::nx1_vector;
        //! copied typename for conciseness
        using mx1_vector = typename jacobian_function<N,M>::mx1_vector;
        //! copied typename for conciseness
        using mxn_matrix = typename jacobian_function<N,M>::mxn_matrix;
        //! copied typename for conciseness
        using nxm_matrix = typename jacobian_function<N,M>::nxm_matrix;
        //! copied typename for conciseness
        using mxm_matrix = typename jacobian_function<N,M>::mxm_matrix;
        //! copied typename for conciseness
        using nxn_matrix = typename jacobian_function<N,M>::nxn_matrix;
        //! copied typename for conciseness
        /** \brief Shorthand for a useful return type for the kalman filter
         *  * `const nx1_vector & = std::get<0>(x)` = state              
         *  * `const nxn_matrix & = std::get<1>(x)` = prediction error   
         *  * `const nxm_matrix & = std::get<2>(x)` = gain               
         */
        using return_t   = std::tuple<const nx1_vector,
                                      const nxn_matrix,
                                      const nxm_matrix>;

    // constructors
    public:

        /** \brief Default constructor
         * 
         * Template arguments:
         * * `N`  = state dimensions
         * * `M`  = sensor dimensions
         * 
         * Arguments:
         * * `const  jacobian_function< N, N > f`: state transision function with their jacobian
         * * `const  jacobian_function< N, M > h`: prediction scaling function with their jacobian
         * *        `nx1_vector   starting_state`: initial state can be copied (kafi is owner)
         * *        `mx1_vector &    observation`: design decision, we need to explicitly initialize the observation, even if this will not be used in the first step(). Use `set_current_observation()`
         * * `const  nxn_matrix &  process_noise`: the *real world* noise
         * * `const  mxm_matrix &   sensor_noise`: the sensor covariance noise matrix
         *
         *  Initializing `prediction_error` to identity matrix via util::create_identity<N, blaze::rowMajor>()
         */
        kafi(const jacobian_function<N,N>   f
           , const jacobian_function<N,M>   h
           ,       nx1_vector               starting_state
           , const nxn_matrix             & process_noise
           , const mxm_matrix             & sensor_noise)
        : kafi<N,M> (std::move(f)
                   , std::move(h)
                   , starting_state
                   , process_noise
                   , sensor_noise
                   , util::create_identity<N, blaze::rowMajor>())
        { }

        /**
         * \brief The same as the default constructor, but with custom `prediction error` initialization
         */ 
        kafi(const jacobian_function<N,N>   f
           , const jacobian_function<N,M>   h
           ,       nx1_vector               starting_state 
           , const nxn_matrix             & process_noise
           , const mxm_matrix             & sensor_noise
           , const nxn_matrix             & prediction_error)
        : _f(std::move(f))
        , _h(std::move(h))
        , _process_noise(process_noise)
        , _sensor_noise(sensor_noise)
        , _state(starting_state)
        , _f_jacobian_temp(0)
        , _h_jacobian_temp(0)
        , _h_temp(0)
        , _prediction_count(0)
        , _update_count(0)
        , _gain(0)
        , _prediction_error(prediction_error)
        , _identity(util::create_identity<N, blaze::rowMajor>())
        , _new_data_available(false)
        { }

        //! Copy constructor is deleted because kafi owns multiple different potentially big matrices
         kafi(const self_t & other) = delete;
        //! Move constructor is deleted because kafi is probably not movable \todo Test moveability
        kafi(const self_t && other) = delete;

    // methods
    public:
        /**
         * To make the kafi concurrent, this function may be given via the constructor,
         * so this class is responsible for getting the observation
         *
         * Every call to this function is assumed to fill new, previously unknown observation
         * This equality check of observation_t and observation_t-1 is delegated to the caller
         *
         * Modifying:
         *     * `_observation`
         *     * `_new_data_available`
         *
         */
        void set_current_observation(std::shared_ptr<mx1_vector> observation)
        {
            _observation = observation;
            _new_data_available = true;
        }

        /**\brief Main function that runs the Kalman Filter based on new or old observation, and apply the prediction and update step
         *
         * Modifying:
         *     * `_gain`
         *     * `_state`
         *     * `_prediction_error`
         *
         * Return:
         *     * tuple of
         *         - state
         *         - prediction_error
         *         - gain
         */
        return_t step()
        {
            apply_prediction();
            if (new_data_available())
            {
                apply_update();
            }
            
            //print_state_to(std::cerr);
            DEBUG_MSG_KAFI(*this);
            return std::make_tuple(_state, _prediction_error, _gain);
        }
        /** \brief Overloading stream operator for logging purposes
         *
         * Might look like this:
         * ```
         * Kafi:
         *   Update      # calls: 1
         *   Predictions # calls: 1
         *  [S] _state:
         * (      19.6226 )
         * ============================
         *  [O] _observation:
         * (       18.625 )
         * (           20 )
         * ============================
         *  [P] _prediction_error:
         * (     0.245255 )
         * ============================
         *  [G] _gain:
         * (     0.383212     0.383212 )
         * ============================
         * ```
         */
        friend std::ostream & operator<<(std::ostream& stream, const self_t & rhs)
        {
            std::shared_ptr<mx1_vector> o  = rhs._observation.lock();
            std::string line = "============================\n";
            stream << "Kafi:\n"
                   << "  Update      # calls: "   << rhs._update_count     << '\n'
                   << "  Predictions # calls: "   << rhs._prediction_count << '\n'
                   << " [S] _state:\n"            << rhs._state            << line
                   << " [O] _observation:\n"      << (*o)                  << line
                   << " [P] _prediction_error:\n" << rhs._prediction_error << line
                   << " [G] _gain:\n"             << rhs._gain             << line; 
            return stream;
        }

        /** print helper for conciseness
         */
        void print_state_to(std::ostream & stream)
        {
            stream << *this;
        }

    //! Private methods
    private:
        /** \brief A check if the flag `_new_data_available` is true and flips it 
         */ 
        bool new_data_available()
        {   
            if (_new_data_available) {
                _new_data_available = false;
                return true;
            } else {
                return false;
            }
        }

        /** \brief Applying the prediction formulae
         * 
         * Modifying:
         *     * `_prediction_error`
         *     * `_state`
         *     * `_prediction_count`
         */ 
        void apply_prediction()
        {   
            // Using some zero cost abstraction renaming for mathematical understanding
            const nxn_matrix & P = _prediction_error;
            const nxn_matrix & Q = _process_noise;
            const nxn_matrix & F = _f.jacobian(_state, _f_jacobian_temp);

            _prediction_error = F * P * blaze::trans(F) + Q; 
            _f(_state, _state);
            _prediction_count++;
        }

        /** \brief Applying the update formulae
         * 
         * **Invariant**:
         *     * `_observation` has to be initialized, implemented through kafi::new_data_available()
         *
         * Modifying:
         *     * `_gain `
         *     * `_state`
         *     * `_prediction_error`
         *     * `_update_count`
         *     * `_h_temp`
         *
         * \todo cache `trans(H)`
         */
        void apply_update()
        {
            // Using zero cost abstraction renaming for mathematical understanding
            _h(_state, _h_temp);
            const mx1_vector & h  = _h_temp;
            const mxn_matrix & H  = _h.jacobian(_state, _h_jacobian_temp);
            const nxn_matrix & P  = _prediction_error;
            const mxm_matrix & cN = _sensor_noise;
            const nxn_matrix & I  = _identity;
            const nx1_vector & s  = _state;
            std::shared_ptr<mx1_vector> o  = _observation.lock();
                  nxm_matrix & G  = _gain;

            _gain  = P * blaze::trans(H) * blaze::inv(H * P * blaze::trans(H) + cN);
            _state = s + G * (*o - h);
            _prediction_error = (I - G * H) * P;
            _update_count++;
        }

    // member
    public:
        // functions with their respective preallocated resources

        //! state transition function
        const jacobian_function<N,N> _f;
        //! preallocated jacobian matrix space for `_f`
              nxn_matrix _f_jacobian_temp;
        //! prediction scaling function
        const jacobian_function<N,M> _h;
        //! preallocated vector space for `_h`
              mx1_vector _h_temp;
        //! preallocated jacobian matrix space for `_h`
              mxn_matrix _h_jacobian_temp;

        // const matrices
        //! `Q` (covariance of real world)
        const nxn_matrix               _process_noise;
        //! `cN` (covariance of sensors)
        const mxm_matrix               _sensor_noise;
        //! `I` (identity matrix)
        const nxn_matrix               _identity;

        //       matrices
        //! `s_t` (at time `t`), used as the preallocated vector space of `_f`
              nx1_vector               _state;
        //! `o_t` (reference, the caller is responsible for the allocation)
        std::weak_ptr< mx1_vector >    _observation;
        //! `P_t` 
              nxn_matrix               _prediction_error;
        //! `G_t`
              nxm_matrix               _gain;
        //! used to run the kafi::apply_update() function, changed in kafi::new_data_available()
              bool                     _new_data_available;
        // logging
        //! used for logging purposes, tracks how often kafi::apply_prediction() was run
              size_t                   _prediction_count;
        //! used for logging purposes, tracks how often kafi::apply_update() was run 
              size_t                   _update_count;
};

} // namespace kafi

/*! @} End of Doxygen Groups*/
#endif // KAFI_H
