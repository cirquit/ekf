# Extended Kalman Filter

An implementation of an [EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter) in C++ with `O(1)` runtime complexity.

Based on the [excellent tutorial by Simon D. Levy](https://home.wlu.edu/~levys/kalman_tutorial/).

---

Inteded to use on embedded systems. Uses [Blaze](https://bitbucket.org/blaze-lib/blaze/overview)
for linear algebra.

Example usage may be found in [tests/kafi_tests.cc](tests/kafi_tests.cc).

No automatic derivation, but also the *state* and *prection scaling* are defined as [std::function](https://en.cppreference.com/w/cpp/utility/functional/function), not as matricies. You can define **non-linear** transformations by hand.

---

If you see this on github, it's only a mirror of our internal [municHMotorsport](https://www.munichmotorsport.de/) gitlab repository. The repository name may not match in the following build instructions.

---

### Build

```bash
> git clone://gitlab.munichmotorsport.com/fsd-software/Kalman-Filter
> cd Kalman-Filter
> mkdir build && cd build
> cmake .. 
> make -j
> ./tests/kafi_tests
```

Trigger tests (default `ON`):

```bash
> cmake .. -DENABLE_TESTS_KAFI=OFF
```

Change debug level (default: 2 ~ everything will be printed to `cerr`):

```bash
> cmake .. -DDEBUG_LEVEL_KAFI=[0,1,2]
```

Enable optimizations (default `OFF`):

```bash
> cmake .. -DENABLE_OPTIMIZATIONS_KAFI=ON
```

### Installation (cmake only)

##### Subdirectory

You can easily reference the headers by hand and link to the `kafi` library in your `CMakeLists.txt` if you have this repository as a subdirectory of your main repository. Add this to your `CMakeLists.txt`:

```cmake
target_link_libraries(${your-awesome-executable} ${your-awesome-library} kafi )
```

##### System level installation

If you want a system-level installation just type `make install` in your `build` directory. Then you need to add the following to your own code `CMakeLists.txt`:

```cmake
find_package(kafi version 1.0 REQUIRED)
target_link_libraries(${your-awesome-executable} ${your-awesome-library} kafi )
```

To use the library, include this header and check out the [documentation](documentation)

```c++
#include <kafi-1.0/kafi.h>
```


### Basic runthrough of the API

Explanation of the first test in [tests/kafi_tests.cc](tests/kafi_tests.cc).

We try to estimate the temperature based on two uncorrelated thermometers.

* `N` - this is the state dimension. We estimate only a single temperature, hence `N = 1`
* `M` - this is the sensor dimension. We have two thermometers, hence `M = 2`

We use multiple typenames to avoid `auto` in the examples.

---

```c++
// state transition
kafi::jacobian_function<N,N> f(
    std::move(kafi::util::create_identity_jacobian<N,N>()));

// prediction scaling (state -> observations)
kafi::jacobian_function<N,M> h(
    std::move(kafi::util::create_identity_jacobian<N,M>()));
```

First we have to create the `f` function, which is the **state transition**. This means it takes a vector of `N` element (the *state*) and returns the  modified state based on the knowledge of the environment. If we would've known any *control parameters* this might look different.

Therefore we can simply have the identity function as the **state transition**. This looks slightly convoluted, because we also need the derivative of this function, which is also automatically generated for the identity function based on the size `N`.

The same goes for the `h` function, which scales the state (`N`) to the observation dimension (`M`). The function broadcasts automatically first value of the state vector to all observation dimensions. In our example:

```python
state = [20] # estimated C°
observations = [21.1, 21.5] # noisy C°

[20, 20] = h(state)

# do stuff with observations and h(state)
``` 

---

```c++
// given by our example, read as "the real world temperature changes are 0.22° (0.22^2 =~ 0.05)"
nxn_matrix process_noise( { { 0.05 } } );
// given by our example, read as "both temperature sensors fluctuate by 0.8° (0.8^2 = 0.64)"
mxm_matrix sensor_noise( { { 0.64, 0    }
                         , { 0,    0.64 } });
// given by our example, read as "first time we measured temperature, we got these values"
std::shared_ptr< mx1_vector > first_observation = std::make_shared< mx1_vector >(
                                                       mx1_vector({ { 18.625 } 
                                                                  , { 20     } }));
```

We create the process noise and the sensor noise as blaze matricies. The first observation is packed in a [`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr) because the EKF **does not own** it.

We share it with a shared pointer, the EKF uses it, but the ownership never transitions, because **we don't want to allocate** unnecessarly. The callee has to manage the allocation space for the observations.

---

```c++
kafi.set_current_observation(first_observation);
// run the estimation
return_t   result          = kafi.step();
nx1_vector estimated_state = std::get<0>(result);
```

The `step()` function applies the *prediction* step and *update* step consecutively.

`result_t` consists of:

```c++
const nx1_vector & estimated_state  = std::get<0>(result)
const nxn_matrix & prediction error = std::get<1>(result)
const nxm_matrix & gain             = std::get<2>(result)
```

To access the elements, use the [blaze matrix access reference](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!element-access).

```c++
double temperature = estimated_state(0,0)
```

### Documentation

Created with doxygen (with Markdown support)
* [PDF](documentation/latex/refman.pdf)
* [HTML](documentation/html/index.html)

Update documentation:

```bash
> cd Kalman-Filter
> doxygen doxygen.config
> cd documentation/latex
> make
```
### Dependencies

* [fast-cpp-csv-parser](https://github.com/ben-strasser/fast-cpp-csv-parser)
    - header only, used for tests (already included)
* [blaze](https://bitbucket.org/blaze-lib/blaze/overview) (tested on ubuntu 16.04)
```bash 
> wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.3.tar.gz
> tar -xvf blaze-3.3.tar.gz
> sudo apt-get install libopenblas-dev
> sudo apt-get install libboost-all-dev
> cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
> sudo make install
```