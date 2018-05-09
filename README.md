
Extended Kalman Filter Library
============

### Build

#### Production build

```bash
> cd KaFi
> mkdir build && cd build
> cmake .. 
> make
> ./tests/kafi_tests
```

#### Test build

```bash
> cd KaFi
> mkdir build-debug && cd build-debug
> cmake .. -DDEBUG_MODE=1
> make
> ./tests/kafi_tests
```

### Documentation

Created with doxygen (with Markdown support)
* [PDF](documentation/latex/refman.pdf)
* [HTML](documentation/html/index.html)

Update documentation:

```bash
> cd KaFi
> doxygen doxygen.config
> cd documentation/latex
> make
```

### Parameters that need to be variable:

  *  `n` -> state dimension i.e. how many values we want to estimate  

        => applied to out case `n = 2` (car position in `X` and `Y`)

  *  `m` ->  sensor dimensions, i.e. how many sensor values do we have.  

        => applied to our case: velocity, acceleration, SLAM(?)

  *  `l` -> the amount of control dimensions, e.g how much information do we have to apply to the state.  

  *  *s_t* the estimated state is a vector of `(n x 1)` dimensions

    Our case: *s_t(x,y)*  *[x y]*



### Dependencies

* [fast-cpp-csv-parser](https://github.com/ben-strasser/fast-cpp-csv-parser)
    - header only (used for tests)

* [blaze](https://bitbucket.org/blaze-lib/blaze/overview)  
    - for ubuntu 16.04

```bash 
> wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.3.tar.gz
> tar -xvf blaze-3.3.tar.gz
> sudo apt-get install libopenblas-dev
> sudo apt-get install libboost-all-dev
> cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
> sudo make install
```

** otherwise (Mac OS) (unfinished)
```bash
    brew install boost
```
* optional, but needed for excellent performance 
    - a [BLAS](http://www.netlib.org/blas/)
    - and [LAPACK](http://www.netlib.org/lapack/)  
  
