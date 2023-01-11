# CytnxBenchmarks
Benchmarks for the Cytnx library.

## Adding and running benchmark test case
Make sure you have your Cytnx, Itensor and Google Benchmark installed and include them in the `Makefile` properly, then just add your own benchmark `.cpp` files in this directory (also specify them in the `Makefile`) and run `make` to generate the benchmark executable. Note that before running the benchmark executable you may want to specify the the MKL and OMP numthread, see `run.sh` for exmaple.  

## Note 
If you want to include both of the header file `<cytnx.hpp>` and `<itensor/all.h>`, and the itensor library is compiled in mkl platform, you may encounter some redefinition errors, please 
command out the lines 114~118 in the file : `<itensor path you installed>/itensor/tensor/lapack_wrap.h`
like:

```
//
//
// Intel MKL
//
//
#elif defined PLATFORM_mkl

#define ITENSOR_USE_CBLAS
#define ITENSOR_USE_ZGEMM

#include "mkl_cblas.h"
#include "mkl_lapack.h"
    namespace itensor {
    using LAPACK_INT = MKL_INT;
    using LAPACK_REAL = double;
    using LAPACK_COMPLEX = MKL_Complex16;

    //inline LAPACK_REAL& 
    //realRef(LAPACK_COMPLEX & z) { return z.real; }

    //inline LAPACK_REAL& 
    //imagRef(LAPACK_COMPLEX & z) { return z.imag; }
    }
```