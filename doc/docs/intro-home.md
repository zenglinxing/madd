Math Addition - Madd
===
Author: Lin-Xing Zeng

Email: jasonphysics@outlook.com

Introduction
---

Madd is a C library for numerical computation. Even though some source codes are C++, the interfaces are all C.

This project is built upon `CMake` and supports modern compilers, like GNU GCC, Microsoft Visual Studio, Intel OneAPI, and Clang. Even if your C compiler is not a main stream, compilation of Madd only demands C99 standard and a few C++ (C++ compiler is required), supposing your configure options in cmake are default ones. If you open the options like CUDA or multithread, C11, C++17 or later standards may be compulsory.

Dependencies & Requirements
---

Required:

* C & C++ compiler (at least C99 support)
* [CMake](https://cmake.org) - Building system of Madd.
* [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) - For linear algebra computations.

Optional:

* [thread_base](https://gitee.com/zenglinxing/thread_base) - My cross-platform multithread API.
* [CUDAToolkit](https://developer.nvidia.com/cuda-toolkit) - For some cuda functions.
* [Benchmark](https://github.com/google/benchmark) - Google benchmark

Your machine should be binary 64-bit, supporting 64-bit float number (double precision).

Third-Parties Licences
---

1. OpenBLAS - [BSD-3](https://opensource.org/licenses/BSD-3-Clause)

2. CUDAToolkit - [NVIDIA End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html)

3. Benchmark - [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)