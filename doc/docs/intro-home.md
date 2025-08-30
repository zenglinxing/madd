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

* C & C++ compiler (at least C99 support)
* [CMake](https://cmake.org)
* [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)

Your machine should be binary 64-bit, supporting 64-bit float number (double precision).