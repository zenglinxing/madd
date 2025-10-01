Linear Algebra
===

Linear algebra (linalg) should be one of the most vital parts of a numerical library, since many functions depend on it.
The linalg of Madd is based on OpenBLAS, so you should build OpenBLAS before building Madd.
OpenBLAS is a very sofisticated linear algebra library for efficient computing, and the goal of Madd is to provide a comprehensive way to call the functions of OpenBLAS.

One thing you should notice is that the matrices or vectors input to Madd linalg functions may be overwritten even if they are just the input parameters.
So backup your input matrices and vectors before calling.

Another thing you may notice is that the integer type of Madd linalg is usually `int` rather than `uint64_t` in other functions.
This is because OpenBLAS uses `lapack_int` which is actually `int` type for functions' inputs.
In this case, you may suffer from problem of data length.

All matrices in Madd is supposed to be **row-major**.
You may notice the matrices by lapack is column-major.
Row-major is much intuitively friendly for human, so I adopt it when developing.