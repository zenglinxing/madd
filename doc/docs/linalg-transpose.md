Transpose
===

Matrix Transpose means the matrix of shape $m\times n$ will be converted into shape of $n\times m$.

For a complex number matrix, there is an operation called 'Hermitian transpose' or 'conjugate transpose'.
That is to say, the matrix is transposed, and the numbers become its conjugate.

Functions
---

```C
bool Matrix_Transpose(uint64_t m, uint64_t n, double *matrix);
/* for complex number matrix */
bool Matrix_Transpose_c64(uint64_t m, uint64_t n, Cnum *matrix);
/* Hermitian transpose */
bool Matrix_Hermitian_Transpose_c64(uint64_t m, uint64_t n, Cnum *matrix);
```

`Matrix_Transpose` applies *outplace* mode for transpose by default, thus it will require extra memory.
You may worry the transpose operation may cost a large amount of memory space if the matrix is tremendous.
However, if the extra memory space couldn't be allocated, `Matrix_Transpose` will automatically switches to *inplace* mode.