Matrix Multiply
===

The matrix multiplication is the most fundamental operation in linear algebra.

$$
C = A \times B
$$

$$
c_{ij} = \sum_{k} a_{i,k}b_{k,j}
$$

Functions
---

```C
bool Matrix_Multiply(int32_t m, int32_t n, int32_t l,
                     double *a, double *b, double *res);
// if CUDA is available
bool Matrix_Multiply_cuda(int64_t m, int64_t n, int64_t l,
                          double *a, double *b, double *res);
```

If the functions succeed, they will return `true`.

Note: the functions will not overwrite the input `a` and `b`.