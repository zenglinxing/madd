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
bool Matrix_Multiply(int m, int n, int l,
                     double *a, double *b, double *res);
// if CUDA is available
bool Matrix_Multiply_cuda(int64_t m, int64_t n, int64_t l,
                          double *a, double *b, double *res);
```

If the functions succeed, they will return `true`.

Note: the functions will not overwrite the input `a` and `b`.