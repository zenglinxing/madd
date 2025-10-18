Matrix Inverse
===

The inverse of a $n\times n$ matrix $A$ is described as $A^{-1}$, which has properties:

* $AA^{-1} = A^{-1}A = I$
* if $det(A) \neq 0$, then $det(A^{-1}) = \frac{1}{det(A)}$

Functions
---

```C
bool Matrix_Inverse(int32_t n, double *matrix);
bool Matrix_Inverse_c64(int32_t n, Cnum *matrix);
```

If the functions fail, they will return `false`.