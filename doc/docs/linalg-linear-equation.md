Linear Equation
===

In a linear equation system, it is supposed to have a $n\times n$ matrix $A$, a coefficient vector $b$ and a $n$ vector $x$ to be solved,

$$
Ax = b
$$

Functions
---

```C
bool Linear_Equations(int n, double *eq, int n_vector, double *vector);
// if CUDA is available
bool Linear_Equations_cuda(int n, double *eq, int n_vector, double *vector);
```

Note the `eq` will be overwritten.
You should set the coefficient vector $b$ to `vector`, and the result will be put into `vector` as well.
If the function failed, it will return `false`.