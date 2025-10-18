Determinant
===

For a matrix $A$

$$
A = \left[ \begin{matrix}
    a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
    a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n,1} & a_{n,2} & \cdots & a_{n,n}
\end{matrix} \right]
$$

the determinant of it is defined as

$$
det(A) = \sum_{P} P\left( \begin{matrix}
    1 & 2 & \cdots & n \\
    j_{1} & j_{2} & \cdots & j_{n}
\end{matrix} \right)
(-1)^{n_{P}}
\prod_{i=1}^{n} a_{i,j_{i}}
$$

where $P$ denotes the permutation of indices $j_{i}$, and $n_{P}$ refers to the exchanging times to restore the order of indices $j_{i}$ to $1,2,\cdots n$.

Functions
---

```C
bool Determinant(int32_t n, double *matrix, double *res);
bool Determinant_c64(int32_t n, Cnum *matrix, Cnum *res);
```

If the functions fail, they will return `false`.