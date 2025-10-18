Gauss-Legendre Quadrature
===

```C
double Integrate_Gauss_Legendre(double func(double, void *), double xmin, double xmax,
                                int32_t n_int, void *other_param);
```

Gauss-Legendre is a very efficient quadrature method. With only `n_int` points, the precision improved to $2*n\_int-1$ order. If your integrand is **smooth** and **continuous**, Gauss-Legendre quadrature is adequate.

Optimized Using
---

Gauss-Legendre quadrature has a time comsuming process, calculating the integrating points `x_int` and the weights `w_int`. You can calculate them first, and then apply Gauss-Legendre quadrature.

For example, you call the function in this way.

```C
    uint64_t n_int = 20;
    double xmin = -2, xmax = 2;
    Integrate_Gauss_Legendre(func, xmin, xmax, n_int, NULL);
```

If `Integrate_Gauss_Legendre` is called multiple times, and you won't change `n_int`, then you can integrate in this way.

```C
    uint64_t n_int = 20;
    // prepare x_int & w_int
    double *x_int = (double*)malloc(n_int*sizeof(double)), *w_int = (double*)malloc(n_int*sizeof(double));
    Integrate_Gauss_Legendre_x(n_int, x_int);
    Integrate_Gauss_Legendre_w_f32(n_int, x_int, w_int);

    // call Integrate_Gauss_Legendre_via_xw instead to bypass re-computing x_int & w_int
    Integrate_Gauss_Legendre_via_xw(func, xmin, xmax, n_int, NULL, x_int, w_int);

    free(x_int);
    free(w_int);
```

The first 5 parameters of `Integrate_Gauss_Legendre_via_xw` are the same as `Integrate_Gauss_Legendre`. The next 2 parameters are `x_int` and `w_int`.

Math Detail
---

Gauss-Legendre quadrature prepares `n_int` points of `x_int` (integral points in [-1, 1]) and `w_int` (weight points). The `n_int` points of `x_int` are the roots of Legendre polynomial.

$$
P_{n}(x) = 0
$$

The *i*-th weight point $w_{i}$ depends on the *i*-th root $x_{i}$.

$$
w_{i} = \frac{2}{(1-x_{i}^{2})(P_{n}(x_{i}))^{2}}
$$

Then the quadrature

$$
\int_{xmin}^{xmax} f(x) dx = \sum_{i=1}^{n\_int}f(x_{middle} + x_{i} \times x_{mod}) \times w_{i} \times x_{mod}
$$

where 

$$
x_{middle} = \frac{xmax + xmin}{2}
$$

$$
x_{mod} = \frac{xmax - xmin}{2}
$$

# How to Get the Roots of $P_{n}(x) = 0$

Construct a symmetric matrix, in which all other elements are 0, except the secondary diagonal.

```C
for (uint64_t i=1; i<n_int; i++){
    matrix[(i-1)*n_int + i] = matrix[i*n_int + i-1] = i/sqrt(4.*i*i-1);
}
```

Calculate the eigenvalues. The eigenvalues are the roots of $P_{n}(x) = 0$. Since there is only secondary diagonal in the matrix, you can call `dsteqr`/`ssteqr` in LAPACK to get the eigen values.