Gauss-Legendre Quadrature
===

```C
double Integrate_Gauss_Legendre(double func(double, void *), double xmin, double xmax,
                                uint64_t n_int, void *other_param);
```

Gauss-Legendre is a very efficient quadrature method. With only `n_int` points, the precision improved to $2*n\_int-1$ order.

Optimized Using
---

Gauss-Legendre quadrature has a time comsuming process, calculating the integrating points `x_int` and the weights `w_int`. You can calculate them first, and then apply Gauss-Legendre quadrature.

For example, you call the function in this way.

```C
    uint64_t n_int = 20;
    double xmin = -2, xmax = 2;
    Integrate_Gauss_Legendre(func, xmin, xmax, n_int, NULL);
```

If `Integrate_Gauss_Legendre` is called multiple times, and you won't change `n_int`, then you can do it in this way.

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