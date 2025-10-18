Clenshaw-Curtis Quadrature
===

Clenshaw-Curtis quadrature is an effective method for smooth function.

It precomputes the integrate points `x_int` and `w_int`. However, computing these two is time consuming, since the calculation of `x_int` involves triangular functions and `w_int` needs FFT (the time complexity is $O(n \log n)$). So I recommend you to apply `Integrate_Clenshaw_Curtis_x`, `Integrate_Clenshaw_Curtis_w` and `Integrate_Clenshaw_Curtis_via_xw` respectively rather than `Integrate_Clenshaw_Curtis` directly if you need to use the quadrature multiple times with fixed `n_int`.

```C
bool Integrate_Clenshaw_Curtis_x(uint64_t n_int, double *x_int);
bool Integrate_Clenshaw_Curtis_w(uint64_t n_int, double *w_int);
double Integrate_Clenshaw_Curtis_via_xw(double func(double, void *), double xmin, double xmax,
                                        uint64_t n_int, void *other_param,
                                        double *x_int, double *w_int);
double Integrate_Clenshaw_Curtis(double func(double, void *), double xmin, double xmax,
                                 uint64_t n_int, void *other_param);
```