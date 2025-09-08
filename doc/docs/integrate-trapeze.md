Trapezoidal Integral
===

```C
double Integrate_Trapeze(double func(double, void*), double xmin, double xmax,
                         uint64_t n_int, void *other_param);
```

Math
---

Trapezoidal integral uniformly separates the integrating range [`xmin`, `xmax`] into `n_int` parts.

$$
\int_{xmin}^{xmax} f(x) dx = \frac{xmax - xmin}{n\_int} \sum_{i=1}^{n\_int} \frac{f(x_{i-1}) + f(x_{i})}{2},
$$

where
$$
f(x_{i}) = xmin + i * gap
$$
$$
gap = \frac{xmax - xmin}{n\_int}
$$