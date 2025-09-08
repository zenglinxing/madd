Simpson Integral
===

```C
double Integrate_Simpson(double func(double,void*),
                         double xmin, double xmax,
                         uint64_t n_int,void *other_param);
```

Math
---

$$
\int_{xmin}^{xmax} f(x) dx = \frac{xmax - xmin}{n\_int} \sum_{i=1}^{n\_int} \frac{f(x_{i-1}) + 4 * f(\frac{x_{i-1}+x_{i}}{2}) + f(x_{i})}{6},
$$

where
$$
f(x_{i}) = xmin + i * gap
$$
$$
gap = \frac{xmax - xmin}{n\_int}
$$