Introduction
===

$$
\int_{a}^{b} f(x) dx
$$

Numerical integration approximates the definite integral through numerical methods.
There are many integration algorithms. Madd implements the major algorithms suitable for most of cases.

Generally, you should prepare the function (integrand).

```C
double func(double x, void *other_param);
```

`other_param` allows you to custom the function.

One-Dimension Integrand
---

Suppose you choose an integrate function:

```C
// this function does not exist in Madd!
double Integrate(double func(double, void*), double xmin, double xmax,
                 uint64_t n_int, void *other_param);
```

Supposing the integration problem

$$
\int_{0}^{1} x \sin (x) dx
$$

```C
#include<math.h>
#include<stdio.h>
#include<stdint.h>
#include<madd.h>

double func(double x, void *param)
{
    return x * sin(x);
}

int main(int argc, char *argv[])
{
    uint64_t n_int = 20;
    double xmin = 0, xmax = 1;
    double res = Integrate(func, xmin, xmax, n_int, NULL);
    printf("result:\t%f\n", res);
    return 0;
}
```

Multi-Dimension Integrand
---

Although Madd only provides 1-D integrate function, you can realize multi-dimension integral by using the `other_param`.

Suppose you choose an integrate function:

```C
// this function does not exist in Madd!
double Integrate(double func(double, void*), double xmin, double xmax,
                 uint64_t n_int, void *other_param);
```

We have an integration problem.

$$
\int_{0}^{1} dx \int_{-x}^{x} dy \ e^{xy}
$$

You can write your code in this way.

```C
#include<stdio.h>
#include<math.h>
#include<stdint.h>
#include<madd.h>

typedef struct{
    double x;
    uint64_t n_int;
} Param;

double func_y(double y, void *param_)
{
    Param *param = (Param*)param_;
    double x = param->x;
    return exp(x * y);
}

double func_x(double x, void *param_)
{
    Param *param = (Param*)param_;
    // integrate y: from -x -> x
    double res = Integrate(func_y, -x, x, param->x_int, param_);
    return res;
}

int main(int argc, char *argv[])
{
    uint64_t n_int = 20;
    Param param = {.n_int = n_int};
    double res = Integrate(func_x, 0, 1, n_int, &param);

    printf("integrate result:\t%f\n", res);
    return 0;
}
```