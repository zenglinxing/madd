/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include"madd.h"

struct func_param{
    double offset;
    double multi;
};

double func1(double x, void *other_param)
{
    struct func_param *param = (struct func_param*)other_param;
    return 1 + param->offset;
}

double func2(double x, void *other_param)
{
    struct func_param *param = (struct func_param*)other_param;
    return (x + param->offset) * param->multi;
}

int main(int argc, char *argv[])
{
    int n_int = 15, i;
    if (argc >= 2){
        n_int = atoi(argv[1]);
    }

    double *x_int = (double*)malloc(n_int*sizeof(double)), *w_int = (double*)malloc(n_int*sizeof(double));
    Integrate_Gauss_Laguerre_xw(n_int, x_int, w_int);
    printf("x points & weights\n");
    for (i=0; i<n_int; i++){
        printf("%llu\t%f\t%f\n", i, x_int[i], w_int[i]);
    }

    printf("====\n1 * e^-x\n0 -> infty\n====\n");
    struct func_param param1 = {.offset = 0};
    double res1 = Integrate_Gauss_Laguerre(func1, 0, 1, n_int, &param1);
    printf("res = %f\n", res1);

    printf("====\n2 * e^-x\n0 -> infty\n====\n");
    struct func_param param2 = {.offset = 1};
    double res2 = Integrate_Gauss_Laguerre(func1, 0, 1, n_int, &param2);
    printf("res = %f\n", res2);

    printf("====\ne^-2x\n0 -> infty\n====\n");
    struct func_param param5 = {.offset = 0};
    double res5 = Integrate_Gauss_Laguerre(func1, 0, 2, n_int, &param5);
    printf("res = %f\n", res5);

    printf("====\nx * e^-x\n0 -> infty\n====\n");
    struct func_param param3 = {.offset = 0, .multi = 1};
    double res3 = Integrate_Gauss_Laguerre(func2, 0, 1, n_int, &param3);
    printf("res = %f\n", res3);

    printf("====\n(x - 1) * e * e^-x = (x - 1) * e^-(x - 1)\n1 -> infty\n====\n");
    struct func_param param4 = {.offset = -1, .multi = exp(1)};
    double res4 = Integrate_Gauss_Laguerre(func2, 1, 1, n_int, &param4);
    printf("res = %f\n", res4);

    free(x_int);
    free(w_int);
    return 0;
}