/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include"madd.h"

double func(double x,void *other_param)
{
    return x+x*x;
}

int main(int argc,char *argv[])
{
    int n_int = 10, i;
    if (argc > 1){
        n_int = atoi(argv[1]);
        if (n_int <= 0) n_int = 10;
    }
    double xmin=0, xmax=20;
    double *x_int = (double*)malloc(n_int*sizeof(double));
    double *w_int = (double*)malloc(n_int*sizeof(double));
    Integrate_Clenshaw_Curtis_x(x_int, n_int);
    Integrate_Clenshaw_Curtis_w(w_int, n_int);
    double res = Integrate_Clenshaw_Curtis_via_xw(func, xmin, xmax, n_int, NULL, x_int, w_int);
    for (i=0; i<n_int; i++){
        printf("x[%d]=%f\tw[%d]=%f\n", i, x_int[i], i, w_int[i]);
    }
    free(x_int);
    free(w_int);

    printf("res\t%f\n",res);
    return 0;
}