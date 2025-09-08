#include<math.h>
#include<stdio.h>
#include<stdint.h>
#include"madd.h"

uint64_t n_order = 8;

double func(double x,void *other_param)
{
    return x+x*x;
}

double func2(double x,void *other_param)
{
    return x+exp(x);
}

int main(int argc, char*argv[])
{
    if (argc >= 2){
        n_order = atoi(argv[1]);
    }
    /*double xmin=0, xmax=20;*/
    double xmin=-1, xmax=1;
    double res_Simpson=Integrate_Simpson(func, xmin, xmax, 7, NULL), res_Gauss=Integrate_Gauss_Legendre(func, xmin, xmax, n_order, NULL);
    printf("Simpson:\t%f\n", res_Simpson);
    printf("Gauss:\t%f\n", res_Gauss);

    xmin=-1;
    xmax=1;
    res_Simpson=Integrate_Simpson(func2, xmin, xmax, 7, NULL);
    res_Gauss=Integrate_Gauss_Legendre(func2, xmin, xmax, n_order, NULL);
    printf("Simpson:\t%f\n", res_Simpson);
    printf("Gauss:\t%f\n", res_Gauss);
    return 0;
}
