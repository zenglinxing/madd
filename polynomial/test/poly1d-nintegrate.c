/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

#define N 3
#define _N 3

double func(double x, void *other_param)
{
    return Poly1d_Value(x, (Poly1d*)other_param);
}

int main(int argc,char *argv[])
{
    double a[_N+N+1]={-1.,-2.,-3.,4.,3.,2.,1.}, xmin=2., xmax=4.;
    printf("initializing...\n");
    Poly1d poly=Poly1d_Init(N, _N, a);
    printf("initialized\n");
    double res1,res2;
    res1=Integrate_Simpson(func, xmin, xmax, 29, &poly);
    printf("res real=%f\n",res1);
    res2=Poly1d_NIntegrate(&poly, xmin, xmax);
    printf("res test=%f\n", res2);
    Poly1d_Free(&poly);
    return 0;
}
