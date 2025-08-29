/* coding: utf-8 */
#include<stdio.h>
#include<stdbool.h>
#include"madd.h"

#define N 3
#define _N 3

int main(int argc,char *argv[])
{
    madd_error_keep_print = true;
    double a[_N+N+1]={-1.,-2.,-3.,4.,3.,2.,1.}, x, value;
    Poly1d poly = Poly1d_Init(N, _N, a);
    printf("init\n");
    x = 1.;
    value = Poly1d_Value(x, &poly);
    printf("value = %f\n", value);
    x = 2.;
    value = Poly1d_Value(x, &poly);
    printf("value = %f\n", value);
    Poly1d_Free(&poly);
    return 0;
}
