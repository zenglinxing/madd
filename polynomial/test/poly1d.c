/* coding: utf-8 */
#include<stdio.h>
#include<stdbool.h>
#include"madd.h"

#define N 3
#define _N 3

double func(double x, void *other_param)
{
    return Poly1d_Value(x, (Poly1d*)other_param);
}

int main(int argc,char *argv[])
{
    madd_error_keep_print = true;
    int i;
    double a[_N+N+1]={-1.,-2.,-3.,4.,3.,2.,1.}, x, value;

    printf("====\nTest Init\n====\n");
    Poly1d poly = Poly1d_Init(N, _N, a);
    printf("init\n");
    x = 1.;
    value = Poly1d_Value(x, &poly);
    printf("value = %f\n", value);
    x = 2.;
    value = Poly1d_Value(x, &poly);
    printf("value = %f\n", value);

    printf("====\nTest Derivative\n====\n");
    Poly1d dpoly=Poly1d_Create(N-1, _N+1);
    Poly1d_Derivative(&poly, &dpoly);
    for (i=dpoly._n; i>0; i--){
        printf("-%d\t%f\n", i, dpoly.a[-i]);
    }
    for (i=0; i<N; i++){
        printf("%d\t%f\n", i, dpoly.a[i]);
    }
    printf("_n=%u\n", poly._n);

    printf("====\nTest Derivative\n====\n");
    Cnum ca[_N+N+1];
    Cnum cx;
    ca[0] = Cnum_Value(-1., 0.);
    ca[1] = Cnum_Value(-2., 0.);
    ca[2] = Cnum_Value(-3., 0.);
    ca[3] = Cnum_Value(4.,  0.);
    ca[4] = Cnum_Value(3.,  0.);
    ca[5] = Cnum_Value(2.,  0.);
    ca[6] = Cnum_Value(4.,  0.);
    Poly1d_c cpoly = Poly1d_Init_c(N, _N, ca), cdpoly = Poly1d_Create_c(N-1, _N+1);
    Poly1d_Derivative_c(&cpoly, &cdpoly);
    for (i=cdpoly._n; i>0; i--){
        printf("%d\t%f\t%f\n", i, cdpoly.a[-i].real, cdpoly.a[-i].imag);
    }
    for (i=0; i<N-1; i++){
        printf("%d\t%f\t%f\n",i, cdpoly.a[i].real, cdpoly.a[i].imag);
    }

    printf("====\nTest Integrate\n====\n");
    Poly1d ipoly=Poly1d_Create(N+1,_N-1);
    double log_coefficient=Poly1d_Integrate(&poly, &ipoly);
    for (i=ipoly._n; i>0; i--){
        printf("-%d\t%f\n", i, ipoly.a[-i]);
    }
    for (i=0; i<N+2; i++){
        printf("%d\t%f\n", i, ipoly.a[i]);
    }
    printf("log coefficient\t%f\n", log_coefficient);
    printf("_n=%u\n", poly._n);

    printf("====\nTest NIntegrate\n====\n");
    double xmin=2., xmax=4.;
    double res1,res2;
    res1 = Integrate_Simpson(func, xmin, xmax, 29, &poly);
    printf("res real=%f\n",res1);
    res2 = Poly1d_NIntegrate(&poly, xmin, xmax);
    printf("res test=%f\n", res2);

    /* free */
    Poly1d_Free(&poly);
    Poly1d_Free_c(&cpoly);
    Poly1d_Free_c(&cdpoly);
    Poly1d_Free(&ipoly);
    return 0;
}
