/* coding: utf-8 */
#include<stdio.h>
#define ENABLE_COMPLEX
#include"madd.h"

#define N 3
#define _N 3

int main(int argc,char *argv[])
{
    int i;
    Cnum a[_N+N+1];
    Cnum x,value;
    a[0] = Cnum_Value(-1., 0.);
    a[1] = Cnum_Value(-2., 0.);
    a[2] = Cnum_Value(-3., 0.);
    a[3] = Cnum_Value(4.,  0.);
    a[4] = Cnum_Value(3.,  0.);
    a[5] = Cnum_Value(2.,  0.);
    a[6] = Cnum_Value(4.,  0.);
    Poly1d_c poly = Poly1d_Init_c(N, _N, a), dpoly=Poly1d_Create_c(N-1, _N+1);
    Poly1d_Derivative_c(&poly, &dpoly);
    for (i=dpoly._n; i>0; i--){
        printf("%d\t%f\t%f\n", i, dpoly.a[-i].real, dpoly.a[-i].imag);
    }
    for (i=0; i<N-1; i++){
        printf("%d\t%f\t%f\n",i, dpoly.a[i].real, dpoly.a[i].imag);
    }
    Poly1d_Free_c(&poly);
    Poly1d_Free_c(&dpoly);
    return 0;
}
