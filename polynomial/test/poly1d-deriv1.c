/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

#define N 3
#define _N 3

int main(int argc,char *argv[])
{
    int i;
    double a[_N+N+1]={-1.,-2.,-3.,4.,3.,2.,1.},x,value;
    Poly1d poly = Poly1d_Init(N, _N, a), dpoly=Poly1d_Create(N-1, _N+1);
    Poly1d_Derivative(&poly, &dpoly);
    for (i=dpoly._n; i>0; i--){
        printf("-%d\t%f\n", i, dpoly.a[-i]);
    }
    for (i=0; i<N; i++){
        printf("%d\t%f\n", i, dpoly.a[i]);
    }
    printf("_n=%u\n", poly._n);
    Poly1d_Free(&poly);
    Poly1d_Free(&dpoly);
    return 0;
}
