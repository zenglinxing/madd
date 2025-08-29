/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

#define N 3
#define _N 3

int main(int argc,char *argv[])
{
    int i;
    double a[_N+N+1]={-1.,-2.,-3.,4.,3.,2.,1.},x,value,log_coefficient;
    Poly1d poly = Poly1d_Init(N,_N,a), ipoly=Poly1d_Create(N+1,_N-1);
    log_coefficient=Poly1d_Integrate(&poly, &ipoly);
    for (i=ipoly._n; i>0; i--){
        printf("-%d\t%f\n", i, ipoly.a[-i]);
    }
    for (i=0; i<N+2; i++){
        printf("%d\t%f\n", i, ipoly.a[i]);
    }
    printf("log coefficient\t%f\n", log_coefficient);
    printf("_n=%u\n", poly._n);
    Poly1d_Free(&poly);
    Poly1d_Free(&ipoly);
    return 0;
}
