/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

#define N 5

int main(int argc,char *argv[])
{
    printf("====\nTest -1 0 1\n====\n");
    printf("N=%d\n", N);
    int i;
    Poly1d poly = Poly1d_Create(N, 0);
    Special_Func_Legendre(&poly);
    for (i=0;i<N+1;i++){
        printf("%d\t%f\n", i, poly.a[i]);
    }
    double x=0.;
    printf("x=%f\tvalue=%f\n", x, Poly1d_Value(x, &poly));
    x=1.;
    printf("x=%f\tvalue=%f\n", x, Poly1d_Value(x, &poly));
    x=-1.;
    printf("x=%f\tvalue=%f\n", x, Poly1d_Value(x, &poly));
    Poly1d_Free(&poly);

    printf("====\nTest Iter\n====\n");
    int n2=N>>1;
    double coefficient=Special_Func_Legendre_Coefficient_First(N);
    printf("n=%d\t%f\n",N,coefficient);
    for (i=1; i<=n2; i++){
        coefficient = Special_Func_Legendre_Coefficient_Iter(N, i, coefficient);
        printf("n=%d\t%f\n", N-2*i, coefficient);
    }

    printf("====\nTest Value\n====\n");
    uint32_t n=6, n_v=100, i_v;
    double dx=2./(n_v-1), y1, y2;
    poly = Poly1d_Create(n, 0);
    Special_Func_Legendre(&poly);
    printf("poly coefficient:\n");
    for (i=0; i<=n; i++){
        printf("a%u=\t%f\n", i, poly.a[i]);
    }
    for (i_v=0; i_v<n_v; i_v++){
        x = -1 + dx*i_v;
        y1 = Poly1d_Value(x, &poly);
        y2 = Special_Func_Legendre_Value(n, x);
        printf("i=%u\t%f\t%f\t%f\n", i_v, x, y1, y2);
    }
    /* free */
    Poly1d_Free(&poly);
    return 0;
}
