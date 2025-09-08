/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<errno.h>
#include<signal.h>

#include"madd.h"

void signal_handler(int signal)
{
    printf("errno:\t%d\n", errno);
}

int main(int argc, char *argv[])
{
    signal(SIGABRT, signal_handler);
    uint32_t n_pos=4, n_neg=4, n_order = 2;
    if (argc >= 2){
        n_pos = atoi(argv[1]);
    }
    if (argc >= 3){
        n_neg = atoi(argv[2]);
    }
    if (argc >= 4){
        n_order = atoi(argv[3]);
    }
    /* poly */
    Poly1d poly = Poly1d_Create(n_pos, n_neg);
    printf("poly.mem pointer:\t%p\n", poly.mem);
    int32_t i;
    for (i=0; i<=n_pos; i++){
        poly.a[i] = 1;
    }
    for (i=1; i<=n_neg; i++){
        poly.a[-i] = 1;
    }
    /* dpoly */
    Poly1d dpoly = Poly1d_Derivative_N_order(&poly, n_order);
    printf("dpoly.mem pointer:\t%p\n", dpoly.mem);
    uint32_t nd_pos=(n_order>n_pos) ? 0 : n_pos-n_order, nd_neg=n_neg+n_order;
    double a_poly, a_dpoly;
    for (i=0; i<=n_pos; i++){
        a_poly = poly.a[n_pos-i];
        a_dpoly = (n_pos-i > dpoly.n) ? 0 : dpoly.a[n_pos-i];
        printf("order= %d\t%f\t%f\n", n_pos-i, a_poly, a_dpoly);
    }
    for (i=1; i<=dpoly._n; i++){
        a_poly = (i<=poly._n) ? poly.a[-i] : 0;
        a_dpoly = dpoly.a[-i];
        printf("order= %d\t%f\t%f\n", -i, a_poly, a_dpoly);
    }
    printf("poly.mem\t%f\n", poly.mem[0]);
    printf("dpoly.mem\t%f\n", dpoly.mem[0]);
    /* free */
    printf("poly.mem pointer:\t%p\n", poly.mem);
    Poly1d_Free(&poly);
    //free(poly.mem);
    printf("poly freed\n");
    printf("dpoly.mem pointer:\t%p\n", dpoly.mem);
    //free(dpoly.mem);
    Poly1d_Free(&dpoly);
    printf("dpoly freed\n");
    return 0;
}
