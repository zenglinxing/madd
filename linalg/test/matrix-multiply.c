/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include"madd.h"

uint64_t m = 2, n = 3, l = 4;

int main(int argc, char *argv[])
{
    uint64_t i, j;
    double *a = (double*)malloc(m*l*sizeof(double));
    double *b = (double*)malloc(l*n*sizeof(double));
    double *c = (double*)malloc(m*n*sizeof(double));
    double *c_naive = (double*)malloc(m*n*sizeof(double));
    printf("matrix a\n");
    for (i=0; i<m; i++){
        for (j=0; j<l; j++){
            a[i*l+j] = i*l + j;
            printf("%f\t", a[i*l+j]);
        }
        printf("\n");
    }
    printf("matrix b\n");
    for (i=0; i<l; i++){
        for (j=0; j<n; j++){
            b[i*n+j] = i*l + j;
            printf("%f\t", b[i*n+j]);
        }
        printf("\n");
    }
    // matrix c
    for (i=0; i<m; i++){
        for (j=0; j<n; j++){
            c[i*n+j] = c_naive[i*n+j] = 0;
        }
    }

    Matrix_Multiply(m, n, l, a, b, c);
    Matrix_Multiply_Naive(m, n, l, a, b, c_naive);

    printf("matrix c\n");
    bool flag_different = false;
    for (i=0; i<m; i++){
        for (j=0; j<n; j++){
            printf("%f", c[i*n+j]);
            if (c[i*n+j] != c_naive[i*n+j]){
                flag_different = true;
                printf("(d)");
            }
            printf("\t");
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    free(c_naive);

    if (flag_different){
        exit(EXIT_FAILURE);
    }

    return 0;
}