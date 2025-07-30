/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    double a[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    Cnum b[4];
    int i;
    for (i=0; i<4; i++){
        b[i] = Cnum_Value(a[2*i], a[2*i+1]);
    }

    /* print */
    printf("a[8]\n");
    for (i=0; i<8; i++){
        printf("%d\t%f\n", i, a[i]);
    }
    printf("b[4]\n");
    for (i=0; i<4; i++){
        printf("%d\t%f + %f i\n", i, b[i].real, b[i].imag);
    }

    /* norm 1 */
    printf("norm1\n");
    double a1 = Norm1(8, a), b1 = Norm1_c64(4, b);
    printf("a norm1:\t%f\nb norm1:\t%f\n", a1, b1);

    /* norm 2 */
    printf("norm2\n");
    double a2 = Norm2(8, a), b2 = Norm2_c64(4, b);
    printf("a norm2:\t%f\nb norm2:\t%f\n", a2, b2);

    /* norm inf */
    printf("norm infinity\n");
    double ai = Norm_Infinity(8, a), bi = Norm_Infinity_c64(4, b);
    printf("a norm inf:\t%f\nb norm inf:\t%f\n", ai, bi);
    return 0;
}