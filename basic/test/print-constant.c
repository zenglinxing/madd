/* coding: utf-8 */
#include<stdio.h>
#include<stdint.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    /* integer */
    printf("integer:\n");
    printf("Bin4:\t%x\n", Bin4);
    printf("Bin7:\t%x\n", Bin7);
    printf("Bin8:\t%x\n", Bin8);
    printf("Bin15:\t%x\n", Bin15);
    printf("Bin16:\t%x\n", Bin16);
    printf("Bin31:\t%x\n", Bin31);
    printf("Bin32:\t%x\n", Bin32);
    printf("Bin63:\t%llx\n", Bin63);
    printf("Bin64:\t%llx\n", Bin64);
    /* float */
    printf("float:\n");
    printf("pi:\t%f\n", Pi);
    printf("e:\t%f\n", E_Nat);
    printf("inf:\t%f\n", Inf);
    printf("NaN:\t%f\n", NaN);
    return 0;
}