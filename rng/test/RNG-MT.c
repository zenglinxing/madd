/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include"madd.h"

char *file_BE = "test_RNG-MT_BE";
char *file_LE = "test_RNG-MT_LE";

int main(int argc,char *argv[])
{
    RNG_MT_Param mt1, mt2, mt3;
    mt1 = RNG_MT_Init(10);

    FILE *fp2, *fp3;
    fp2 = fopen(file_BE, "wb");
    RNG_MT_Write_BE(mt1, fp2);
    fclose(fp2);
    fp2 = fopen(file_BE, "rb");
    mt2 = RNG_MT_Read_BE(fp2);
    fclose(fp2);

    fp3 = fopen(file_LE, "wb");
    RNG_MT_Write_LE(mt1, fp3);
    fclose(fp3);
    fp3 = fopen(file_LE, "rb");
    mt3 = RNG_MT_Read_LE(fp3);
    fclose(fp3);

    int i;
    /*for (i=0; i<312; i++){
        printf("%d\t%llx\t%llx\t%llx\n", i, mt1.seeds[i], mt2.seeds[i], mt3.seeds[i]);
    }*/
    printf("mt param:\n");
    printf("i\t%x\t%x\t%x\n", mt1.i, mt2.i, mt3.i);
    printf("n_gen\t%x\t%x\t%x\n", mt1.n_gen, mt2.n_gen, mt3.n_gen);

    double num1, num2, num3;
    for (i=0;i<1000;i++){
        num1 = Rand_MT(&mt1);
        num2 = Rand_MT(&mt2);
        num3 = Rand_MT(&mt3);
        if (i%312==0){
            printf("%d\t%f\n",i,num1);
        }
        if (num1 != num2 || num1 != num3){
            printf("%d\t%f\t%f\t%f: not identical\n", i, num1, num2, num3);
            exit(EXIT_FAILURE);
        }
    }
    return 0;
}
