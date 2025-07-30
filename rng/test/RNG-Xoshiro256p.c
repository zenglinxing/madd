/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include"madd.h"

char *file_BE = "test_RNG-Xoshiro256p_BE";
char *file_LE = "test_RNG-Xoshiro256p_LE";

int main(int argc,char *argv[])
{
    RNG_Xoshiro256_Param rxp1, rxp2, rxp3;
    rxp1 = RNG_Xoshiro256p_Init(10);

    FILE *fp2, *fp3;
    fp2 = fopen(file_BE, "wb");
    RNG_Xoshiro256p_Write_BE(rxp1, fp2);
    fclose(fp2);
    fp2 = fopen(file_BE, "rb");
    rxp2 = RNG_Xoshiro256p_Read_BE(fp2);
    fclose(fp2);

    fp3 = fopen(file_LE, "wb");
    RNG_Xoshiro256p_Write_LE(rxp1, fp3);
    fclose(fp3);
    fp3 = fopen(file_LE, "rb");
    rxp3 = RNG_Xoshiro256p_Read_LE(fp3);
    fclose(fp3);

    int i;

    double num1, num2, num3;
    for (i=0;i<1000;i++){
        num1 = Rand_Xoshiro256p(&rxp1);
        num2 = Rand_Xoshiro256p(&rxp2);
        num3 = Rand_Xoshiro256p(&rxp3);
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
