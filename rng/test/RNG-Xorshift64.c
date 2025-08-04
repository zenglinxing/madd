/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include"madd.h"

char *file_BE = "test_RNG-Xorshift64_BE";
char *file_LE = "test_RNG-Xorshift64_LE";

int main(int argc,char *argv[])
{
    RNG_Xorshift64_Param rxp1, rxp2, rxp3;
    rxp1 = RNG_Xorshift64_Init(10);

    FILE *fp2, *fp3;
    fp2 = fopen(file_BE, "wb");
    RNG_Xorshift64_Write_BE(&rxp1, fp2);
    fclose(fp2);
    fp2 = fopen(file_BE, "rb");
    rxp2 = RNG_Xorshift64_Read_BE(fp2);
    fclose(fp2);

    fp3 = fopen(file_LE, "wb");
    RNG_Xorshift64_Write_LE(&rxp1, fp3);
    fclose(fp3);
    fp3 = fopen(file_LE, "rb");
    rxp3 = RNG_Xorshift64_Read_LE(fp3);
    fclose(fp3);

    int i;

    double num1, num2, num3;
    for (i=0;i<1000;i++){
        num1 = Rand_Xorshift64(&rxp1);
        num2 = Rand_Xorshift64(&rxp2);
        num3 = Rand_Xorshift64(&rxp3);
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
