/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include"madd.h"

int print_gap = 200;

int main(int argc, char *argv[])
{
    madd_error_keep_print = true;

    uint32_t rng_type = 0;
    uint64_t seed = 10;
    if (argc > 1){
        rng_type = atoi(argv[1]);
    }
    printf("rng type:\t%d\n", rng_type);

    RNG_Param rng = RNG_Init(seed, rng_type);
    if (madd_error.n_error){
        exit(EXIT_FAILURE);
    }
    double res;
    int i, n=1000;
    for (i=0; i<n; i++){
        res = Rand(&rng);
        if ((i+1)%print_gap == 0){
            printf("%d\t%f\n", i, res);
        }
    }

    /* save & load */
    RNG_Param rng_old, rng_new;
    bool flag_BE_same, flag_LE_same;
    FILE *fp;
    double rand1, rand2;

    /* read from big endian */
    flag_BE_same = true;
    rng_old = rng;
    //printf("try to open BE file\n");
    fp = fopen("test_RNG-Param_BE", "wb");
    RNG_Write_BE(&rng_old, fp);
    //printf("BE file write\n");
    fclose(fp);
    fp = fopen("test_RNG-Param_BE", "rb");
    rng_new = RNG_Read_BE(fp);
    fclose(fp);
    //printf("BE file read\n");
    for (i=0; i<n; i++){
        rand1 = Rand(&rng_old);
        rand2 = Rand(&rng_new);
        if (rand1 != rand2){
            flag_BE_same = false;
        }
    }
    if (flag_BE_same){
        printf("Read from test_RNG-Param_BE are same\n");
    }else{
        printf("Read from test_RNG-Param_BE are different\n");
    }

    /* read from little endian */
    flag_LE_same = true;
    rng_old = rng;
    fp = fopen("test_RNG-Param_LE", "wb");
    RNG_Write_LE(&rng_old, fp);
    fclose(fp);
    fp = fopen("test_RNG-Param_LE", "rb");
    rng_new = RNG_Read_LE(fp);
    fclose(fp);
    for (i=0; i<n; i++){
        rand1 = Rand(&rng_old);
        rand2 = Rand(&rng_new);
        if (rand1 != rand2){
            flag_LE_same = false;
        }
    }
    if (flag_LE_same){
        printf("Read from test_RNG-Param_LE are same\n");
    }else{
        printf("Read from test_RNG-Param_LE are different\n");
    }

    if (!flag_BE_same || !flag_LE_same){
        exit(EXIT_FAILURE);
    }
    return 0;
}