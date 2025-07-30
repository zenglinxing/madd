#include<stdio.h>
#include<stdint.h>
#include"madd.h"

int print_gap = 200;

int main(int argc, char *argv[])
{
    uint32_t rng_type = 0;
    uint64_t seed = 10;
    if (argc > 1){
        rng_type = atoi(argv[1]);
    }
    printf("rng type:\t%d\n", rng_type);

    RNG_Param rng = RNG_Init(seed, rng_type);
    double res;
    int i, n=1000;
    for (i=0; i<n; i++){
        res = Rand(&rng);
        if ((i+1)%print_gap == 0){
            printf("%d\t%f\n", i, res);
        }
    }
    return 0;
}