#include<stdio.h>
#include"thread_base.h"

int main(int argc, char *argv[])
{
    uint64_t n_core = N_CPU_Thread();
    printf("n core:\t%llu\n", n_core);
    return 0;
}