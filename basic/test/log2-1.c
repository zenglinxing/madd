#include<stdio.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    uint64_t a=1<<0, b=1<<4, c=1<<31, d=(1<<4)+1, e=(1<<31)+2;
    printf("a=%llu\t%llu\t%llu\n", a, Log2_Floor(a), Log2_Ceil(a));
    printf("b=%llu\t%llu\t%llu\n", b, Log2_Floor(b), Log2_Ceil(b));
    printf("c=%llu\t%llu\t%llu\n", c, Log2_Floor(c), Log2_Ceil(c));
    printf("d=%llu\t%llu\t%llu\n", d, Log2_Floor(d), Log2_Ceil(d));
    printf("e=%llu\t%llu\t%llu\n", e, Log2_Floor(e), Log2_Ceil(e));
    return 0;
}
