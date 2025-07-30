#include<stdio.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    uint64_t a=1<<0, a1, a2, b=1<<4, b1, b2, c=1<<31, c1, c2, d=(1<<4)+1, d1, d2, e=(1<<31)+2, e1, e2;
    Log2_Full(a, &a1, &a2);
    Log2_Full(b, &b1, &b2);
    Log2_Full(c, &c1, &c2);
    Log2_Full(d, &d1, &d2);
    Log2_Full(e, &e1, &e2);
    printf("a=%llu\t%llu\t%llu\n", a, a1, a2);
    printf("b=%llu\t%llu\t%llu\n", b, b1, b2);
    printf("c=%llu\t%llu\t%llu\n", c, c1, c2);
    printf("d=%llu\t%llu\t%llu\n", d, d1, d2);
    printf("e=%llu\t%llu\t%llu\n", e, e1, e2);
    return 0;
}
