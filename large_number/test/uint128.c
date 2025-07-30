#include<stdio.h>
#include"madd.h"

uint64_t n=4;

void print_u128_binary(Uint128_LE a)
{
    uint64_t i, j;
    uint32_t b;
    for (i=0; i<n; i++){
        b = a.u32[n-1-i];
        for (j=0; j<32; j++){
            printf("%d", (b>>j)&0b1);
        }
    }
}

void print_u128_hex(Uint128_LE a)
{
    int i;
    for (i=0; i<n; i++){
        printf("%08u", a.u32[n-1-i]);
    }
}

int main(int argc, char *argv[])
{
    uint64_t i;
    Uint128_LE a={.u32={0xff0f35ac, 0x21435768, 0x86753421, 0xca53f0ff}}, b=a;
    Uint128_LE c=a, d=a;
    for (i=0; i<n; i++){
        c.u32[i] --;
        d.u32[i] ++;
    }

    Uint128_LE and, or, xor, not;
    printf("The following result should be identical as a\n");
    and = Uint128_And(a, b);
    print_u128_binary(and); printf("\n");
    not = Uint128_Not(a);
    print_u128_binary(not); printf("\n");
    xor = Uint128_Xor(a, b);
    print_u128_binary(xor); printf("\n");
    or = Uint128_Or(a, not);
    print_u128_binary(or); printf("\n\n");

    printf("The following 4 results should be equal\n");
    printf("ge=%d\tgeq=%d\tle=%d\tleq=%d\n", Uint128_Ge(a, b), Uint128_Geq(a, b), Uint128_Le(a, b), Uint128_Leq(a, b));

    printf("The following 4 results should be greater\n");
    printf("ge=%d\tgeq=%d\tle=%d\tleq=%d\n", Uint128_Ge(a, c), Uint128_Geq(a, c), Uint128_Le(a, c), Uint128_Leq(a, c));

    printf("The following 4 results should be less\n");
    printf("ge=%d\tgeq=%d\tle=%d\tleq=%d\n", Uint128_Ge(a, d), Uint128_Geq(a, d), Uint128_Le(a, d), Uint128_Leq(a, d));
    printf("\n");

    printf("The following %llu results should be greater\n", n);
    for (i=0; i<n; i++){
        b.u32[i] --;
        printf("le=%d\t", Uint128_Ge(a, b));
        b.u32[i] ++;
    }
    printf("\n");

    printf("The following %llu results should be greater\n", n);
    for (i=0; i<n; i++){
        b.u32[i] --;
        printf("leq=%d\t", Uint128_Geq(a, b));
        b.u32[i] ++;
    }
    printf("\n");

    printf("The following %llu results should be less\n", n);
    for (i=0; i<n; i++){
        b.u32[i] ++;
        printf("Ge=%d\t", Uint128_Le(a, b));
        b.u32[i] --;
    }
    printf("\n");

    printf("The following %llu results should be less\n", n);
    for (i=0; i<n; i++){
        b.u32[i] ++;
        printf("Geq=%d\t", Uint128_Leq(a, b));
        b.u32[i] --;
    }
    printf("\n");

    Uint128_LE u128_log2_test0=Bin128_Zero,
               u128_log2_test1={.u32={1, 0, 0, 0}},
               u128_log2_test2={.u32={0, 1, 0, 0}},
               u128_log2_test3={.u32={0, 0, 1, 0}},
               u128_log2_test4={.u32={0, 0, 0, 1}},
               u128_log2_test
               ;
    uint64_t log2_low, log2_high;

    printf("\n");
    printf("The following results should be 0 & 0\n");
    Uint128_Log2(u128_log2_test0, &log2_low, &log2_high);
    printf("%llu\t%llu\n", log2_low, log2_high);

    printf("The following results should be 0 & 0\n");
    Uint128_Log2(u128_log2_test1, &log2_low, &log2_high);
    printf("%llu\t%llu\n", log2_low, log2_high);

    printf("The following results should be 32 & 32\n");
    Uint128_Log2(u128_log2_test2, &log2_low, &log2_high);
    printf("%llu\t%llu\n", log2_low, log2_high);

    printf("The following results should be 64 & 64\n");
    Uint128_Log2(u128_log2_test3, &log2_low, &log2_high);
    printf("%llu\t%llu\n", log2_low, log2_high);

    printf("The following results should be 96 & 96\n");
    Uint128_Log2(u128_log2_test4, &log2_low, &log2_high);
    printf("%llu\t%llu\n", log2_low, log2_high);

    printf("\n");
    u128_log2_test = u128_log2_test4;
    u128_log2_test.u32[3] |= 0b10;
    printf("The following results should be 97 & 98\n");
    Uint128_Log2(u128_log2_test, &log2_low, &log2_high);
    printf("%llu\t%llu\n", log2_low, log2_high);

    u128_log2_test = u128_log2_test4;
    u128_log2_test.u32[1] |= 0b10;
    printf("The following results should be 96 & 97\n");
    Uint128_Log2(u128_log2_test, &log2_low, &log2_high);
    printf("%llu\t%llu\n", log2_low, log2_high);

    u128_log2_test = u128_log2_test3;
    u128_log2_test.u32[1] |= 0b10;
    printf("The following results should be 64 & 65\n");
    Uint128_Log2(u128_log2_test, &log2_low, &log2_high);
    printf("%llu\t%llu\n", log2_low, log2_high);

    return 0;
}