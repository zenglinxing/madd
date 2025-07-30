/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    uint64_t a = 0x0123456789abcdefL;
    printf("a:\t%016llx\n", a);
    union _union64 u64={.u=a}, ret64;
    union _union16 u16={.u=u64.u16[0]}, ret16;
    union _union32 u32={.u=u64.u32[0]}, ret32;
    
    ret16 = Byte_Reverse_16(u16);
    printf("ret16:\t%04x\n", ret16.u);

    ret32 = Byte_Reverse_32(u32);
    printf("ret32:\t%08x\n", ret32.u);

    ret64 = Byte_Reverse_64(u64);
    printf("ret64:\t%016llx\n", ret64.u);
    return 0;
}