/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    uint8_t a8=0x12, b8;
    union _union8 u8={.u=a8}, ret8;
    ret8=Bit_Reverse_8(u8);
    b8 = ret8.u;
    printf("a=%x\n", a8);
    printf("b=%x\n", b8);
    uint16_t a16=0x12ff, b16;
    union _union16 u16={.u=a16}, ret16;
    ret16=Bit_Reverse_16(u16);
    b16 = ret16.u;
    printf("a=%x\n", a16);
    printf("b=%x\n", b16);
    uint32_t a32=0x123456ff, b32;
    union _union32 u32={.u=a32}, ret32;
    ret32=Bit_Reverse_32(u32);
    b32 = ret32.u;
    printf("a32=%x\n", a32);
    printf("b32=%x\n", b32);
    uint64_t a64=0x123456ff123456ffL, b64;
    union _union64 u64={.u=a64}, ret64;
    ret64=Bit_Reverse_64(u64);
    b64 = ret64.u;
    printf("a64=%llx\n", a64);
    printf("b64=%llx\n", b64);
    return 0;
}
