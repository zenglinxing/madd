/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    uint64_t num_origin = 0xf70f07037ff07030;
    printf("tested number:\t%llx\n", num_origin);
    union _union8 u8={.u=num_origin & Bin8};
    union _union16 u16={.u=num_origin & Bin16};
    union _union32 u32={.u=num_origin & Bin32};
    union _union64 u64={.u=num_origin & Bin64};
    uint8_t n8, n16, n32, n64;
    n8 = Binary_Number_of_1_8bit(u8);
    n16 = Binary_Number_of_1_16bit(u16);
    n32 = Binary_Number_of_1_32bit(u32);
    n64 = Binary_Number_of_1_64bit(u64);
    printf("n8:\t%u\n", n8);
    printf("n16:\t%u\n", n16);
    printf("n32:\t%u\n", n32);
    printf("n64:\t%u\n", n64);
    if (n8!=2 || n16!=5 || n32!=16 || n64!=32){
        exit(1);
    }
    return 0;
}