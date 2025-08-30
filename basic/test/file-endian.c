/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    bool endian_type=Endian_Type(), r_endian;
    uint64_t a[2]={0x0001020304050607, 0x08090a0b0c0d0e0f}, aa[2];
    double b[2]={9.4, Inf}, bb[2];
    printf("endian:\t%d\n", endian_type);
    printf("a:\t%016llx\t%016llx\n", a[0], a[1]);
    printf("b:\t%f\t%f\n", b[0], b[1]);

    printf("BE:\n");
    FILE *fp=fopen("test_basic-file-endian_BE", "wb");
    Write_1byte(fp, &endian_type);
    Write_Array_BE(fp, a, sizeof(a)/sizeof(uint64_t), sizeof(uint64_t));
    Write_Array_BE(fp, b, sizeof(b)/sizeof(double), sizeof(double));
    fclose(fp);

    fp = fopen("test_basic-file-endian_BE", "rb");
    r_endian = Read_1byte(fp).u;
    Read_Array_BE(fp, aa, sizeof(a)/sizeof(uint64_t), sizeof(uint64_t));
    Read_Array_BE(fp, bb, sizeof(b)/sizeof(uint64_t), sizeof(uint64_t));
    fclose(fp);
    printf("endian:\t%d\n", r_endian);
    printf("aa:\t%016llx\t%016llx\n", aa[0], aa[1]);
    printf("bb:\t%f\t%f\n", bb[0], bb[1]);

    printf("LE:\n");
    fp=fopen("test_basic-file-endian_LE", "wb");
    Write_1byte(fp, &endian_type);
    Write_Array_LE(fp, a, sizeof(a)/sizeof(uint64_t), sizeof(uint64_t));
    Write_Array_LE(fp, b, sizeof(b)/sizeof(double), sizeof(double));
    fclose(fp);

    fp = fopen("test_basic-file-endian_LE", "rb");
    r_endian = Read_1byte(fp).u;
    Read_Array_LE(fp, aa, sizeof(a)/sizeof(uint64_t), sizeof(uint64_t));
    Read_Array_LE(fp, bb, sizeof(b)/sizeof(uint64_t), sizeof(uint64_t));
    fclose(fp);
    printf("endian:\t%d\n", r_endian);
    printf("aa:\t%016llx\t%016llx\n", aa[0], aa[1]);
    printf("bb:\t%f\t%f\n", bb[0], bb[1]);
    return 0;
}