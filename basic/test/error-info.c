/* coding: utf-8 */
#include<stdio.h>
#include<wchar.h>
#include<stdbool.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    Madd_Error_Enable_Logfile("test_error-info.log");
    madd_error_keep_print = true;

    Madd_Error_Add(MADD_ERROR, L"func 1: error info");
    Madd_Error_Add(MADD_WARNING, L"func 2: warning info");

    FILE *fp=fopen("test_error-info.txt", "wb");

    printf("print last:\n");
    Madd_Error_Print_Last();
    Madd_Error_Save_Last(fp);

    printf("print all:\n");
    Madd_Error_Print_All();
    Madd_Error_Save_All(fp);

    Madd_Error_Item mei;
    int madd_error_ret = Madd_Error_Get_Last(&mei);
    printf("last sign:\t%d\n", mei.sign);
    wprintf(L"%ls\n", mei.info);

    fclose(fp);
    return 0;
}