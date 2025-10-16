/* coding: utf-8 */
#include<stdio.h>
#include<wchar.h>
#include<stdbool.h>
#include<stdlib.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    Madd_Error_Enable_Logfile("test_basic-error-info.log");
    madd_error_keep_print = true;
    madd_save_wide  = true;
    madd_print_wide = true;
    if (argc > 1) madd_error_keep_print = atoi(argv[1]) != 0;
    if (argc > 2) madd_save_wide  = atoi(argv[2]) != 0;
    if (argc > 3) madd_print_wide = atoi(argv[3]) != 0;

    int i;
    for (i=0; i<8; i++){
        Madd_Error_Add(MADD_ERROR, L"func 1: error info");
    }
    for (i=0; i<10; i++){
        Madd_Error_Add(MADD_WARNING, L"func 2: warning info");
    }
    for (i=0; i<12; i++){
        Madd_Error_Add(MADD_ERROR, L"func 1: error info");
    }
    for (i=0; i<6; i++){
        Madd_Error_Add(MADD_WARNING, L"func 2: warning info");
    }

    FILE *fp=fopen("test_basic-error-info.txt", "wb");

    madd_error_color_print = true;
    Madd_Print(L"\n\nprint last:\n");
    Madd_Error_Print_Last();
    Madd_Error_Save_Last(fp);

    madd_error_color_print = true;
    Madd_Print(L"\n\nprint all:\n");
    Madd_Error_Print_All();
    Madd_Error_Save_All(fp);

    Madd_Error_Item mei;
    int madd_error_ret = Madd_Error_Get_Last(&mei);
    wchar_t print_info[80];
    swprintf(print_info, 80, L"\n\nlast sign:\t%d\n", mei.sign);
    Madd_Print(print_info);
    swprintf(print_info, 80, L"%ls\n", mei.info);
    Madd_Print(print_info);

    fclose(fp);
    return 0;
}