/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/madd_print.c
These functions won't call Madd_Error_Add, because they are the basis of Madd_Error_Add.
*/
#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<string.h>
#include<wchar.h>
#include<stdbool.h>
#include"basic.h"

bool madd_print_wide = false, madd_save_wide = false;

static char *Wide2Char(wchar_t *wstr)
{
    if (wstr == NULL){
        if (madd_print_wide) wprintf(L"%hs\t%hs line %d:\n\tMadd Exit!\tThe wide character string pointer is NULL.", __func__, __FILE__, __LINE__);
        else printf("%s\t%s line %d:\n\tMadd Exit!\tThe wide character string pointer is NULL.", __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    size_t len_char = wcstombs(NULL, wstr, 0);
    //wprintf(wstr);
    if (len_char == (size_t)-1 || len_char == 0){
        if (madd_print_wide) wprintf(L"%hs\t%hs line %d:\n\tMadd Exit!\tOccur failure to calculate the wide string.", __func__, __FILE__, __LINE__);
        else printf("%s\t%s line %d:\n\tMadd Exit!\tOccur failure to calculate the wide string.", __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    char *str = (char*)malloc(len_char+1);
    if (str == NULL){
        if (madd_print_wide) wprintf(L"%hs\t%hs line %d:\n\tMadd Exit!\tFailed to allocate narrow string mem.", __func__, __FILE__, __LINE__);
        else printf("%s\t%s line %d:\n\tMadd Exit!\tFailed to allocate narrow string mem.", __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    size_t len_converted = wcstombs(str, wstr, len_char+1);
    if (len_converted == (size_t)-1){
        free(str);
        if (madd_print_wide) wprintf(L"%hs\t%hs line %d:\n\tMadd Exit!\tFailed to convert wide string to narrow string.", __func__, __FILE__, __LINE__);
        else printf("%s\t%s line %d:\n\tMadd Exit!\tFailed to convert wide string to narrow string.", __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    if (str[len_char] != '\0'){
        free(str);
        if (madd_print_wide) wprintf(L"%hs\t%hs line %d:\n\tMadd Exit!\tThe end of string is not '\\0'.", __func__, __FILE__, __LINE__);
        else printf("%s\t%s line %d:\n\tMadd Exit!\tMadd Exit!\tThe end of string is not '\\0'.", __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    return str;
}

void Madd_Print(wchar_t *wstr)
{
    if (wstr == NULL){
        if (madd_print_wide) wprintf(L"%hs\t%hs line %d:\n\tMadd Exit!\tThe given wide string is NULL.", __func__, __FILE__, __LINE__);
        else printf("%s\t%s line %d:\n\tMadd Exit!\tThe given wide string is NULL.", __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (madd_print_wide){
        wprintf(L"%ls", wstr);
    }else{
        char *str = Wide2Char(wstr);
        printf("%s", str);
        free(str);
    }
}

void Madd_Save(FILE *fp, wchar_t *wstr)
{
    if (wstr == NULL){
        if (madd_print_wide) wprintf(L"%hs\t%hs line %d:\n\tMadd Exit!\tThe given wide string is NULL.", __func__, __FILE__, __LINE__);
        else printf("%s\t%s line %d:\n\tMadd Exit!\tThe given wide string is NULL.", __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (madd_save_wide){
        fwprintf(fp, L"%ls", wstr);
    }else{
        char *str = Wide2Char(wstr);
        fprintf(fp, "%s", str);
        free(str);
    }
}