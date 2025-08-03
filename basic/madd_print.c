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
    size_t len_char = wcstombs(NULL, wstr, 0);
    if (len_char == (size_t)-1){
        if (madd_print_wide) wprintf(L"Madd Exit!\tOccur failure to calculate the wide string.");
        else printf("Madd Exit!\tOccur failure to calculate the wide string.");
        exit(EXIT_FAILURE);
    }

    char *str = (char*)malloc(len_char+1);
    if (str == NULL){
        if (madd_print_wide) wprintf(L"Madd Exit!\tFailed to allocate narrow string mem.");
        else printf("Madd Exit!\tFailed to allocate narrow string mem.");
        exit(EXIT_FAILURE);
    }

    size_t len_converted = wcstombs(str, wstr, len_char+1);
    if (len_converted == (size_t)-1){
        free(str);
        if (madd_print_wide) wprintf(L"Madd Exit!\tFailed to convert wide string to narrow string.");
        else printf("Madd Exit!\tFailed to convert wide string to narrow string.");
        exit(EXIT_FAILURE);
    }

    return str;
}

void Madd_Print(wchar_t *wstr)
{
    if (wstr == NULL){
        if (madd_print_wide) wprintf(L"Madd Exit!\tThe given wide string is NULL.");
        else printf("Madd Exit!\tThe given wide string is NULL.");
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
        if (madd_print_wide) wprintf(L"Madd Exit!\tThe given wide string is NULL.");
        else printf("Madd Exit!\tThe given wide string is NULL.");
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