/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/error_info.c
*/
#define _FILE_OFFSET_BITS 64
#include<time.h>
#include<stdio.h>
#include<wchar.h>
#include<string.h>
#include<stdlib.h>
#include<stdbool.h>
#include"basic.h"

bool madd_error_keep_print = false;
Madd_Error madd_error={.n=0, .flag_n_exceed=false, .n_error=0, .n_warning=0};
static FILE *madd_error_fp = NULL;
/*
note: if Madd_Error_Enable_Logfile is used, turn true;
if Madd_Error_Set_Logfile, turn false;
*/
static bool madd_error_file_enable = false;
uint64_t madd_error_n = 0;

bool Madd_Error_Enable_Logfile(const char *log_file_name)
{
    if (madd_error_file_enable){
        fclose(madd_error_fp);
        madd_error_fp = NULL;
    }
    madd_error_fp = fopen(log_file_name, "wb");
    if (madd_error_fp==NULL){
        Madd_Error_Add(MADD_ERROR, L"Madd_Error_Enable_Logfile: Unable to create madd error log file");
        return false;
    }
    madd_error_file_enable = true;
    return true;
}

void Madd_Error_Set_Logfile(FILE *fp)
{
    if (madd_error_file_enable){
        fclose(madd_error_fp);
    }
    madd_error_file_enable = false;
    madd_error_fp = fp;
}

void Madd_Error_Close_Logfile(void)
{
    if (madd_error_fp == NULL){
        Madd_Error_Add(MADD_ERROR, L"Madd_Error_Close_Logfile: cannot close a NULL FILE.");
    }
    else{
        fclose(madd_error_fp);
        madd_error_fp = NULL;
    }
    madd_error_file_enable = false;
}

void Madd_Error_Add(char sign, const wchar_t *info)
{
    if (madd_error.n==0){
        madd_error.n = 1;
    }else if (madd_error.n==MADD_ERROR_MAX){
        madd_error.flag_n_exceed = true;
        memmove(&madd_error.item[0], &madd_error.item[1], (MADD_ERROR_MAX-1)*sizeof(Madd_Error_Item));
    }else{
        /*memmove(&madd_error.item[0], &madd_error.item[1], madd_error.n*sizeof(Madd_Error_Item));*/
        madd_error.n ++;
    }
    madd_error_n ++;
    madd_error.item[madd_error.n-1].sign = sign;
    if (sign == MADD_ERROR){
        madd_error.n_error ++;
    }
    else if (sign == MADD_WARNING){
        madd_error.n_warning ++;
    }
    madd_error.item[madd_error.n-1].time_stamp = time(NULL);
    /* copy info */
    size_t n_char = wcslen(info), n_max_copy = (MADD_ERROR_INFO_LEN <= n_char) ? MADD_ERROR_INFO_LEN : n_char+1;
    memcpy(madd_error.item[madd_error.n-1].info, info, (n_max_copy+1)*sizeof(wchar_t));

    /* print error/warning */
    if (madd_error_keep_print){
        Madd_Error_Print_Last();
    }

    /* log file */
    wchar_t wtime_stamp[100], *wsign;
    struct tm local_tm;
    Madd_Error_Item mei;
    uint64_t n_sign;
    size_t log_temp_len;
    if (madd_error_fp!=NULL){
        mei = madd_error.item[madd_error.n-1];
#ifdef _WIN32
        localtime_s(&local_tm, &mei.time_stamp);
#else
        localtime_r(&mei.time_stamp, &local_tm);
#endif
        wsign = (mei.sign==MADD_ERROR) ? L"Error  " : L"Warning";
        n_sign = (mei.sign==MADD_ERROR) ? madd_error.n_error : madd_error.n_warning;
        Time_Stamp_String(mei.time_stamp, wtime_stamp);
        fwprintf(madd_error_fp, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, n_sign, wtime_stamp, mei.info);
        fflush(madd_error_fp);
    }
}

void Madd_Error_Print_Last(void)
{
    if (madd_error.flag_n_exceed){
        wprintf(L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t n_sign;
    if (madd_error.n == 0){
        wprintf(L"Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        if (madd_error.item[madd_error.n-1].sign == MADD_ERROR){
            wsign = L"Error  ";
        }else if (madd_error.item[madd_error.n-1].sign == MADD_WARNING){
            wsign = L"Warning";
        }else{
            wsign = L"Unknown";
        }
        n_sign = (madd_error.item[madd_error.n-1].sign==MADD_ERROR) ? madd_error.n_error : madd_error.n_warning;
        Time_Stamp_String(madd_error.item[madd_error.n-1].time_stamp, wtime_stamp);
        wprintf(L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, n_sign, wtime_stamp, madd_error.item[madd_error.n-1].info);
    }
}

void Madd_Error_Save_Last(FILE *fp)
{
    if (madd_error.flag_n_exceed){
        fwprintf(fp, L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t n_sign;
    if (madd_error.n == 0){
        fwprintf(fp, L"Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        if (madd_error.item[madd_error.n-1].sign == MADD_ERROR){
            wsign = L"Error  ";
        }else if (madd_error.item[madd_error.n-1].sign == MADD_WARNING){
            wsign = L"Warning";
        }else{
            wsign = L"Unknown";
        }
        n_sign = (madd_error.item[madd_error.n-1].sign==MADD_ERROR) ? madd_error.n_error : madd_error.n_warning;
        Time_Stamp_String(madd_error.item[madd_error.n-1].time_stamp, wtime_stamp);
        fwprintf(fp, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, n_sign, wtime_stamp, madd_error.item[madd_error.n-1].info);
    }
}

void Madd_Error_Print_All(void)
{
    if (madd_error.flag_n_exceed){
        wprintf(L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t i_item, n_sign;
    if (madd_error.n==0){
        wprintf(L"Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        for (i_item=0; i_item<madd_error.n; i_item++){
            if (madd_error.item[i_item].sign == MADD_ERROR){
                wsign = L"Error  ";
            }else if (madd_error.item[i_item].sign == MADD_WARNING){
                wsign = L"Warning";
            }else{
                wsign = L"Unknown";
            }
            n_sign = (madd_error.item[i_item].sign==MADD_ERROR) ? madd_error.n_error : madd_error.n_warning;
            Time_Stamp_String(madd_error.item[i_item].time_stamp, wtime_stamp);
            wprintf(L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, n_sign, wtime_stamp, madd_error.item[i_item].info);
        }
        free(wtime_stamp);
    }
}

void Madd_Error_Save_All(FILE *fp)
{
    if (madd_error.flag_n_exceed){
        fwprintf(fp, L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t i_item, n_sign;
    if (madd_error.n==0){
        fwprintf(fp, L"Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        for (i_item=0; i_item<madd_error.n; i_item++){
            if (madd_error.item[i_item].sign == MADD_ERROR){
                wsign = L"Error  ";
            }else if (madd_error.item[i_item].sign == MADD_WARNING){
                wsign = L"Warning";
            }else{
                wsign = L"Unknown";
            }
            n_sign = (madd_error.item[i_item].sign==MADD_ERROR) ? madd_error.n_error : madd_error.n_warning;
            Time_Stamp_String(madd_error.item[i_item].time_stamp, wtime_stamp);
            fwprintf(fp, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, n_sign, wtime_stamp, madd_error.item[i_item].info);
        }
        free(wtime_stamp);
    }
}

char Madd_Error_Get_Last(Madd_Error_Item *mei)
{
    if (madd_error.n){
        if (mei){
            memcpy(mei, &madd_error.item[0], sizeof(Madd_Error_Item));
        }
        return madd_error.item[0].sign;
    }else{
        return MADD_SUCCESS;
    }
}