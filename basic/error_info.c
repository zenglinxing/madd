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

bool madd_error_keep_print = false, madd_error_print_wide = false, madd_error_save_wide = false;
bool madd_error_stop = false, madd_warning_stop = false;
Madd_Error madd_error={.n=0, .flag_n_exceed=false, .n_error=0, .n_warning=0};
static FILE *madd_error_fp = NULL;
/*
note: if Madd_Error_Enable_Logfile is used, turn true;
if Madd_Error_Set_Logfile, turn false;
*/
static bool madd_error_file_enable = false;
uint64_t madd_error_n = 0;

static void Madd_Error_Print_Wide2Narrow(const wchar_t *wstr)
{
    size_t required = wcstombs(NULL, wstr, 0);  // 计算所需字节数
    if (required == (size_t)-1) {
        if (madd_error_print_wide) wprintf(L"Madd_Error_Print_Wide2Narrow: unable to convert wide character.\n");
        else printf("Madd_Error_Print_Wide2Narrow: unable to convert wide character.\n");
        exit(EXIT_FAILURE);
    }
    char *str = malloc(required + 1);
    wcstombs(str, wstr, required + 1);
    printf("%s", str);
    free(str);
}

static void Madd_Error_Save_Wide2Narrow(FILE *fp, const wchar_t *wstr)
{
    size_t required = wcstombs(NULL, wstr, 0);  // 计算所需字节数
    if (required == (size_t)-1) {
        if (madd_error_print_wide) wprintf(L"Madd_Error_Save_Wide2Narrow: unable to convert wide character.\n");
        else printf("Madd_Error_Save_Wide2Narrow: unable to convert wide character.\n");
        exit(EXIT_FAILURE);
    }
    char *str = malloc(required + 1);
    wcstombs(str, wstr, required + 1);
    fprintf(fp, "%s", str);
    free(str);
}

static uint64_t Madd_Error_Warning_ID(uint64_t i_item)
{
    uint64_t n_warning=0, i;
    for (i=i_item; i<MADD_ERROR_MAX; i++){
        if (madd_error.item[i].sign == MADD_WARNING) n_warning ++;
    }
    return madd_error.n_warning - n_warning + 1;
}

static uint64_t Madd_Error_Error_ID(uint64_t i_item)
{
    uint64_t n_error=0, i;
    for (i=i_item; i<MADD_ERROR_MAX; i++){
        if (madd_error.item[i].sign == MADD_ERROR) n_error ++;
    }
    return madd_error.n_error - n_error + 1;
}

bool Madd_Error_Enable_Logfile(const char *log_file_name)
{
    if (madd_error_file_enable){
        fclose(madd_error_fp);
        madd_error_fp = NULL;
    }
    madd_error_fp = fopen(log_file_name, "wb");
    if (madd_error_fp==NULL){
        Madd_Error_Add(MADD_ERROR, L"Madd_Error_Enable_Logfile: unable to create madd error log file");
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
    }else{
        fclose(madd_error_fp);
        madd_error_fp = NULL;
    }
    madd_error_file_enable = false;
}

void Madd_Error_Add(char sign, const wchar_t *info)
{
#ifdef MADD_ENABLE_MULTITHREAD
    Mutex_Lock(&madd_error.mutex);
#endif

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
    size_t n_char = wcslen(info), n_max_copy = (MADD_ERROR_INFO_LEN <= n_char) ? MADD_ERROR_INFO_LEN-1 : n_char;
    wcsncpy(madd_error.item[madd_error.n-1].info, info, n_max_copy);
    madd_error.item[madd_error.n-1].info[n_max_copy] = L'\0';

    /* print error/warning */
    if (madd_error_keep_print){
        Madd_Error_Print_Last();
    }

    /* log file */
    wchar_t wtime_stamp[100], *wsign;
    struct tm local_tm;
    Madd_Error_Item mei;
    uint64_t id_error;
    if (madd_error_fp!=NULL){
        mei = madd_error.item[madd_error.n-1];
#ifdef _WIN32
        localtime_s(&local_tm, &mei.time_stamp);
#else
        localtime_r(&mei.time_stamp, &local_tm);
#endif
        wsign = (mei.sign==MADD_ERROR) ? L"Error  " : L"Warning";
        if (mei.sign == MADD_ERROR) id_error = Madd_Error_Error_ID(madd_error.n-1);
        else if (mei.sign == MADD_WARNING) id_error = Madd_Error_Warning_ID(madd_error.n-1);
        Time_Stamp_String(mei.time_stamp, wtime_stamp);
        wchar_t print_info[MADD_ERROR_INFO_LEN+100];
        swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, id_error, wtime_stamp, mei.info);
        if (madd_error_save_wide){
            fwprintf(madd_error_fp, print_info);
        }else{
            Madd_Error_Save_Wide2Narrow(madd_error_fp, print_info);
        }
        fflush(madd_error_fp);
    }

    /* check if the program should be stopped */
    if (sign == MADD_ERROR && madd_error_stop){
        if (madd_error_print_wide) wprintf(L"Madd Error triggered, program stopped.\n");
        else printf("Madd Error triggered, program stopped.\n");
        exit(EXIT_FAILURE);
    }
    if (sign == MADD_WARNING && madd_warning_stop){
        if (madd_error_print_wide) wprintf(L"Madd Warning triggered, program stopped.\n");
        else printf("Madd Warning triggered, program stopped.\n");
        exit(EXIT_FAILURE);
    }

#ifdef MADD_ENABLE_MULTITHREAD
    Mutex_Unlock(&madd_error.mutex);
#endif
}

void Madd_Error_Print_Last(void)
{
    if (madd_error.flag_n_exceed){
        if (madd_error_print_wide) wprintf(L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
        else printf("Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t id_error;
    if (madd_error.n == 0){
        if (madd_error_print_wide) wprintf(L"Madd Success.\n");
        else printf("Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        if (madd_error.item[madd_error.n-1].sign == MADD_ERROR){
            wsign = L"Error  ";
            id_error = Madd_Error_Error_ID(madd_error.n-1);
        }else if (madd_error.item[madd_error.n-1].sign == MADD_WARNING){
            wsign = L"Warning";
            id_error = Madd_Error_Warning_ID(madd_error.n-1);
        }else{
            wsign = L"Unknown";
        }
        Time_Stamp_String(madd_error.item[madd_error.n-1].time_stamp, wtime_stamp);
        wchar_t print_info[MADD_ERROR_INFO_LEN+100];
        swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, id_error, wtime_stamp, madd_error.item[madd_error.n-1].info);
        if (madd_error_print_wide){
            wprintf(print_info);
        }else{
            Madd_Error_Print_Wide2Narrow(print_info);
        }
        free(wtime_stamp);
    }
}

void Madd_Error_Save_Last(FILE *fp)
{
    if (madd_error.flag_n_exceed){
        if (madd_error_save_wide) fwprintf(fp, L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
        else fprintf(fp, "Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t id_error;
    if (madd_error.n == 0){
        if (madd_error_save_wide) fwprintf(fp, L"Madd Success.\n");
        else fprintf(fp, "Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        if (madd_error.item[madd_error.n-1].sign == MADD_ERROR){
            wsign = L"Error  ";
            id_error = Madd_Error_Error_ID(madd_error.n-1);
        }else if (madd_error.item[madd_error.n-1].sign == MADD_WARNING){
            wsign = L"Warning";
            id_error = Madd_Error_Warning_ID(madd_error.n-1);
        }else{
            wsign = L"Unknown";
        }
        Time_Stamp_String(madd_error.item[madd_error.n-1].time_stamp, wtime_stamp);
        wchar_t print_info[MADD_ERROR_INFO_LEN+100];
        swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, id_error, wtime_stamp, madd_error.item[madd_error.n-1].info);
        if (madd_error_save_wide){
            fwprintf(fp, print_info);
        }else{
            Madd_Error_Save_Wide2Narrow(fp, print_info);
        }
        free(wtime_stamp);
    }
}

void Madd_Error_Print_All(void)
{
    if (madd_error.flag_n_exceed){
        if (madd_error_print_wide) wprintf(L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
        else printf("Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t i_item, id_error;
    if (madd_error.n==0){
        wprintf(L"Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        for (i_item=0; i_item<madd_error.n; i_item++){
            if (madd_error.item[i_item].sign == MADD_ERROR){
                wsign = L"Error  ";
                id_error = Madd_Error_Error_ID(i_item);
            }else if (madd_error.item[i_item].sign == MADD_WARNING){
                wsign = L"Warning";
                id_error = Madd_Error_Warning_ID(i_item);
            }else{
                wsign = L"Unknown";
            }
            Time_Stamp_String(madd_error.item[i_item].time_stamp, wtime_stamp);
            wchar_t print_info[MADD_ERROR_INFO_LEN+100];
            swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n-madd_error.n+i_item+1, wsign, id_error, wtime_stamp, madd_error.item[i_item].info);
            if (madd_error_print_wide){
                wprintf(print_info);
            }else{
                Madd_Error_Print_Wide2Narrow(print_info);
            }
        }
        free(wtime_stamp);
    }
}

void Madd_Error_Save_All(FILE *fp)
{
    if (madd_error.flag_n_exceed){
        if (madd_error_save_wide) fwprintf(fp, L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
        else fprintf(fp, "Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t i_item, id_error;
    if (madd_error.n==0){
        if (madd_error_save_wide) fwprintf(fp, L"Madd Success.\n");
        else fprintf(fp, "Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        for (i_item=0; i_item<madd_error.n; i_item++){
            if (madd_error.item[i_item].sign == MADD_ERROR){
                wsign = L"Error  ";
                id_error = Madd_Error_Error_ID(i_item);
            }else if (madd_error.item[i_item].sign == MADD_WARNING){
                wsign = L"Warning";
                id_error = Madd_Error_Warning_ID(i_item);
            }else{
                wsign = L"Unknown";
            }
            Time_Stamp_String(madd_error.item[i_item].time_stamp, wtime_stamp);
            wchar_t print_info[MADD_ERROR_INFO_LEN+100];
            swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n-madd_error.n+i_item+1, wsign, id_error, wtime_stamp, madd_error.item[i_item].info);
            if (madd_error_save_wide){
                fwprintf(fp, print_info);
            }else{
                Madd_Error_Save_Wide2Narrow(fp, print_info);
            }
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