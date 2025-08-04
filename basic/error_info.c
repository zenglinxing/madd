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
bool madd_error_exit = false, madd_warning_exit = false;
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

static void Madd_Error_Print_Last_Internal(bool flag_print_exceed_note)
{
    if (madd_error.flag_n_exceed && flag_print_exceed_note){
        wchar_t print_info[MADD_ERROR_INFO_LEN];
        swprintf(print_info, MADD_ERROR_INFO_LEN, L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
        Madd_Print(print_info);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t id_error;
    if (madd_error.n == 0){
        Madd_Print(L"Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        if (madd_error.item[madd_error.n-1].sign == MADD_ERROR){
            wsign = L"Error  ";
        }else if (madd_error.item[madd_error.n-1].sign == MADD_WARNING){
            wsign = L"Warning";
        }else{
            wsign = L"Unknown";
        }
        id_error = madd_error.item[madd_error.n-1].i_sign;
        Time_Stamp_String(madd_error.item[madd_error.n-1].time_stamp, wtime_stamp);
        wchar_t print_info[MADD_ERROR_INFO_LEN+100];
        swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, id_error, wtime_stamp, madd_error.item[madd_error.n-1].info);
        Madd_Print(print_info);
        free(wtime_stamp);
    }
}

void Madd_Error_Add(char sign, const wchar_t *info)
{
    time_t time_stamp = time(NULL);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Lock(&madd_error.rwlock);
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
    madd_error.item[madd_error.n-1].i_all = madd_error_n;
    madd_error.item[madd_error.n-1].sign = sign;
    if (sign == MADD_ERROR){
        madd_error.n_error ++;
        madd_error.item[madd_error.n-1].i_sign = madd_error.n_error;
    }
    else if (sign == MADD_WARNING){
        madd_error.n_warning ++;
        madd_error.item[madd_error.n-1].i_sign = madd_error.n_warning;
    }
    madd_error.item[madd_error.n-1].time_stamp = time_stamp;
    /* copy info */
    size_t n_char = wcslen(info), n_max_copy = (MADD_ERROR_INFO_LEN <= n_char) ? MADD_ERROR_INFO_LEN-1 : n_char;
    wcsncpy(madd_error.item[madd_error.n-1].info, info, n_max_copy);
    madd_error.item[madd_error.n-1].info[n_max_copy] = 0;

    /* print error/warning */
    if (madd_error_keep_print){
        Madd_Error_Print_Last_Internal(false);
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
        id_error = mei.i_sign;
        Time_Stamp_String(mei.time_stamp, wtime_stamp);
        wchar_t print_info[MADD_ERROR_INFO_LEN+100];
        swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, id_error, wtime_stamp, mei.info);
        Madd_Save(madd_error_fp, print_info);
        fflush(madd_error_fp);
    }

    /* check if the program should be stopped */
    if (sign == MADD_ERROR && madd_error_exit){
        Madd_Print(L"Madd Error triggered, program stopped.\n");
        exit(EXIT_FAILURE);
    }
    if (sign == MADD_WARNING && madd_warning_exit){
        Madd_Print(L"Madd Warning triggered, program stopped.\n");
        exit(EXIT_FAILURE);
    }

#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&madd_error.rwlock);
#endif
}

void Madd_Error_Print_Last(void)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Lock(&madd_error.rwlock);
#endif

    Madd_Error_Print_Last_Internal(true);

#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&madd_error.rwlock);
#endif
}

void Madd_Error_Save_Last(FILE *fp)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Lock(&madd_error.rwlock);
#endif

    if (madd_error.flag_n_exceed){
        wchar_t save_info[MADD_ERROR_INFO_LEN];
        swprintf(save_info, MADD_ERROR_INFO_LEN-1, L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
        Madd_Save(fp, save_info);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t id_error;
    if (madd_error.n == 0){
        Madd_Save(fp, L"Madd Success.\n");
    }else{
        wtime_stamp = (wchar_t*)malloc(MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t));
        if (madd_error.item[madd_error.n-1].sign == MADD_ERROR){
            wsign = L"Error  ";
        }else if (madd_error.item[madd_error.n-1].sign == MADD_WARNING){
            wsign = L"Warning";
        }else{
            wsign = L"Unknown";
        }
        id_error = madd_error.item[madd_error.n-1].i_sign;
        Time_Stamp_String(madd_error.item[madd_error.n-1].time_stamp, wtime_stamp);
        wchar_t print_info[MADD_ERROR_INFO_LEN+100];
        swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error_n, wsign, id_error, wtime_stamp, madd_error.item[madd_error.n-1].info);
        Madd_Save(fp, print_info);
        free(wtime_stamp);
    }
    fflush(fp);

#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&madd_error.rwlock);
#endif
    Madd_Print(L"save last end\n");
}

void Madd_Error_Print_All(void)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Lock(&madd_error.rwlock);
#endif

    if (madd_error.flag_n_exceed){
        wchar_t print_info[MADD_ERROR_INFO_LEN];
        swprintf(print_info, MADD_ERROR_INFO_LEN-1, L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
        Madd_Print(print_info);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t i_item, id_error;
    if (madd_error.n==0){
        Madd_Print(L"Madd Success.\n");
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
            id_error = madd_error.item[i_item].i_sign;
            Time_Stamp_String(madd_error.item[i_item].time_stamp, wtime_stamp);
            wchar_t print_info[MADD_ERROR_INFO_LEN+100];
            swprintf(print_info, MADD_ERROR_INFO_LEN+100-1, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error.item[i_item].i_all, wsign, id_error, wtime_stamp, madd_error.item[i_item].info);
            Madd_Print(print_info);
        }
        free(wtime_stamp);
    }

#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&madd_error.rwlock);
#endif
}

void Madd_Error_Save_All(FILE *fp)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Lock(&madd_error.rwlock);
#endif

    if (madd_error.flag_n_exceed){
        wchar_t save_info[MADD_ERROR_INFO_LEN];
        swprintf(save_info, MADD_ERROR_INFO_LEN, L"Madd Note: madd error info are more than %d now\n", MADD_ERROR_MAX);
        Madd_Save(fp, save_info);
    }
    wchar_t *wsign, *wtime_stamp;
    uint64_t i_item, id_error;
    if (madd_error.n==0){
        Madd_Save(fp, L"Madd Success.\n");
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
            id_error = madd_error.item[i_item].i_sign;
            Time_Stamp_String(madd_error.item[i_item].time_stamp, wtime_stamp);
            wchar_t print_info[MADD_ERROR_INFO_LEN+100];
            swprintf(print_info, MADD_ERROR_INFO_LEN+100, L"Madd %llu - %ls %llu:\t%ls\n\t%ls\n", madd_error.item[i_item].i_all, wsign, id_error, wtime_stamp, madd_error.item[i_item].info);
            Madd_Save(fp, print_info);
        }
        free(wtime_stamp);
    }
    fflush(fp);

#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&madd_error.rwlock);
#endif
}

char Madd_Error_Get_Last(Madd_Error_Item *mei)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Lock(&madd_error.rwlock);
#endif

    if (madd_error.n){
        if (mei){
            memcpy(mei, &madd_error.item[madd_error.n-1], sizeof(Madd_Error_Item));
        }
        return madd_error.item[0].sign;
    }else{
        return MADD_SUCCESS;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&madd_error.rwlock);
#endif
}