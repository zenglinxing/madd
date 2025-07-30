/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/time_stamp.c
*/
#include<stdio.h>
#include<string.h>
#include<stdint.h>
#include<time.h>
#include<wchar.h>
#include"basic.h"

static wchar_t *Week[7] = {L"Sun", L"Mon", L"Tue", L"Wed", L"Thu", L"Fri", L"Sat"};

void Time_Stamp_String(time_t t, wchar_t *str)
{
    struct tm local_tm;
#ifdef _WIN32
    localtime_s(&local_tm, &t);
#else
    localtime_r(&t, &local_tm);
#endif
    int ret = swprintf(str, MADD_TIME_STAMP_STRING_LEN*sizeof(wchar_t),
                       L"%d/%02d/%02d %s %02d:%02d:%02d",
                       local_tm.tm_year + 1900,
                       local_tm.tm_mon,
                       local_tm.tm_hour,
                       Week[local_tm.tm_wday],
                       local_tm.tm_hour,
                       local_tm.tm_min,
                       local_tm.tm_sec
                       );
    if (ret < 0){
        Madd_Error_Add(MADD_ERROR, L"Time_Stamp_String: cannot write time into string by func sprintf.");
    }
}