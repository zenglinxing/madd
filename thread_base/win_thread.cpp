/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./thread_base/win_thread.cpp
*/
#include<stdio.h>
#include<stdlib.h>
#include<windows.h>
#include"thread_base.h"

/* check if size of mutex etc. is larger than expected */
class CppThreadBase_SizeCheck{
    public:
    CppThreadBase_SizeCheck(){
        size_t size_mutex = sizeof(CRITICAL_SECTION), size_condition_variable = sizeof(CONDITION_VARIABLE), size_rwlock=sizeof(SRWLOCK);
        bool flag_fail = false;
        if (size_mutex > MADD_THREAD_BASE_MUTEX_LEN){
            printf("Madd Error!\nSize of mutex is %llu, larger than expected %d. Try to re-compile Madd after resetting macro MADD_THREAD_BASE_MUTEX_LEN.\n", size_mutex, MADD_THREAD_BASE_MUTEX_LEN);
            flag_fail = true;
        }
        if (size_condition_variable > MADD_THREAD_BASE_CONDITION_VARIABLE_LEN){
            printf("Madd Error!\nSize of condition variable is %llu, larger than expected %d. Try to re-compile Madd after resetting macro MADD_THREAD_BASE_MUTEX_LEN.\n", size_condition_variable, MADD_THREAD_BASE_CONDITION_VARIABLE_LEN);
            flag_fail = true;
        }
        if (size_rwlock > MADD_THREAD_BASE_RWLOCK_LEN){
            printf("Madd Error!\nSize of read-write lock is %llu, larger than expected %d. Try to re-compile Madd after resetting macro MADD_THREAD_BASE_RWLOCK_LEN.\n", size_rwlock, MADD_THREAD_BASE_RWLOCK_LEN);
            flag_fail = true;
        }
        if (flag_fail){
            exit(EXIT_FAILURE);
        }
    }
};

static CppThreadBase_SizeCheck threadbase_sizecheck;