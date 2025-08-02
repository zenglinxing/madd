/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/error_info.cpp
To initialize the mutex in madd_error.
*/
#ifdef MADD_ENABLE_MULTITHREAD

extern "C"{
#include"basic.h"
#include"../thread_base/thread_base.h"
}

Madd_Error madd_error;

class CppMaddErrorMutexInit{
    public:
    CppMaddErrorMutexInit(){
        Mutex_Init(&madd_error.mutex);
    }
};

static CppMaddErrorMutexInit madd_error_mutex_init;

#endif