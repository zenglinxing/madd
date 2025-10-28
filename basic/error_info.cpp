/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/error_info.cpp
To initialize madd_error.
*/
extern "C"{
#include"basic.h"
#include<thread_base/thread_base.h>
}

Madd_Error madd_error;

class CppMaddErrorRWLockInit{
    public:
    CppMaddErrorRWLockInit(){
        madd_error.flag_n_exceed = false;
        madd_error.n_error = madd_error.n_warning = madd_error.n = 0;
#ifdef MADD_ENABLE_MULTITHREAD
        RWLock_Init(&madd_error.rwlock);
#endif
    }
};

static CppMaddErrorRWLockInit madd_error_rwlock_init;