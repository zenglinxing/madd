/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./thread_base/win_thread.c
*/
#include<stdlib.h>
#include<windows.h>
#include"thread_base.h"

typedef DWORD (*thread_input_func)(void *);

Thread Thread_Create(void func(void*), void *param)
{
    HANDLE *handle=(HANDLE*)malloc(sizeof(HANDLE));
    thread_input_func tif = func;
    *handle = CreateThread(NULL, 0, tif, param, 0, 0);
    return handle;
}

void Thread_Join(Thread th)
{
    HANDLE *handle=(HANDLE*)th;
    DWORD ret = WaitForSingleObject(*handle, INFINITE);
    BOOL ret_close = CloseHandle(*handle);
    free(handle);
}

void Thread_Detach(Thread th)
{
    HANDLE *handle=(HANDLE*)th;
    BOOL ret_close = CloseHandle(*handle);
    free(handle);
}

/* mutex */
void Mutex_Init(Mutex *m)
{
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    InitializeCriticalSection(cs);
}

void Mutex_Lock(Mutex *m)
{
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    EnterCriticalSection((CRITICAL_SECTION*)m);
}

bool Mutex_Trylock(Mutex *m)
{
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    return TryEnterCriticalSection(cs);
}

void Mutex_Unlock(Mutex *m)
{
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    LeaveCriticalSection(cs);
}

void Mutex_Destroy(Mutex *m)
{
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    DeleteCriticalSection(cs);
}

/* condition variable */
void Condition_Variable_Init(Condition_Variable *cv)
{
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    InitializeConditionVariable(scv);
}

void Condition_Variable_Wait(Condition_Variable *cv, Mutex *m)
{
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    SleepConditionVariableCS(scv, cs, INFINITE);
}

bool Condition_Variable_Timed_Wait(Condition_Variable *cv, Mutex *m, double wait_sec)
{
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    DWORD ms = wait_sec * 1000;
    BOOL res = SleepConditionVariableCS(scv, cs, ms);
    return res;
}

void Condition_Variable_Wake(Condition_Variable *cv)
{
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    WakeConditionVariable(scv);
}

void Condition_Variable_Wake_All(Condition_Variable *cv)
{
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    WakeAllConditionVariable(scv);
}

void Condition_Variable_Destroy(Condition_Variable *cv)
{
    /*CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;*/
}

uint64_t N_CPU_Thread(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}