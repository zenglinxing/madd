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
    madd_thread_base_triggered = true;
    HANDLE *handle=(HANDLE*)malloc(sizeof(HANDLE));
    thread_input_func tif = func;
    *handle = CreateThread(NULL, 0, tif, param, 0, 0);
    return handle;
}

void Thread_Join(Thread th)
{
    madd_thread_base_triggered = true;
    HANDLE *handle=(HANDLE*)th;
    DWORD ret = WaitForSingleObject(*handle, INFINITE);
    BOOL ret_close = CloseHandle(*handle);
    free(handle);
}

void Thread_Detach(Thread th)
{
    madd_thread_base_triggered = true;
    HANDLE *handle=(HANDLE*)th;
    BOOL ret_close = CloseHandle(*handle);
    free(handle);
}

/* mutex */
void Mutex_Init(Mutex *m)
{
    madd_thread_base_triggered = true;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    InitializeCriticalSection(cs);
}

void Mutex_Destroy(Mutex *m)
{
    madd_thread_base_triggered = true;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    DeleteCriticalSection(cs);
}

void Mutex_Lock(Mutex *m)
{
    madd_thread_base_triggered = true;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    EnterCriticalSection((CRITICAL_SECTION*)cs);
}

bool Mutex_Trylock(Mutex *m)
{
    madd_thread_base_triggered = true;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    return TryEnterCriticalSection(cs);
}

void Mutex_Unlock(Mutex *m)
{
    madd_thread_base_triggered = true;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    LeaveCriticalSection(cs);
}

/* condition variable */
void Condition_Variable_Init(Condition_Variable *cv)
{
    madd_thread_base_triggered = true;
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    InitializeConditionVariable(scv);
}

void Condition_Variable_Destroy(Condition_Variable *cv)
{
    madd_thread_base_triggered = true;
    /*CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;*/
}

void Condition_Variable_Wait(Condition_Variable *cv, Mutex *m)
{
    madd_thread_base_triggered = true;
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    SleepConditionVariableCS(scv, cs, INFINITE);
}

bool Condition_Variable_Timed_Wait(Condition_Variable *cv, Mutex *m, double wait_sec)
{
    madd_thread_base_triggered = true;
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    CRITICAL_SECTION *cs = (CRITICAL_SECTION*)m->buf;
    DWORD ms = wait_sec * 1000;
    BOOL res = SleepConditionVariableCS(scv, cs, ms);
    return res;
}

void Condition_Variable_Wake(Condition_Variable *cv)
{
    madd_thread_base_triggered = true;
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    WakeConditionVariable(scv);
}

void Condition_Variable_Wake_All(Condition_Variable *cv)
{
    madd_thread_base_triggered = true;
    CONDITION_VARIABLE *scv = (CONDITION_VARIABLE*)cv->buf;
    WakeAllConditionVariable(scv);
}

/* read-write lock */
void RWLock_Init(RWLock *rw)
{
    madd_thread_base_triggered = true;
    InitializeSRWLock((SRWLOCK*)rw->buf);
}

void RWLock_Destroy(RWLock *rw)
{
    madd_thread_base_triggered = true;
    /*SRWLOCK *sl = (SRWLOCK*)rw->buf;*/
}

void RWLock_Read_Lock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    AcquireSRWLockShared((SRWLOCK*)rw->buf);
}

bool RWLock_Try_Read_Lock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    BOOLEAN ret = TryAcquireSRWLockShared((SRWLOCK*)rw->buf);
    return ret != 0;
}

void RWLock_Write_Lock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    AcquireSRWLockExclusive((SRWLOCK*)rw->buf);
}

bool RWLock_Try_Write_Lock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    BOOLEAN ret = TryAcquireSRWLockExclusive((SRWLOCK*)rw->buf);
    return ret != 0;
}

void RWLock_Read_Unlock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    ReleaseSRWLockShared((SRWLOCK*)rw->buf);
}

void RWLock_Write_Unlock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    ReleaseSRWLockExclusive((SRWLOCK*)rw->buf);
}

uint64_t N_CPU_Thread(void)
{
    madd_thread_base_triggered = true;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}