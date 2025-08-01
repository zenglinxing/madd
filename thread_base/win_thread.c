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

Mutex Mutex_Create(void)
{
    CRITICAL_SECTION *cs=(CRITICAL_SECTION*)malloc(sizeof(CRITICAL_SECTION));
    InitializeCriticalSection(cs);
    return cs;
}

void Mutex_Lock(Mutex m)
{
    CRITICAL_SECTION *cs=(CRITICAL_SECTION*)m;
    EnterCriticalSection(cs);
}

bool Mutex_Trylock(Mutex m)
{
    CRITICAL_SECTION *cs=(CRITICAL_SECTION*)m;
    return TryEnterCriticalSection(cs);
}

void Mutex_Unlock(Mutex m)
{
    CRITICAL_SECTION *cs=(CRITICAL_SECTION*)m;
    LeaveCriticalSection(cs);
}

void Mutex_Destroy(Mutex m)
{
    CRITICAL_SECTION *cs=(CRITICAL_SECTION*)m;
    DeleteCriticalSection(cs);
    free(cs);
}

Condition_Variable Condition_Variable_Create(void)
{
    CONDITION_VARIABLE *cv = (CONDITION_VARIABLE*)malloc(sizeof(CONDITION_VARIABLE));
    InitializeConditionVariable(cv);
    return cv;
}

void Condition_Variable_Wait(Condition_Variable cv, Mutex m)
{
    SleepConditionVariableCS(cv, m, INFINITE);
}

void Condition_Variable_Wake(Condition_Variable cv)
{
    WakeConditionVariable(cv);
}

void Condition_Variable_Wake_All(Condition_Variable cv)
{
    WakeAllConditionVariable(cv);
}

void Condition_Variable_Destroy(Condition_Variable cv)
{
    free((CONDITION_VARIABLE*)cv);
}

uint64_t N_CPU_Thread(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}