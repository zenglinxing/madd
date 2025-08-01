#ifndef _THREAD_BASE_H
#define _THREAD_BASE_H

#include<stdint.h>
#include<stdbool.h>
#ifdef MADD_ENABLE_MULTITHREAD
#include<stdatomic.h>
#endif

typedef void* Thread;
typedef void* Mutex;
typedef void* Condition_Variable;

Thread Thread_Create(void func(void*), void* param);
void Thread_Join(Thread th);
void Thread_Detach(Thread th);

Mutex Mutex_Create(void);
void Mutex_Lock(Mutex m);
bool Mutex_Trylock(Mutex m);
void Mutex_Unlock(Mutex m);
void Mutex_Destroy(Mutex m);

Condition_Variable Condition_Variable_Create(void);
void Condition_Variable_Wait(Condition_Variable cv, Mutex m);
void Condition_Variable_Wake(Condition_Variable cv);
void Condition_Variable_Wake_All(Condition_Variable cv);
void Condition_Variable_Destroy(Condition_Variable cv);

uint64_t N_CPU_Thread(void);

#endif /* _THREAD_BASE_H */