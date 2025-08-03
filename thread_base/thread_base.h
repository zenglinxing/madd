/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./thread_base/thread_base.h
*/
#ifndef _THREAD_BASE_H
#define _THREAD_BASE_H

#include<stdint.h>
#include<stdbool.h>

#define MADD_THREAD_BASE_MUTEX_LEN 40
#define MADD_THREAD_BASE_CONDITION_VARIABLE_LEN 48
#define MADD_THREAD_BASE_RWLOCK_LEN 40

typedef void* Thread;
typedef union{
    unsigned char buf[MADD_THREAD_BASE_MUTEX_LEN];
    long long aligner;
} Mutex;
typedef union{
    unsigned char buf[MADD_THREAD_BASE_CONDITION_VARIABLE_LEN];
    long long aligner;
} Condition_Variable;
typedef union{
    unsigned char buf[MADD_THREAD_BASE_RWLOCK_LEN];
    long long aligner;
} RWLock;

Thread Thread_Create(void func(void*), void* param);
void Thread_Join(Thread th);
void Thread_Detach(Thread th);

void Mutex_Init(Mutex *m);
void Mutex_Lock(Mutex *m);
bool Mutex_Trylock(Mutex *m);
void Mutex_Unlock(Mutex *m);
void Mutex_Destroy(Mutex *m);

void Condition_Variable_Init(Condition_Variable *cv);
void Condition_Variable_Wait(Condition_Variable *cv, Mutex *m);
bool Condition_Variable_Timed_Wait(Condition_Variable *cv, Mutex *m, double wait_sec);
void Condition_Variable_Wake(Condition_Variable *cv);
void Condition_Variable_Wake_All(Condition_Variable *cv);
void Condition_Variable_Destroy(Condition_Variable *cv);

void RWLock_Init(RWLock *rw);
void RWLock_Destroy(RWLock *rw);
void RWLock_Read_Lock(RWLock *rw);
bool RWLock_Try_Read_Lock(RWLock *rw);
void RWLock_Write_Lock(RWLock *rw);
bool RWLock_Try_Write_Lock(RWLock *rw);
void RWLock_Read_Unlock(RWLock *rw);
void RWLock_Write_Unlock(RWLock *rw);

uint64_t N_CPU_Thread(void);

#endif /* _THREAD_BASE_H */