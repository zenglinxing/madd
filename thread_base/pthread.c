/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./thread_base/pthread.c
*/
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<pthread.h>
#include"thread_base.h"

#ifdef _WIN32
#include<windows.h>
#elif defined(__APPLE__)
#include<sys/sysctl.h>
#else
#include<unistd.h>
#include<sys/sysinfo.h>
#include<sys/time.h>
#endif

typedef void *(*thread_input_func)(void *);

Thread Thread_Create(void func(void*), void *param)
{
    madd_thread_base_triggered = true;
    thread_input_func tif = (thread_input_func)func;
    pthread_t *th = (pthread_t*)malloc(sizeof(pthread_t));
    pthread_create(th, NULL, tif, param);
    return th;
}

void Thread_Join(Thread th_)
{
    madd_thread_base_triggered = true;
    pthread_t *th=(pthread_t*)th_;
    int ret = pthread_join(*th, NULL);
    free(th);
}

void Thread_Detach(Thread th_)
{
    madd_thread_base_triggered = true;
    pthread_t *th=(pthread_t*)th_;
    int ret = pthread_detach(*th);
    free(th);
}

/* mutex */
void Mutex_Init(Mutex *m)
{
    madd_thread_base_triggered = true;
    pthread_mutex_t *pmt = (pthread_mutex_t*)m->buf;
    int ret = pthread_mutex_init(pmt, NULL);
}

void Mutex_Destroy(Mutex *m)
{
    madd_thread_base_triggered = true;
    pthread_mutex_t *pmt=(pthread_mutex_t*)m->buf;
    int ret = pthread_mutex_destroy(pmt);
}

void Mutex_Lock(Mutex *m)
{
    madd_thread_base_triggered = true;
    pthread_mutex_t *pmt=(pthread_mutex_t*)m->buf;
    int ret = pthread_mutex_lock(pmt);
}

bool Mutex_Trylock(Mutex *m)
{
    madd_thread_base_triggered = true;
    pthread_mutex_t *pmt=(pthread_mutex_t*)m->buf;
    int ret = pthread_mutex_trylock(pmt);
    return ret == 0;
}

void Mutex_Unlock(Mutex *m)
{
    madd_thread_base_triggered = true;
    pthread_mutex_t *pmt=(pthread_mutex_t*)m->buf;
    int ret = pthread_mutex_unlock(pmt);
}

/* condition variable */
void Condition_Variable_Init(Condition_Variable *cv)
{
    madd_thread_base_triggered = true;
    pthread_cond_init((pthread_cond_t*)cv->buf, NULL);
}

void Condition_Variable_Destroy(Condition_Variable *cv)
{
    madd_thread_base_triggered = true;
    pthread_cond_destroy((pthread_cond_t*)cv->buf);
}

void Condition_Variable_Wait(Condition_Variable *cv, Mutex *m)
{
    madd_thread_base_triggered = true;
    pthread_cond_wait((pthread_cond_t*)cv->buf, (pthread_mutex_t*)m->buf);
}

bool Condition_Variable_Timed_Wait(Condition_Variable *cv, Mutex *m, double wait_sec)
{
    madd_thread_base_triggered = true;
    if (wait_sec <= 0) return false;
    struct timespec time_now, time_out;
    timespec_get(&time_now, TIME_UTC);

    const uint64_t ns_count = 1e9; /* nano seconds in a second */
    int64_t wait_in_sec = floor(wait_sec), wait_nsec = round((wait_sec - wait_in_sec)*ns_count);
    uint64_t until_ns = wait_nsec + time_now.tv_nsec, until_s = wait_in_sec + time_now.tv_sec;
    if (until_ns >= ns_count){
        until_s += until_ns / ns_count;
        until_ns = until_ns % ns_count;
    }
    time_out.tv_nsec = until_ns;
    time_out.tv_sec = until_s;
    int res = pthread_cond_timedwait((pthread_cond_t*)cv->buf, (pthread_mutex_t*)m->buf, &time_out);
    return res == 0;
}

void Condition_Variable_Wake(Condition_Variable *cv)
{
    madd_thread_base_triggered = true;
    pthread_cond_signal((pthread_cond_t*)cv->buf);
}

void Condition_Variable_Wake_All(Condition_Variable *cv)
{
    madd_thread_base_triggered = true;
    pthread_cond_broadcast((pthread_cond_t*)cv->buf);
}

/* read-write lock */
void RWLock_Init(RWLock *rw)
{
    madd_thread_base_triggered = true;
    pthread_rwlock_init((pthread_rwlock_t*)rw->buf, NULL);
}

void RWLock_Destroy(RWLock *rw)
{
    madd_thread_base_triggered = true;
    pthread_rwlock_destroy((pthread_rwlock_t*)rw->buf);
}

void RWLock_Read_Lock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    pthread_rwlock_rdlock((pthread_rwlock_t*)rw->buf);
}

bool RWLock_Try_Read_Lock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    int res = pthread_rwlock_tryrdlock((pthread_rwlock_t*)rw->buf);
    return res == 0;
}

void RWLock_Write_Lock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    pthread_rwlock_wrlock((pthread_rwlock_t*)rw->buf);
}

bool RWLock_Try_Write_Lock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    int res = pthread_rwlock_trywrlock((pthread_rwlock_t*)rw->buf);
    return res == 0;
}

void RWLock_Read_Unlock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    int res = pthread_rwlock_unlock((pthread_rwlock_t*)rw->buf);
}

void RWLock_Write_Unlock(RWLock *rw)
{
    madd_thread_base_triggered = true;
    int res = pthread_rwlock_unlock((pthread_rwlock_t*)rw->buf);
}

uint64_t N_CPU_Thread(void)
{
    madd_thread_base_triggered = true;
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
#elif defined(__APPLE__)
    int count;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &size, NULL, 0) != 0) 
        return 1;
    return count;
#else
    uint64_t n_core = sysconf(_SC_NPROCESSORS_ONLN);
    return (n_core) ? n_core : 1;
#endif
}
