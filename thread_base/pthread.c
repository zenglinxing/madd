#include<stdlib.h>
#include<pthread.h>
#include"thread_base.h"

#ifdef _WIN32
#include<windows.h>
#elif defined(__APPLE__)
#include<sys/sysctl.h>
#else
#include<unistd.h>
#include<sys/sysinfo.h>
#endif

typedef void *(*thread_input_func)(void *);

Thread Thread_Create(void func(void*), void *param)
{
    thread_input_func tif = (thread_input_func)func;
    pthread_t *th = (pthread_t*)malloc(sizeof(pthread_t));
    pthread_create(th, NULL, tif, param);
    return th;
}

void Thread_Join(Thread th_)
{
    pthread_t *th=(pthread_t*)th_;
    int ret = pthread_join(*th, NULL);
    free(th);
}

void Thread_Detach(Thread th_)
{
    pthread_t *th=(pthread_t*)th_;
    int ret = pthread_detach(*th);
    free(th);
}

Mutex Mutex_Create(void)
{
    pthread_mutex_t *pmt = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    int ret = pthread_mutex_init(pmt, NULL);
    return pmt;
}

void Mutex_Lock(Mutex m)
{
    pthread_mutex_t *pmt=(pthread_mutex_t*)m;
    int ret = pthread_mutex_lock(pmt);
}

bool Mutex_Trylock(Mutex m)
{
    pthread_mutex_t *pmt=(pthread_mutex_t*)m;
    int ret = pthread_mutex_trylock(pmt);
    return ret == 0;
}

void Mutex_Unlock(Mutex m)
{
    pthread_mutex_t *pmt=(pthread_mutex_t*)m;
    int ret = pthread_mutex_unlock(pmt);
}

void Mutex_Destroy(Mutex m)
{
    pthread_mutex_t *pmt=(pthread_mutex_t*)m;
    int ret = pthread_mutex_destroy(pmt);
    free(pmt);
}

Condition_Variable Condition_Variable_Create(void)
{
    pthread_cond_t *cv=(pthread_cond_t*)malloc(sizeof(pthread_cond_t));
    pthread_cond_init(cv, NULL);
    return cv;
}

void Condition_Variable_Wait(Condition_Variable cv, Mutex m)
{
    pthread_cond_wait(cv, m);
}

void Condition_Variable_Wake(Condition_Variable cv)
{
    pthread_cond_signal(cv);
}

void Condition_Variable_Wake_All(Condition_Variable cv)
{
    pthread_cond_broadcast(cv);
}

void Condition_Variable_Destroy(Condition_Variable cv)
{
    pthread_cond_destroy(cv);
    free(cv);
}

uint64_t N_CPU_Thread(void)
{
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
