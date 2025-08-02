/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./thread_base/cpp_thread.cpp
*/
#include<stdlib.h>
#include<thread>
#include<mutex>
#include<shared_mutex>
#include<condition_variable>
#include<chrono>
#include<new>
#include<stdio.h>
#include<stdbool.h>
extern "C"{
#include"thread_base.h"
}

/* check if size of mutex etc. is larger than expected */
class CppThreadBase_SizeCheck{
    public:
    CppThreadBase_SizeCheck(){
        size_t size_mutex = sizeof(std::mutex), size_condition_variable = sizeof(sizeof(std::condition_variable));
        bool flag_fail = false;
        if (size_mutex > MADD_THREAD_BASE_MUTEX_LEN){
            printf("Madd Error!\nSize of mutex is %llu, larger than expected %d. Try to re-compile Madd after resetting macro MADD_THREAD_BASE_MUTEX_LEN.\n", size_mutex, MADD_THREAD_BASE_MUTEX_LEN);
            flag_fail = true;
        }
        if (size_condition_variable > MADD_THREAD_BASE_CONDITION_VARIABLE_LEN){
            printf("Madd Error!\nSize of condition variable is %llu, larger than expected %d. Try to re-compile Madd after resetting macro MADD_THREAD_BASE_MUTEX_LEN.\n", size_condition_variable, MADD_THREAD_BASE_CONDITION_VARIABLE_LEN);
            flag_fail = true;
        }
        if (flag_fail){
            exit(EXIT_FAILURE);
        }
    }
};

static CppThreadBase_SizeCheck threadbase_sizecheck;

extern "C"{

Thread Thread_Create(void func(void*), void *param)
{
    return new std::thread(func, param);
}

void Thread_Join(Thread th_)
{
    auto th=static_cast<std::thread*>(th_);
    if (th->joinable()){
        th->join();
    }
    delete th;
}

void Thread_Detach(Thread th_)
{
    auto th=static_cast<std::thread*>(th_);
    th->detach();
    delete th;
}

/* mutex */
void Mutex_Init(Mutex *m)
{
    new (m->buf) std::mutex();
}

void Mutex_Lock(Mutex *m)
{
    reinterpret_cast<std::mutex*>(m->buf)->lock();
}

bool Mutex_Trylock(Mutex *m)
{
    return reinterpret_cast<std::mutex*>(m->buf)->try_lock();
}

void Mutex_Unlock(Mutex *m)
{
    reinterpret_cast<std::mutex*>(m->buf)->unlock();
}

void Mutex_Destroy(Mutex *m)
{
    reinterpret_cast<std::mutex*>(m->buf)->~mutex();
}

/* condition variable */
void Condition_Variable_Init(Condition_Variable *cv)
{
    new (cv->buf) std::condition_variable();
}

void Condition_Variable_Wait(Condition_Variable *cv, Mutex *m)
{
    auto& mutex_ref = *reinterpret_cast<std::mutex*>(m->buf);
    std::unique_lock<std::mutex> lock(mutex_ref, std::adopt_lock);
    reinterpret_cast<std::condition_variable*>(cv->buf)->wait(lock);
    lock.release();
}

bool Condition_Variable_Timed_Wait(Condition_Variable *cv, Mutex *m, double wait_sec)
{
    int64_t ms_timeout = wait_sec*1000;
    auto timeout = std::chrono::seconds(ms_timeout);
    auto& mutex_ref = *reinterpret_cast<std::mutex*>(m->buf);
    std::unique_lock<std::mutex> lock(mutex_ref, std::adopt_lock);
    auto res = reinterpret_cast<std::condition_variable*>(cv->buf)->wait_for(lock, timeout);
    lock.release();
    return res != std::cv_status::timeout;
}

void Condition_Variable_Wake(Condition_Variable *cv)
{
    reinterpret_cast<std::condition_variable*>(cv->buf)->notify_one();
}

void Condition_Variable_Wake_All(Condition_Variable *cv)
{
    reinterpret_cast<std::condition_variable*>(cv->buf)->notify_all();
}

void Condition_Variable_Destroy(Condition_Variable *cv)
{
    reinterpret_cast<std::condition_variable*>(cv->buf)->~condition_variable();
}

uint64_t N_CPU_Thread(void)
{
    return std::thread::hardware_concurrency();
}


}