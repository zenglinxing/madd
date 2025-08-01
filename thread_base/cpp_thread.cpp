#include<thread>
#include<mutex>
#include<condition_variable>

extern "C"{

#include"thread_base.h"

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

Mutex Mutex_Create(void)
{
    return new std::mutex;
}

void Mutex_Lock(Mutex m)
{
    static_cast<std::mutex*>(m)->lock();
}

bool Mutex_Trylock(Mutex m)
{
    return static_cast<std::mutex*>(m)->try_lock();
}

void Mutex_Unlock(Mutex m)
{
    static_cast<std::mutex*>(m)->unlock();
}

void Mutex_Destroy(Mutex m)
{
    delete static_cast<std::mutex*>(m);
}

Condition_Variable Condition_Variable_Create(void)
{
    return new std::condition_variable;
}

void Condition_Variable_Wait(Condition_Variable cv, Mutex m)
{
    std::condition_variable *scv=static_cast<std::condition_variable*>(cv);
    std::mutex *sm = static_cast<std::mutex*>(m);
    std::unique_lock<std::mutex> lock(*sm, std::adopt_lock);
    scv->wait(lock);
    lock.release();
}

void Condition_Variable_Wake(Condition_Variable cv)
{
    std::condition_variable *scv=static_cast<std::condition_variable*>(cv);
    scv->notify_one();
}

void Condition_Variable_Wake_All(Condition_Variable cv)
{
    std::condition_variable *scv=static_cast<std::condition_variable*>(cv);
    scv->notify_all();
}

void Condition_Variable_Destroy(Condition_Variable cv)
{
    delete static_cast<std::condition_variable*>(cv);
}

uint64_t N_CPU_Thread(void)
{
    return std::thread::hardware_concurrency();
}


}