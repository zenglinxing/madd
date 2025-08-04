/* coding: utf-8 */
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include "thread_base.h"

#define NUM_READERS 5
#define NUM_WRITERS 2
#define TEST_LOOPS 3

RWLock rwlock;
int shared_data = 0;
int readers_active = 0;
int writers_active = 0;
Mutex stats_mutex;

void clock_sleep_ms(long long ms)
{
    double sec = ms / 1000., pass;
    clock_t c_start=clock(), c_now;
    do {
        c_now = clock();
        pass = (c_now - c_start) / (double)CLOCKS_PER_SEC;
    } while (pass < sec);
}

void reader_thread(void* arg) {
    int id = *(int*)arg;
    
    for (int i = 0; i < TEST_LOOPS; i++) {
        // 获取读锁
        RWLock_Read_Lock(&rwlock);
        
        // 更新统计信息
        Mutex_Lock(&stats_mutex);
        readers_active++;
        printf("Reader %d started reading (readers: %d, writers: %d)\n", 
               id, readers_active, writers_active);
        Mutex_Unlock(&stats_mutex);
        
        // 模拟读取操作
        printf("Reader %d reads value: %d\n", id, shared_data);
        
        // 更新统计信息
        Mutex_Lock(&stats_mutex);
        readers_active--;
        printf("Reader %d finished reading (readers: %d, writers: %d)\n", 
               id, readers_active, writers_active);
        Mutex_Unlock(&stats_mutex);
        
        // 释放读锁
        RWLock_Read_Unlock(&rwlock);
        
        // 稍微等待一下
        clock_sleep_ms(100);
    }
    
    free(arg);
}

void writer_thread(void* arg) {
    int id = *(int*)arg;
    
    for (int i = 0; i < TEST_LOOPS; i++) {
        // 尝试获取写锁
        if (RWLock_Try_Write_Lock(&rwlock)) {
            printf("Writer %d acquired write lock immediately\n", id);
        } else {
            printf("Writer %d waiting for write lock...\n", id);
            RWLock_Write_Lock(&rwlock);
        }
        
        // 更新统计信息
        Mutex_Lock(&stats_mutex);
        writers_active++;
        printf("Writer %d started writing (readers: %d, writers: %d)\n", 
               id, readers_active, writers_active);
        Mutex_Unlock(&stats_mutex);
        
        // 模拟写入操作
        shared_data++;
        printf("Writer %d writes new value: %d\n", id, shared_data);
        
        // 更新统计信息
        Mutex_Lock(&stats_mutex);
        writers_active--;
        printf("Writer %d finished writing (readers: %d, writers: %d)\n", 
               id, readers_active, writers_active);
        Mutex_Unlock(&stats_mutex);
        
        // 释放写锁
        RWLock_Write_Unlock(&rwlock);
        
        // 稍微等待一下
        clock_sleep_ms(200);
    }
    
    free(arg);
}

void test_basic_rwlock_operations() {
    printf("\n=== Testing basic RWLock operations ===\n");
    
    RWLock_Init(&rwlock);
    
    // 测试读锁
    printf("Acquiring read lock...\n");
    RWLock_Read_Lock(&rwlock);
    printf("Read lock acquired\n");
    
    // 测试尝试获取写锁（应该失败）
    printf("Trying to get write lock (should fail)...\n");
    bool try_write = RWLock_Try_Write_Lock(&rwlock);
    printf("Try write lock result: %s\n", try_write ? "success (UNEXPECTED)" : "fail (expected)");
    
    // 释放读锁
    RWLock_Read_Unlock(&rwlock);
    printf("Read lock released\n");
    
    // 测试写锁
    printf("Acquiring write lock...\n");
    RWLock_Write_Lock(&rwlock);
    printf("Write lock acquired\n");
    
    // 测试尝试获取读锁（应该失败）
    printf("Trying to get read lock (should fail)...\n");
    bool try_read = RWLock_Try_Read_Lock(&rwlock);
    printf("Try read lock result: %s\n", try_read ? "success (UNEXPECTED)" : "fail (expected)");
    
    // 释放写锁
    RWLock_Write_Unlock(&rwlock);
    printf("Write lock released\n");
    
    RWLock_Destroy(&rwlock);
}

void test_concurrent_access() {
    printf("\n=== Testing concurrent access ===\n");
    
    RWLock_Init(&rwlock);
    Mutex_Init(&stats_mutex);
    shared_data = 0;
    readers_active = 0;
    writers_active = 0;
    
    Thread readers[NUM_READERS];
    Thread writers[NUM_WRITERS];
    
    // 创建读者线程
    for (int i = 0; i < NUM_READERS; i++) {
        int* id = malloc(sizeof(int));
        *id = i + 1;
        readers[i] = Thread_Create(reader_thread, id);
    }
    
    // 创建写者线程
    for (int i = 0; i < NUM_WRITERS; i++) {
        int* id = malloc(sizeof(int));
        *id = i + 1;
        writers[i] = Thread_Create(writer_thread, id);
    }
    
    // 等待所有读者线程完成
    for (int i = 0; i < NUM_READERS; i++) {
        Thread_Join(readers[i]);
    }
    
    // 等待所有写者线程完成
    for (int i = 0; i < NUM_WRITERS; i++) {
        Thread_Join(writers[i]);
    }
    
    printf("Final shared data value: %d (expected: %d)\n", 
           shared_data, NUM_WRITERS * TEST_LOOPS);
    
    RWLock_Destroy(&rwlock);
    Mutex_Destroy(&stats_mutex);
}

int main() {
    printf("RWLock Test Program\n");
    printf("Number of CPU cores: %llu\n", N_CPU_Thread());
    
    // 测试基本操作
    test_basic_rwlock_operations();
    
    // 测试并发访问
    test_concurrent_access();
    
    printf("\nAll tests completed!\n");
    return 0;
}