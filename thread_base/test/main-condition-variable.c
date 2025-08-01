#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "thread_base.h"  // 包含你的线程库头文件

#define BUFFER_SIZE 5
#define PRODUCERS 2
#define CONSUMERS 2
#define TOTAL_ITEMS 20

// 共享数据结构
typedef struct {
    int buffer[BUFFER_SIZE];
    int count;
    int in;
    int out;
    Mutex mutex;
    Condition_Variable not_full;
    Condition_Variable not_empty;
} SharedData;

SharedData data;

// 生产者线程函数
void producer(void *arg) {
    int thread_id = (int)(uintptr_t)arg; // 使用参数作为线程ID
    
    for (int i = 0; i < TOTAL_ITEMS/PRODUCERS; i++) {
        Mutex_Lock(data.mutex);
        
        // 等待缓冲区有空位
        while (data.count == BUFFER_SIZE) {
            printf("Producer %d: Buffer full, waiting...\n", thread_id);
            Condition_Variable_Wait(data.not_full, data.mutex);
        }
        
        // 生产项目
        int item = rand() % 100;
        data.buffer[data.in] = item;
        data.in = (data.in + 1) % BUFFER_SIZE;
        data.count++;
        
        printf("Producer %d: Produced %d (count=%d)\n", 
               thread_id, item, data.count);
        
        // 通知消费者
        Condition_Variable_Wake(data.not_empty);
        Mutex_Unlock(data.mutex);
        
        // 模拟工作时间
        //struct timespec sleep_time = {0, (rand() % 100) * 1000000}; // 毫秒转纳秒
        //nanosleep(&sleep_time, NULL);
    }
}

// 消费者线程函数
void consumer(void *arg) {
    int thread_id = (int)(uintptr_t)arg; // 使用参数作为线程ID
    
    for (int i = 0; i < TOTAL_ITEMS/CONSUMERS; i++) {
        Mutex_Lock(data.mutex);
        
        // 等待缓冲区有数据
        while (data.count == 0) {
            printf("Consumer %d: Buffer empty, waiting...\n", thread_id);
            Condition_Variable_Wait(data.not_empty, data.mutex);
        }
        
        // 消费项目
        int item = data.buffer[data.out];
        data.out = (data.out + 1) % BUFFER_SIZE;
        data.count--;
        
        printf("Consumer %d: Consumed %d (count=%d)\n", 
               thread_id, item, data.count);
        
        // 通知生产者
        Condition_Variable_Wake(data.not_full);
        Mutex_Unlock(data.mutex);
        
        // 模拟工作时间
        //struct timespec sleep_time = {0, (rand() % 100) * 1000000}; // 毫秒转纳秒
        //nanosleep(&sleep_time, NULL);
    }
}

int main() {
    Thread producers[PRODUCERS];
    Thread consumers[CONSUMERS];
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    // 初始化共享数据
    data.count = 0;
    data.in = 0;
    data.out = 0;
    
    // 创建同步对象
    data.mutex = Mutex_Create();
    data.not_full = Condition_Variable_Create();
    data.not_empty = Condition_Variable_Create();
    
    printf("System has %lu CPU cores\n", N_CPU_Core());
    printf("Starting %d producers and %d consumers\n", PRODUCERS, CONSUMERS);
    
    // 创建生产者线程
    for (int i = 0; i < PRODUCERS; i++) {
        // 传递线程ID作为参数
        producers[i] = Thread_Create(producer, (void*)(uintptr_t)(i+1));
    }
    
    // 创建消费者线程
    for (int i = 0; i < CONSUMERS; i++) {
        // 传递线程ID作为参数
        consumers[i] = Thread_Create(consumer, (void*)(uintptr_t)(i+1));
    }
    
    // 等待生产者完成
    for (int i = 0; i < PRODUCERS; i++) {
        Thread_Join(producers[i]);
    }
    
    // 等待消费者完成
    for (int i = 0; i < CONSUMERS; i++) {
        Thread_Join(consumers[i]);
    }
    
    // 清理资源
    Mutex_Destroy(data.mutex);
    Condition_Variable_Destroy(data.not_full);
    Condition_Variable_Destroy(data.not_empty);
    
    printf("All threads completed successfully\n");
    return 0;
}