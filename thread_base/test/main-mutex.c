#include<stdio.h>
#include"thread_base.h"

#define N 5

struct Param{
    int id, a, b;
};

int count = 0, ind=0;
int arr[N];
Mutex cm;

void thread_func(void *param_)
{
    struct Param *param=param_;
    param->a = param->id * 3;
    param->b = param->a + 2;
    Mutex_Lock(&cm);
    count++;
    arr[ind] = count;
    ind++;
    printf("id = %d\n\ta,b = %d,%d\n\tcount = %d\n", param->id, param->a, param->b, count);
    Mutex_Unlock(&cm);
}

int main(int argc, char *argv[])
{
    Mutex_Init(&cm);

    struct Param param[N];
    Thread ct[N];
    int i;
    for (i=0; i<N; i++){
        param[i].id = i;
        ct[i] = Thread_Create(thread_func, param+i);
    }
    for (i=0; i<N; i++){
        Thread_Join(ct[i]);
    }
    Mutex_Destroy(&cm);

    for (i=0; i<N; i++){
        printf("ind:%d\tarr=%d\n", i, arr[i]);
    }
    return 0;
}