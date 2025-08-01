#include<stdio.h>
#include"thread_base.h"

struct Param{
    int a, b;
};

void thread_func(void *param_)
{
    struct Param *param=param_;
    param->b = param->a + 1;
}

int main(int argc, char *argv[])
{
    struct Param param={.a=2};
    Thread ct=Thread_Create(thread_func, &param);
    Thread_Join(ct);
    printf("a:\t%d\n", param.a);
    printf("b:\t%d\n", param.b);
    return 0;
}