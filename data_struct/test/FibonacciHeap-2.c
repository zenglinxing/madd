/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

#define N1 0
#define N2 6

char func(void *a_, void *b_, void *other_param)
{
    double *a=a_, *b=b_;
    return *a < *b;
}

int main(int argc,char *argv[])
{
    int i;
    double key[N1+N2]={1, 3, 2, 4, 5, 7};
    Fibonacci_Heap_Node *fhe=(Fibonacci_Heap_Node*)malloc((N1+N2)*sizeof(Fibonacci_Heap_Node));
    Fibonacci_Heap H1=Fibonacci_Heap_Make(), H2=Fibonacci_Heap_Make(), H;
    for (i=0; i<N1+N2; i++){
        fhe[i].key = &key[i];
        if (i<N1){
            Fibonacci_Heap_Insert(&H1, fhe+i, func, NULL);
        }
        else{
            Fibonacci_Heap_Insert(&H2, fhe+i, func, NULL);
        }
    }
    H = Fibonacci_Heap_Union(H1,H2,func,NULL);
    if (H.min == NULL){
        printf("No H.min\n");
    }
    else{
        printf("H.min=%u\nH.min key=%f\n", H.n, *(double*)H.min->key);
    }
    free(fhe);
    return 0;
}
