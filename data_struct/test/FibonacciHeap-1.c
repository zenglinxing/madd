/* coding: utf-8 */
/*
Test Deleting Node
*/
#include<stdio.h>
#include"madd.h"

#define N 10000

struct Param{
    uint32_t id;
    double v;
};

char func_compare(void *key1_, void *key2_, void *other_param)
{
    struct Param *key1=key1_, *key2=key2_;
    if (key1->v < key2->v){
        return FIBONACCI_HEAP_LESS;
    }
    else if (key1->v > key2->v){
        return FIBONACCI_HEAP_GREATER;
    }
    else{
        return FIBONACCI_HEAP_SAME;
    }
}

int main(int argc,char *argv[])
{
    int i;
    RNG_MT_Param mt=RNG_MT_Init(40);
    Fibonacci_Heap_Node fhe[N];
    Fibonacci_Heap H=Fibonacci_Heap_Make();
    struct Param value[N];
    for (i=0; i<N; i++){
        value[i].id = i;
        value[i].v = Rand_MT(&mt);
        fhe[i].key = (void*)&value[i];
        Fibonacci_Heap_Insert(&H,&fhe[i],func_compare,NULL);
    }
    /* find the min value */
    uint64_t min_id=0;
    double min_value=value[0].v;
    for (i=1; i<N; i++){
        if (min_value > value[i].v){
            min_value = value[i].v;
            min_id = i;
        }
    }
    printf("min value: %f\tid: %llu\n", min_value, min_id);
    printf("Heap min: %f\tid: %llu\n", ((struct Param*)(H.min->key))->v ,((struct Param*)(H.min->key))->id);
    /*
    delete node
    There are 4 steps, which I had marked with ********** at the end of them
    */
    printf("======================\nDeleting Node Example\n======================\n");
    uint64_t id_delete = 5640;
    printf("deleting id: %llu\n",id_delete);
    printf("id %llu is the left sibling of id %llu\n", ((struct Param*)((&fhe[id_delete])->left->key))->id, ((struct Param*)((&fhe[id_delete])->key))->id );
    printf("id %llu is the right sibling of id %llu\n", ((struct Param*)((&fhe[id_delete])->right->key))->id, ((struct Param*)((&fhe[id_delete])->key))->id );
    Fibonacci_Heap_Delete(&H, &fhe[id_delete], func_compare, NULL);
    printf("H.n=%llu\n",H.n);
    printf("H.min\tid=%llu\tvalue=%f\n", ((struct Param*)(H.min->key))->id, ((struct Param*)(H.min->key))->v);
    printf("id %llu is the right sibling of id %llu's left sibling\n", ((struct Param*)((&fhe[id_delete])->left->right->key))->id, ((struct Param*)((&fhe[id_delete])->key))->id );
    printf("id %llu is the left sibling of id %llu's right sibling\n", ((struct Param*)((&fhe[id_delete])->right->left->key))->id, ((struct Param*)((&fhe[id_delete])->key))->id );
    printf("H.min's left sibling's id %llu\n", ((struct Param*)(H.min->left->key))->id );
    return 0;
}
