/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

#define N 1000

struct Param{
    double v;
    uint32_t id;
};

char func(void *key1_, void *key2_, void *other_param)
{
    struct Param *key1=key1_, *key2=key2_;
    if (key1->v < key2->v) return MADD_LESS;
    else if (key1->v == key2->v) return MADD_SAME;
    else return MADD_GREATER;
}

int main(int argc,char *argv[])
{
    int i;
    struct Param value[N];
    RB_Tree_Node rtn[N];
    RB_Tree T;
    RB_Tree_Init(&T);
    RNG_MT_Param mt=RNG_MT_Init(10);
    for (i=0; i<N; i++){
        value[i].v = Rand_MT(&mt);
        value[i].id = i;
        rtn[i].key = &value[i];
        //printf("Element %d inserting. value=%f\n",i, value[i].v);
        RB_Tree_Insert(&T, &rtn[i], func, NULL, RB_TREE_ENABLE_SAME_KEY);
        //printf("Element %d inserted. color=%s.\n",i, rtn[i].color=='r' ? "red" : "black");
    }
    /* delete nodes iteratively */
    for (i=0; i<N; i++){
        printf("===Delete node id %d\n", i);
        RB_Tree_Delete(&T, rtn+i);
    }
    return 0;
}
