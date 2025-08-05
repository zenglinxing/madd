/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
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
    struct Param *value=(struct Param*)malloc(N*sizeof(struct Param));
    RB_Tree_Node *rtn=(RB_Tree_Node*)malloc(N*sizeof(RB_Tree_Node));
    RB_Tree T;
    RB_Tree_Create(&T);
    RNG_MT_Param mt=RNG_MT_Init(10);
    for (i=0; i<N; i++){
        value[i].v = Rand_MT(&mt);
        value[i].id = i;
        rtn[i].key = &value[i];
        printf("Element %d inserting. value=%f\n",i, value[i].v);
        RB_Tree_Insert(&T, &rtn[i], func, NULL, RB_TREE_ENABLE_SAME_KEY);
        /*printf("Element %d inserted. color=%s.\n",i, rtn[i].color=='r' ? "red" : "black");*/
    }
    free(value);
    free(rtn);
    return 0;
}
