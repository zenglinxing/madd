/* coding: utf-8 */
/*
Test for searching minimum & maximum
*/
#include<stdio.h>
#include"madd.h"

#define N 1000

struct Param{
    uint64_t id;
    double v;
};

char func_compare(void *key1_, void *key2_, void *other_param)
{
    struct Param *key1=key1_, *key2=key2_;
    if (key1->v < key2->v) return MADD_LESS;
    else if (key1->v == key2->v) return MADD_SAME;
    else return MADD_GREATER;
}

int main(int argc,char *argv[])
{
    int i;
    RNG_MT_Param mt=RNG_MT_Init(30);
    struct Param value[N];
    Binary_Search_Tree_Node node[N];
    Binary_Search_Tree T=Binary_Search_Tree_Make();
    for (i=0; i<N; i++){
        /*printf("i=%d\n",i);*/
        value[i].id = i;
        value[i].v = Rand_MT(&mt);
        node[i].key = &value[i];
        node[i].left = node[i].right = NULL;
        Binary_Search_Tree_Insert(&T, &node[i], func_compare, NULL);
    }
    printf("finish inserting\n");
    /* maximum & minimum node */
    Binary_Search_Tree_Node *node_min, *node_max;
    struct Param *key;
    node_min = Binary_Search_Tree_Minimum(T.root);
    node_max = Binary_Search_Tree_Maximum(T.root);
    double v_min, v_max;
    for (i=0; i<N; i++){
        if (i==0){
            v_min = v_max = value[i].v;
        }
        else{
            v_min = (value[i].v < v_min) ? value[i].v : v_min;
            v_max = (value[i].v > v_max) ? value[i].v : v_max;
        }
    }
    printf("min:\t%f\t%f\n", ((struct Param*)node_min->key)->v, v_min);
    printf("max:\t%f\t%f\n", ((struct Param*)node_max->key)->v, v_max);
    return 0;
}
