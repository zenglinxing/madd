/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

#define N 100

char func_compare(void *key1_, void *key2_, void *other_param)
{
    double *key1=key1_, *key2=key2_;
    if (*key1 < *key2) return BINARY_SEARCH_TREE_LESS;
    else if (*key1 == *key2) return BINARY_SEARCH_TREE_SAME;
    else return BINARY_SEARCH_TREE_GREATER;
}

int main(int argc,char *argv[])
{
    int i;
    RNG_MT_Param mt=RNG_MT_Init(10);
    double value[N];
    Binary_Search_Tree_Node node[N];
    Binary_Search_Tree T=Binary_Search_Tree_Make();
    for (i=0; i<N; i++){
        printf("i=%d\n",i);
        value[i] = Rand_MT(&mt);
        node[i].key = &value[i];
        node[i].left = node[i].right = NULL;
        Binary_Search_Tree_Insert(&T, &node[i], func_compare, NULL);
    }
    printf("finish inserting\n");
    return 0;
}
