/* coding: utf-8 */
/*
Test for inserting & deleting
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
    if (key1->v < key2->v) return BINARY_SEARCH_TREE_LESS;
    else if (key1->v == key2->v) return BINARY_SEARCH_TREE_SAME;
    else return BINARY_SEARCH_TREE_GREATER;
}

int main(int argc,char *argv[])
{
    int i;
    RNG_MT_Param mt=RNG_MT_Init(20);
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
    /* deleting node */
    uint64_t id_delete = 30;
    Binary_Search_Tree_Delete(&T, &node[id_delete]);
    printf("deleted\n");
    /* search */
    uint64_t id_search = 90;
    Binary_Search_Tree_Node *node_search=Binary_Search_Tree_Search(T.root, &value[id_search], func_compare, NULL);
    printf("searched, searched id=%llu, found id=%llu\n",id_search,((struct Param*)(node_search->key))->id);
    return 0;
}
