/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./data_struct/binary_search_tree.h
*/
#ifndef _BINARY_SEARCH_TREE_H
#define _BINARY_SEARCH_TREE_H

#include<stdint.h>
#include<stdlib.h>
#include<stdbool.h>
#include"../thread_base/thread_base.h"

struct _Binary_Search_Tree_Node{
    void *key;
    struct _Binary_Search_Tree_Node *left,*right,*p;
};

typedef struct _Binary_Search_Tree_Node Binary_Search_Tree_Node;

typedef struct{
    bool flag_multithread;
    Binary_Search_Tree_Node *root;
    RWLock rwlock;
} Binary_Search_Tree;

bool Binary_Search_Tree_Init(Binary_Search_Tree *T);
bool Binary_Search_Tree_Enable_Multithread(Binary_Search_Tree *T);
Binary_Search_Tree_Node *Binary_Search_Tree_Search(Binary_Search_Tree *T, Binary_Search_Tree_Node *x, void *k,
                                                   char func(void *key1,void *key2,void *other_param), void *other_param);
Binary_Search_Tree_Node *Binary_Search_Tree_Minimum(Binary_Search_Tree *T, Binary_Search_Tree_Node *x);
Binary_Search_Tree_Node *Binary_Search_Tree_Maximum(Binary_Search_Tree *T, Binary_Search_Tree_Node *x);
Binary_Search_Tree_Node *Binary_Search_Tree_Successor(Binary_Search_Tree *T, Binary_Search_Tree_Node *x);
Binary_Search_Tree_Node *Binary_Search_Tree_Predecessor(Binary_Search_Tree *T, Binary_Search_Tree_Node *x);
void Binary_Search_Tree_Insert(Binary_Search_Tree *T, Binary_Search_Tree_Node *z,
                               char func(void *key1,void *key2,void *other_param), void *other_param);
void Binary_Search_Tree_Transplant(Binary_Search_Tree *T,
                                   Binary_Search_Tree_Node *u, Binary_Search_Tree_Node *v);
void Binary_Search_Tree_Delete(Binary_Search_Tree *T, Binary_Search_Tree_Node *z);

#endif /* _BINARY_SEARCH_TREE_H */