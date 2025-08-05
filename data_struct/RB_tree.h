/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./data_struct/RB_tree.h

Red-Black Tree
*/
#ifndef _RB_TREE_H
#define _RB_TREE_H

#include<stdint.h>
#include<stdlib.h>
#include<stdbool.h>
#include"../thread_base/thread_base.h"

#define RB_TREE_SUCCESS 0
#define RB_TREE_NULL 1

#define RB_TREE_DISABLE_SAME_KEY 0
#define RB_TREE_ENABLE_SAME_KEY 1

struct _RB_Tree_Node{
    char color;
    void *key;
    struct _RB_Tree_Node *left,*right,*p;
};

typedef struct _RB_Tree_Node RB_Tree_Node;

typedef struct{
    bool flag_multithread;
    RB_Tree_Node *root, nil;
    RWLock rwlock;
} RB_Tree;

bool RB_Tree_Init(RB_Tree *T);
bool RB_Tree_Enable_Multithread(RB_Tree *T);
RB_Tree_Node *RB_Tree_Search(RB_Tree *T, RB_Tree_Node *x, void *k,
                             char func(void *key1,void *key2,void *other_param), void *other_param,
                             int flag_allow_same_key);
RB_Tree_Node *RB_Tree_Minimum(RB_Tree *T, RB_Tree_Node *x);
RB_Tree_Node *RB_Tree_Maximum(RB_Tree *T, RB_Tree_Node *x);
int RB_Tree_Left_Rotate(RB_Tree *T, RB_Tree_Node *x);
int RB_Tree_Right_Rotate(RB_Tree *T, RB_Tree_Node *x);
int RB_Tree_Insert_Fixup(RB_Tree *T, RB_Tree_Node *z);
void RB_Tree_Insert(RB_Tree *T, RB_Tree_Node *z,
                    char func(void *key1,void *key2,void *other_param), void *other_param,
                    int flag_allow_same_key);
void RB_Tree_Transplant(RB_Tree *T, RB_Tree_Node *u, RB_Tree_Node *v);
int RB_Tree_Delete_Fixup(RB_Tree *T, RB_Tree_Node *x);
void RB_Tree_Delete(RB_Tree *T, RB_Tree_Node *z);

#endif /* _RB_TREE_H */