/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./data_struct/binary_search_tree.c
*/

#include<stdint.h>
#include<stdlib.h>
#include"binary_search_tree.h"
#include"../basic/basic.h"
/*#include"../thread_base/thread_base.h"*/

#define BST_READ_LOCK(T) \
    if (T->flag_multithread){ \
        RWLock_Read_Lock(&T->rwlock); \
    } \

#define BST_READ_UNLOCK(T) \
    if (T->flag_multithread){ \
        RWLock_Read_Unlock(&T->rwlock); \
    } \

#define BST_WRITE_LOCK(T) \
    if (T->flag_multithread){ \
        RWLock_Write_Lock(&T->rwlock); \
    } \

#define BST_WRITE_UNLOCK(T) \
    if (T->flag_multithread){ \
        RWLock_Write_Unlock(&T->rwlock); \
    } \

bool Binary_Search_Tree_Init(Binary_Search_Tree *T)
{
    if (T == NULL){
        Madd_Error_Add(MADD_ERROR, L"Binary_Search_Tree_Init: BST pointer is NULL.");
        return false;
    }
    T->root = NULL;
    T->flag_multithread = false;
    return true;
}

bool Binary_Search_Tree_Enable_Multithread(Binary_Search_Tree *T)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (T->flag_multithread){
        Madd_Error_Add(MADD_ERROR, L"Binary_Search_Tree_Enable_Multithread: BST already has read-write lock initialized.");
        return false;
    }
    RWLock_Init(&T->rwlock);
    T->flag_multithread = true;
    return true;
#else
    T->flag_multithread = false;
    Madd_Error_Add(MADD_WARNING, L"Binary_Search_Tree_Enable_Multithread: Madd lib multithread wasn't enabled during compiling. Tried to enable Madd's multithread and re-compile Madd.");
    return false;
#endif
}

Binary_Search_Tree_Node *Binary_Search_Tree_Search(Binary_Search_Tree *T, Binary_Search_Tree_Node *x, void *k, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    if (T == NULL){
        Madd_Error_Add(MADD_ERROR, L"Binary_Search_Tree_Search: get BST pointer NULL.");
        return NULL;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Binary_Search_Tree_Search: get func pointer NULL.");
        return NULL;
    }
#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_LOCK(T)
#endif

    while (x != NULL && func(k, x->key, other_param)!=MADD_SAME){
        if (func(k, x->key, other_param)==MADD_LESS){
            x = x->left;
        }
        else{
            x = x->right;
        }
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_UNLOCK(T)
#endif

    return x;
}

Binary_Search_Tree_Node *Binary_Search_Tree_Minimum(Binary_Search_Tree *T, Binary_Search_Tree_Node *x)
{
    if (x == NULL){
        return NULL;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_LOCK(T)
#endif
    while (x->left != NULL){
        x = x->left;
    }
#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_UNLOCK(T)
#endif
    return x;
}

Binary_Search_Tree_Node *Binary_Search_Tree_Maximum(Binary_Search_Tree *T, Binary_Search_Tree_Node *x)
{
    if (x == NULL){
        return NULL;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_LOCK(T)
#endif
    while (x->right != NULL){
        x = x->right;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_UNLOCK(T)
#endif
    return x;
}

Binary_Search_Tree_Node *Binary_Search_Tree_Successor(Binary_Search_Tree *T, Binary_Search_Tree_Node *x)
{
    if (x == NULL){
        return NULL;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_LOCK(T)
#endif
    if (x->right != NULL){
#ifdef MADD_ENABLE_MULTITHREAD
        BST_READ_UNLOCK(T)
#endif
        return Binary_Search_Tree_Minimum(T, x->right);
    }
    Binary_Search_Tree_Node *y = x->p;
    while (y!=NULL && x == y->right){
        x = y;
        y = y->p;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_UNLOCK(T)
#endif
    return y;
}

Binary_Search_Tree_Node *Binary_Search_Tree_Predecessor(Binary_Search_Tree *T, Binary_Search_Tree_Node *x)
{
    if (x == NULL){
        return NULL;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_LOCK(T)
#endif
    if (x->left != NULL){
#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_UNLOCK(T)
#endif
        return Binary_Search_Tree_Maximum(T, x->left);
    }
    Binary_Search_Tree_Node *y = x->p;
    while (y!=NULL && x == y->left){
        x = y;
        y = y->p;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_READ_UNLOCK(T)
#endif

    return y;
}

void Binary_Search_Tree_Insert(Binary_Search_Tree *T, Binary_Search_Tree_Node *z, char func(void *key1,void *key2,void *other_param), void *other_param)
{
#ifdef MADD_ENABLE_MULTITHREAD
    BST_WRITE_LOCK(T)
#endif

    Binary_Search_Tree_Node *y=NULL, *x=T->root;
    while (x!=NULL){
        y = x;
        if (func(z->key, x->key, other_param)==MADD_LESS){
            x = x->left;
        }
        else{
            x = x->right;
        }
    }
    z->p = y;
    if (y == NULL){
        T->root = z; /* tree T was empty */
    }
    else if (func(z->key, y->key, other_param)==MADD_LESS){
        y->left = z;
    }
    else{
        y->right = z;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_WRITE_UNLOCK(T)
#endif
}

void Binary_Search_Tree_Transplant(Binary_Search_Tree *T, Binary_Search_Tree_Node *u, Binary_Search_Tree_Node *v)
{
    if (u->p == NULL){
        T->root = v;
    }
    else if (u == u->p->left){
        u->p->left = v;
    }
    else{
        u->p->right = v;
    }
    if (v != NULL){
        v->p = u->p;
    }
}

void Binary_Search_Tree_Delete(Binary_Search_Tree *T, Binary_Search_Tree_Node *z)
{
#ifdef MADD_ENABLE_MULTITHREAD
    BST_WRITE_LOCK(T)
#endif

    Binary_Search_Tree_Node *y;
    if (z->left == NULL){
        Binary_Search_Tree_Transplant(T, z, z->right);
    }
    else if (z->right == NULL){
        Binary_Search_Tree_Transplant(T, z, z->left);
    }
    else{
        y = Binary_Search_Tree_Minimum(T, z->right);
        if (y->p != z){
            Binary_Search_Tree_Transplant(T, y, y->right);
            y->right = z->right;
            y->right->p = y;
        }
        Binary_Search_Tree_Transplant(T, z, y);
        y->left = z->left;
        y->left->p = y;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    BST_WRITE_UNLOCK(T)
#endif
}