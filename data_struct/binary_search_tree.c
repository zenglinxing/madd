/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./data_struct/binary_search_tree.c
*/

#include<stdint.h>
#include<stdlib.h>
#include"binary_search_tree.h"

Binary_Search_Tree Binary_Search_Tree_Make(void)
{
    Binary_Search_Tree T={.root=NULL};
    return T;
}

/*
func return:
    0   key1 < key2
    1   key1 = key2
    2   key1 > key2
*/
Binary_Search_Tree_Node *Binary_Search_Tree_Search(Binary_Search_Tree_Node *x, void *k, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    while (x != NULL && func(k, x->key, other_param)!=BINARY_SEARCH_TREE_SAME){
        if (func(k, x->key, other_param)==BINARY_SEARCH_TREE_LESS){
            x = x->left;
        }
        else{
            x = x->right;
        }
    }
    return x;
}

Binary_Search_Tree_Node *Binary_Search_Tree_Minimum(Binary_Search_Tree_Node *x)
{
    if (x == NULL) return NULL;
    while (x->left != NULL){
        x = x->left;
    }
    return x;
}

Binary_Search_Tree_Node *Binary_Search_Tree_Maximum(Binary_Search_Tree_Node *x)
{
    if (x == NULL) return NULL;
    while (x->right != NULL){
        x = x->right;
    }
    return x;
}

Binary_Search_Tree_Node *Binary_Search_Tree_Successor(Binary_Search_Tree_Node *x)
{
    if (x == NULL) return NULL;
    if (x->right != NULL) return Binary_Search_Tree_Minimum(x->right);
    Binary_Search_Tree_Node *y = x->p;
    while (y!=NULL && x == y->right){
        x = y;
        y = y->p;
    }
    return y;
}

Binary_Search_Tree_Node *Binary_Search_Tree_Predecessor(Binary_Search_Tree_Node *x)
{
    if (x == NULL) return NULL;
    if (x->left != NULL) return Binary_Search_Tree_Maximum(x->left);
    Binary_Search_Tree_Node *y = x->p;
    while (y!=NULL && x == y->left){
        x = y;
        y = y->p;
    }
    return y;
}

void Binary_Search_Tree_Insert(Binary_Search_Tree *T, Binary_Search_Tree_Node *z, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    Binary_Search_Tree_Node *y=NULL, *x=T->root;
    while (x!=NULL){
        y = x;
        if (func(z->key, x->key, other_param)==BINARY_SEARCH_TREE_LESS){
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
    else if (func(z->key, y->key, other_param)==BINARY_SEARCH_TREE_LESS){
        y->left = z;
    }
    else{
        y->right = z;
    }
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
    Binary_Search_Tree_Node *y;
    if (z->left == NULL){
        Binary_Search_Tree_Transplant(T, z, z->right);
    }
    else if (z->right == NULL){
        Binary_Search_Tree_Transplant(T, z, z->left);
    }
    else{
        y = Binary_Search_Tree_Minimum(z->right);
        if (y->p != z){
            Binary_Search_Tree_Transplant(T, y, y->right);
            y->right = z->right;
            y->right->p = y;
        }
        Binary_Search_Tree_Transplant(T, z, y);
        y->left = z->left;
        y->left->p = y;
    }
}