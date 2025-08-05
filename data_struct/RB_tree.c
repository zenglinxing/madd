/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./data_struct/RB_tree.c

Red-Black Tree
I did this via refering to the book Introduction to Algorithm by Thomas H. Cormen.
*/
#include<stdint.h>
#include<stdlib.h>

#include"RB_tree.h"
#include"../basic/basic.h"

inline char RB_Tree_Internal_Compare(RB_Tree_Node *node1, RB_Tree_Node *node2,
                                     char func(void *key1, void *key2, void *other_param), void *other_param,
                                     int flag_allow_same_key)
{
    char res = func(node1->key, node2->key, other_param);
    if (res != MADD_SAME) return res;
    if (flag_allow_same_key == RB_TREE_ENABLE_SAME_KEY){
        if (node1 < node2) return MADD_LESS;
        else if (node1 > node2) return MADD_GREATER;
    }
    /* if RB_TREE_DISABLE_SAME_KEY or pointers node1==node2 */
    /*
    code blocks for error info
    */
}

void RB_Tree_Create(RB_Tree *T)
{
    T->nil.color = 'b';
    T->nil.key = NULL;
    T->nil.left = T->nil.right = T->nil.p = &T->nil;
    T->root = &T->nil;
}

RB_Tree_Node *RB_Tree_Search(RB_Tree *T, RB_Tree_Node *x, void *k,
                             char func(void *key1,void *key2,void *other_param), void *other_param,
                             int flag_allow_same_key)
{
    if (x == &T->nil) return NULL;
    char cmp;
    while (x != &T->nil){
        cmp = func(k, x->key, other_param);
        if (cmp == MADD_SAME) return x;
        x = (cmp == MADD_LESS) ? x->left : x->right;
    }
    return NULL;
}

RB_Tree_Node *RB_Tree_Minimum(RB_Tree *T,RB_Tree_Node *x)
{
    if (x == &T->nil) return &T->nil;
    while (x->left != &T->nil){
        x = x->left;
    }
    return x;
}

RB_Tree_Node *RB_Tree_Maximum(RB_Tree *T,RB_Tree_Node *x)
{
    if (x == &T->nil) return &T->nil;
    while (x->right != &T->nil){
        x = x->right;
    }
    return x;
}

RB_Tree_Node *RB_Tree_Successor(RB_Tree *T,RB_Tree_Node *x)
{
    if (x == &T->nil) return &T->nil;
    if (x->right != &T->nil) return RB_Tree_Minimum(T, x->right);
    RB_Tree_Node *y = x->p;
    while (y!=&T->nil && x == y->right){
        x = y;
        y = y->p;
    }
    return y;
}

RB_Tree_Node *RB_Tree_Predecessor(RB_Tree *T,RB_Tree_Node *x)
{
    if (x == &T->nil) return &T->nil;
    if (x->left != &T->nil) return RB_Tree_Maximum(T, x->left);
    RB_Tree_Node *y = x->p;
    while (y!=&T->nil && x == y->left){
        x = y;
        y = y->p;
    }
    return y;
}

int RB_Tree_Left_Rotate(RB_Tree *T, RB_Tree_Node *x)
{
    RB_Tree_Node *y = x->right; /* set y */
    if (y == &T->nil) return RB_TREE_NULL;
    x->right = y->left; /* turn y's left subtree into x's right subtree */
    if (y->left != &T->nil){
        y->left->p = x;
    }
    y->p = x->p; /* link x's parent to y */
    if (x->p == &T->nil){
        T->root = y;
    }
    else if (x == x->p->left){
        x->p->left = y;
    }
    else{
        x->p->right = y;
    }
    y->left = x; /* put x on y's left */
    x->p = y;
    return RB_TREE_SUCCESS;
}

int RB_Tree_Right_Rotate(RB_Tree *T, RB_Tree_Node *x)
{
    RB_Tree_Node *y = x->left; /* set y */
    if (y == &T->nil) return RB_TREE_NULL;
    x->left = y->right; /* turn y's right subtree into x's left subtree */
    if (y->right != &T->nil){
        y->right->p = x;
    }
    y->p = x->p; /* link x's parent to y */
    if (x->p == &T->nil){
        T->root = y;
    }
    else if (x == x->p->right){
        x->p->right = y;
    }
    else{
        x->p->left = y;
    }
    y->right = x; /* put x on y's right */
    x->p = y;
    return RB_TREE_SUCCESS;
}

void RB_Tree_Insert_Fixup(RB_Tree *T, RB_Tree_Node *z)
{
    RB_Tree_Node *y;
    while /*(z->p->color == 'r')*/ ( z->p->color=='r' ){
        if (z->p == z->p->p->left){
            y = z->p->p->right; /* right uncle */
            if (y->color == 'r'){
                z->p->color = y->color = 'b';
                z->p->p->color = 'r';
                z = z->p->p;
            }
            else{
                if (z == z->p->right){
                    z = z->p;
                    RB_Tree_Left_Rotate(T, z);
                }
                z->p->color = 'b';
                z->p->p->color = 'r';
                RB_Tree_Right_Rotate(T, z->p->p);
            }
        }
        else{
            y = z->p->p->left; /* left uncle */
            if (y->color == 'r'){
                z->p->color = y->color = 'b';
                z->p->p->color = 'r';
                z = z->p->p;
            }
            else{
                if (z == z->p->left){
                    z = z->p;
                    RB_Tree_Right_Rotate(T, z);
                }
                z->p->color = 'b';
                z->p->p->color = 'r';
                RB_Tree_Left_Rotate(T, z->p->p);
            }
        }
    }
    T->root->color = 'b';
}

void RB_Tree_Insert(RB_Tree *T, RB_Tree_Node *z,
                    char func(void *key1,void *key2,void *other_param), void *other_param,
                    int flag_allow_same_key)
{
    RB_Tree_Node *x,*y;
    y = &T->nil;
    x = T->root;
    while (x != &T->nil){
        y = x;
        if (RB_Tree_Internal_Compare(z, x, func, other_param, flag_allow_same_key) == MADD_LESS){
            x = x->left;
        }
        else{
            x = x->right;
        }
    }
    z->p = y;
    if (y == &T->nil){
        T->root = z;
    }
    else if (RB_Tree_Internal_Compare(z, y, func, other_param, flag_allow_same_key) == MADD_LESS){
        y->left = z;
    }
    else{
        y->right = z;
    }
    z->left = z->right = &T->nil;
    z->color = 'r';
    RB_Tree_Insert_Fixup(T, z);
}

void RB_Tree_Transplant(RB_Tree *T, RB_Tree_Node *u, RB_Tree_Node *v)
{
    if (u->p == &T->nil){
        T->root = v;
    }
    else if (u == u->p->left){
        u->p->left = v;
    }
    else{
        u->p->right = v;
    }
    if (v != &T->nil) {  /* avoid to change nil */
        v->p = u->p;
    }
}

void RB_Tree_Delete_Fixup(RB_Tree *T, RB_Tree_Node *x)
{
    RB_Tree_Node *w;
    while (x != T->root && x->color == 'b'){
        if (x == x->p->left){
            w = x->p->right;
            if (w->color == 'r'){
                w->color = 'b';
                x->p->color = 'r';
                RB_Tree_Left_Rotate(T, x->p);
                w = x->p->right;
            }
            if (w->left->color == 'b' && w->right->color == 'b'){
                w->color = 'r';
                x = x->p;
            }
            else{
                if (w->right->color == 'b'){
                    w->left->color = 'b';
                    w->color = 'r';
                    RB_Tree_Right_Rotate(T, w);
                    w = x->p->right;
                }
                w->color = x->p->color;
                x->p->color = w->right->color = 'b';
                RB_Tree_Left_Rotate(T, x->p);
                x = T->root;
                x->p = &T->nil;
            }
        }
        else{
            w = x->p->left;
            if (w->color == 'r'){
                w->color = 'b';
                x->p->color = 'r';
                RB_Tree_Right_Rotate(T, x->p);
                w = x->p->left;
            }
            if (w->left->color == 'b' && w->right->color == 'b'){
                w->color = 'r';
                x = x->p;
            }
            else{
                if (w->left->color == 'b'){
                    w->right->color = 'b';
                    w->color = 'r';
                    RB_Tree_Left_Rotate(T, w);
                    w = x->p->left;
                }
                w->color = x->p->color;
                x->p->color = w->left->color = 'b';
                RB_Tree_Right_Rotate(T, x->p);
                x = T->root;
                x->p = &T->nil;
            }
        }
    }
    x->color = 'b';
}

void RB_Tree_Delete(RB_Tree *T, RB_Tree_Node *z)
{
    RB_Tree_Node *x, *y = z;
    char y_original_color = y->color;
    if (z->left == &T->nil){
        x = z->right;
        RB_Tree_Transplant(T, z, z->right);
    }
    else if (z->right == &T->nil){
        x = z->left;
        RB_Tree_Transplant(T, z, z->left);
    }
    else{
        y = RB_Tree_Minimum(T, z->right);
        y_original_color = y->color;
        x = y->right;
        if (y->p == z){
            x->p = y;
        }
        else{
            RB_Tree_Transplant(T, y, y->right);
            y->right = z->right;
            y->right->p = y;
        }
        RB_Tree_Transplant(T, z, y);
        y->left = z->left;
        y->left->p = y;
        y->color = z->color;
    }
    if (y_original_color == 'b'){
        RB_Tree_Delete_Fixup(T, x);
    }
}