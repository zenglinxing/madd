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
#include<wchar.h>

#include"RB_tree.h"
#include"../basic/basic.h"
/*#include"../thread_base/thread_base.h"*/

#define RB_READ_LOCK(T) \
    if ((T)->flag_multithread){ \
        RWLock_Read_Lock(&(T)->rwlock); \
    } \

#define RB_READ_UNLOCK(T) \
    if ((T)->flag_multithread){ \
        RWLock_Read_Unlock(&(T)->rwlock); \
    } \

#define RB_WRITE_LOCK(T) \
    if ((T)->flag_multithread){ \
        RWLock_Write_Lock(&(T)->rwlock); \
    } \

#define RB_WRITE_UNLOCK(T) \
    if ((T)->flag_multithread){ \
        RWLock_Write_Unlock(&(T)->rwlock); \
    } \

static inline char RB_Tree_Internal_Compare(RB_Tree_Node *node1, RB_Tree_Node *node2,
                                     char func(void *key1, void *key2, void *other_param), void *other_param,
                                     int flag_allow_same_key,
                                     char *func_name)
{
    char res = func(node1->key, node2->key, other_param);
    if (res == MADD_LESS || res == MADD_GREATER) return res;
    else if (res == MADD_SAME && flag_allow_same_key == RB_TREE_ENABLE_SAME_KEY){
        if (node1 < node2) return MADD_LESS;
        else if (node1 > node2) return MADD_GREATER;
        else{
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: red-black tree doesn't allow same key! Although you had set RB_TREE_ENABLE_SAME_KEY, but even the RB_Tree_Node(s) have same pointer.", func_name);
            Madd_Error_Add(MADD_ERROR, error_info);
            return MADD_SAME;
        }
    }else if (res == MADD_SAME /*&& flag_allow_same_key != RB_TREE_ENABLE_SAME_KEY*/){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: red-black tree doesn't allow same key! You may try to set RB_TREE_ENABLE_SAME_KEY for %s(... int flag_allow_same_key, ...).", func_name);
        Madd_Error_Add(MADD_ERROR, error_info);
        return MADD_SAME;
    }else{
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: unknown return value: %d. func_compare should only return MADD_LESS/MADD_SAME/MADD_GREATER.", func_name, res);
        Madd_Error_Add(MADD_ERROR, error_info);
        return res;
    }
}

bool RB_Tree_Init(RB_Tree *T)
{
    if (T == NULL){
        Madd_Error_Add(MADD_ERROR, L"RB_Tree_Init: RB Tree pointer is NULL.");
        return false;
    }
    T->nil.color = 'b';
    T->nil.key = NULL;
    T->nil.left = T->nil.right = T->nil.p = &T->nil;
    T->root = &T->nil;
    T->flag_multithread = false;
    return true;
}

bool RB_Tree_Enable_Multithread(RB_Tree *T)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (T->flag_multithread){
        Madd_Error_Add(MADD_WARNING, L"RB_Tree_Enable_Multithread: RB tree already has read-write lock initialized.");
        return false;
    }
    RWLock_Init(&T->rwlock);
    T->flag_multithread = true;
    return true;
#else
    T->flag_multithread = false;
    Madd_Error_Add(MADD_WARNING, L"RB_Tree_Enable_Multithread: Madd lib multithread wasn't enabled during compiling. Tried to enable Madd's multithread and re-compile Madd.");
    return false;
#endif
}

RB_Tree_Node *RB_Tree_Search(RB_Tree *T, RB_Tree_Node *x, void *k,
                             char func(void *key1,void *key2,void *other_param), void *other_param,
                             int flag_allow_same_key)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_LOCK(T)
#endif

    if (x == &T->nil){
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
        return NULL;
    }
    char cmp;
    while (x != &T->nil){
        cmp = func(k, x->key, other_param);
        if (cmp == MADD_SAME){
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
            return x;
        }
        x = (cmp == MADD_LESS) ? x->left : x->right;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
    return NULL;
}

static RB_Tree_Node *RB_Tree_Minimum_Internal(RB_Tree *T, RB_Tree_Node *x)
{
    if (x == &T->nil){
        return NULL;
    }
    while (x->left != &T->nil){
        x = x->left;
    }
    return x;
}

RB_Tree_Node *RB_Tree_Minimum(RB_Tree *T, RB_Tree_Node *x)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_LOCK(T)
#endif

    RB_Tree_Node *ret = RB_Tree_Minimum_Internal(T, x);

#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
    return ret;
}

static RB_Tree_Node *RB_Tree_Maximum_Internal(RB_Tree *T,RB_Tree_Node *x)
{
    if (x == &T->nil){
        return NULL;
    }
    while (x->right != &T->nil){
        x = x->right;
    }
    return x;
}

RB_Tree_Node *RB_Tree_Maximum(RB_Tree *T,RB_Tree_Node *x)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_LOCK(T)
#endif

    RB_Tree_Node *ret = RB_Tree_Maximum_Internal(T, x);

#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
    return ret;
}

RB_Tree_Node *RB_Tree_Successor(RB_Tree *T,RB_Tree_Node *x)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_LOCK(T)
#endif

    if (x == &T->nil){
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
        return NULL;
    }
    if (x->right != &T->nil){
        RB_Tree_Node *ret = RB_Tree_Minimum_Internal(T, x->right);
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
        return ret;
    }
    RB_Tree_Node *y = x->p;
    while (y!=&T->nil && x == y->right){
        x = y;
        y = y->p;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
    return y;
}

RB_Tree_Node *RB_Tree_Predecessor(RB_Tree *T,RB_Tree_Node *x)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_LOCK(T)
#endif

    if (x == &T->nil){
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
        return NULL;
    }
    if (x->left != &T->nil){
        RB_Tree_Node *ret = RB_Tree_Maximum_Internal(T, x->left);
#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif
        return ret;
    }
    RB_Tree_Node *y = x->p;
    while (y!=&T->nil && x == y->left){
        x = y;
        y = y->p;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    RB_READ_UNLOCK(T)
#endif

    return y;
}

int RB_Tree_Left_Rotate(RB_Tree *T, RB_Tree_Node *x)
{
    RB_Tree_Node *y = x->right; /* set y */
    if (y == &T->nil){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: meets NULL pointer. %s line %d.", __func__, __FILE__, __LINE__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return RB_TREE_NULL;
    }
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
    if (y == &T->nil){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: meets NULL pointer. %s line %d.", __func__, __FILE__, __LINE__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return RB_TREE_NULL;
    }
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

int RB_Tree_Insert_Fixup(RB_Tree *T, RB_Tree_Node *z)
{
    int res_rotate;
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
                    res_rotate = RB_Tree_Left_Rotate(T, z);
                    if (res_rotate == RB_TREE_NULL){
                        wchar_t error_info[MADD_ERROR_INFO_LEN];
                        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Left_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                        Madd_Error_Add(MADD_ERROR, error_info);
                        return res_rotate;
                    }
                }
                z->p->color = 'b';
                z->p->p->color = 'r';
                res_rotate = RB_Tree_Right_Rotate(T, z->p->p);
                if (res_rotate == RB_TREE_NULL){
                    wchar_t error_info[MADD_ERROR_INFO_LEN];
                    swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Right_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                    Madd_Error_Add(MADD_ERROR, error_info);
                    return res_rotate;
                }
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
                    res_rotate = RB_Tree_Right_Rotate(T, z);
                    if (res_rotate == RB_TREE_NULL){
                        wchar_t error_info[MADD_ERROR_INFO_LEN];
                        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Right_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                        Madd_Error_Add(MADD_ERROR, error_info);
                        return res_rotate;
                    }
                }
                z->p->color = 'b';
                z->p->p->color = 'r';
                res_rotate = RB_Tree_Left_Rotate(T, z->p->p);
                if (res_rotate == RB_TREE_NULL){
                    wchar_t error_info[MADD_ERROR_INFO_LEN];
                    swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Left_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                    Madd_Error_Add(MADD_ERROR, error_info);
                    return res_rotate;
                }
            }
        }
    }
    T->root->color = 'b';
    return RB_TREE_SUCCESS;
}

void RB_Tree_Insert(RB_Tree *T, RB_Tree_Node *z,
                    char func(void *key1,void *key2,void *other_param), void *other_param,
                    int flag_allow_same_key)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RB_WRITE_LOCK(T)
#endif

    RB_Tree_Node *x,*y;
    int ret_compare;
    y = &T->nil;
    x = T->root;
    while (x != &T->nil){
        y = x;
        ret_compare = RB_Tree_Internal_Compare(z, x, func, other_param, flag_allow_same_key, "RB_Tree_Insert");
        if (ret_compare == MADD_LESS){
            x = x->left;
        }else if (ret_compare == MADD_GREATER || ret_compare == MADD_SAME){
            x = x->right;
        }else{
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: got unknown return %d from compare func. The given RB Tree Node pointers are %p and %p. This info is from Madd source %s line %d.", __func__, ret_compare, z, x, __FILE__, __LINE__);
            Madd_Error_Add(MADD_ERROR, error_info);
            break;
        }
    }
    z->p = y;
    if (y == &T->nil){
        T->root = z;
    }
    else{
        ret_compare = RB_Tree_Internal_Compare(z, y, func, other_param, flag_allow_same_key, "RB_Tree_Insert");
        if (ret_compare == MADD_LESS){
            y->left = z;
        }
        else if (ret_compare == MADD_GREATER || ret_compare == MADD_SAME){
            y->right = z;
        }else{
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: got unknown return %d from compare func. The given RB Tree Node pointers are %p and %p. This info is from Madd source %s line %d.", __func__, ret_compare, z, x, __FILE__, __LINE__);
            Madd_Error_Add(MADD_ERROR, error_info);
        }
    }
    z->left = z->right = &T->nil;
    z->color = 'r';
    int res_fixup = RB_Tree_Insert_Fixup(T, z);
    if (res_fixup == RB_TREE_NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Insert_Fixup. This info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
        Madd_Error_Add(MADD_ERROR, error_info);
    }

#ifdef MADD_ENABLE_MULTITHREAD
    RB_WRITE_UNLOCK(T)
#endif
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

int RB_Tree_Delete_Fixup(RB_Tree *T, RB_Tree_Node *x)
{
    int res_rotate;
    RB_Tree_Node *w;
    while (x != T->root && x->color == 'b'){
        if (x == x->p->left){
            w = x->p->right;
            if (w->color == 'r'){
                w->color = 'b';
                x->p->color = 'r';
                res_rotate = RB_Tree_Left_Rotate(T, x->p);
                if (res_rotate == RB_TREE_NULL){
                    wchar_t error_info[MADD_ERROR_INFO_LEN];
                    swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Left_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                    Madd_Error_Add(MADD_ERROR, error_info);
                    return res_rotate;
                }
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
                    res_rotate = RB_Tree_Right_Rotate(T, w);
                    if (res_rotate == RB_TREE_NULL){
                        wchar_t error_info[MADD_ERROR_INFO_LEN];
                        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Right_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                        Madd_Error_Add(MADD_ERROR, error_info);
                        return res_rotate;
                    }
                    w = x->p->right;
                }
                w->color = x->p->color;
                x->p->color = w->right->color = 'b';
                res_rotate = RB_Tree_Left_Rotate(T, x->p);
                if (res_rotate == RB_TREE_NULL){
                    wchar_t error_info[MADD_ERROR_INFO_LEN];
                    swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Left_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                    Madd_Error_Add(MADD_ERROR, error_info);
                    return res_rotate;
                }
                x = T->root;
                /*x->p = &T->nil;*/
            }
        }
        else{
            w = x->p->left;
            if (w->color == 'r'){
                w->color = 'b';
                x->p->color = 'r';
                res_rotate = RB_Tree_Right_Rotate(T, x->p);
                if (res_rotate == RB_TREE_NULL){
                    wchar_t error_info[MADD_ERROR_INFO_LEN];
                    swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Right_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                    Madd_Error_Add(MADD_ERROR, error_info);
                    return res_rotate;
                }
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
                    res_rotate = RB_Tree_Left_Rotate(T, w);
                    if (res_rotate == RB_TREE_NULL){
                        wchar_t error_info[MADD_ERROR_INFO_LEN];
                        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Left_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                        Madd_Error_Add(MADD_ERROR, error_info);
                        return res_rotate;
                    }
                    w = x->p->left;
                }
                w->color = x->p->color;
                x->p->color = w->left->color = 'b';
                res_rotate = RB_Tree_Right_Rotate(T, x->p);
                if (res_rotate == RB_TREE_NULL){
                    wchar_t error_info[MADD_ERROR_INFO_LEN];
                    swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Right_Rotate. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
                    Madd_Error_Add(MADD_ERROR, error_info);
                    return res_rotate;
                }
                x = T->root;
                x->p = &T->nil;
            }
        }
    }
    x->color = 'b';
    return RB_TREE_SUCCESS;
}

void RB_Tree_Delete(RB_Tree *T, RB_Tree_Node *z)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RB_WRITE_LOCK(T)
#endif

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
        y = RB_Tree_Minimum_Internal(T, z->right);
        y_original_color = y->color;
        x = y->right;
        if (y->p == z){
            if (x != &T->nil) {
                x->p = y;
            }
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
        int res_fixup = RB_Tree_Delete_Fixup(T, x);
        if (res_fixup == RB_TREE_NULL){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"%s: see info from RB_Tree_Delete_Fixup. This error info is from Madd Source %s line %d.", __func__, __FILE__, __LINE__);
            Madd_Error_Add(MADD_ERROR, error_info);
        }
    }

#ifdef MADD_ENABLE_MULTITHREAD
    RB_WRITE_UNLOCK(T)
#endif
}