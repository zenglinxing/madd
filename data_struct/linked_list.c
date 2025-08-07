/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/linked_list.c
*/
#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<stdbool.h>
#include"data_struct.h"
#include"../basic/basic.h"
#include"../sort/sort.h"

#define LL_WRITE_LOCK(a) \
{ \
    if ((a)->flag_multithread){ \
        RWLock_Write_Lock(&(a)->rwlock); \
    } \
} \

#define LL_WRITE_UNLOCK(a) \
{ \
    if ((a)->flag_multithread){ \
        RWLock_Write_Unlock(&(a)->rwlock); \
    } \
} \

#define LL_READ_LOCK(a) \
{ \
    if ((a)->flag_multithread){ \
        RWLock_Read_Lock(&(a)->rwlock); \
    } \
} \

#define LL_READ_UNLOCK(a) \
{ \
    if ((a)->flag_multithread){ \
        RWLock_Read_Unlock(&(a)->rwlock); \
    } \
} \

#define LL_WRITE_LOCK_NODES(a, b) do { \
    if ((uintptr_t)(a) < (uintptr_t)(b)) { \
        LL_WRITE_LOCK(a); \
        LL_WRITE_LOCK(b); \
    } else if ((a) != (b)) { \
        LL_WRITE_LOCK(b); \
        LL_WRITE_LOCK(a); \
    } else { \
        LL_WRITE_LOCK(a); \
    } \
} while(0); \

#define LL_WRITE_UNLOCK_NODES(a, b) do { \
    if ((uintptr_t)(a) < (uintptr_t)(b)) { \
        LL_WRITE_UNLOCK(a); \
        LL_WRITE_UNLOCK(b); \
    } else if ((a) != (b)) { \
        LL_WRITE_UNLOCK(b); \
        LL_WRITE_UNLOCK(a); \
    } else { \
        LL_WRITE_UNLOCK(a); \
    } \
} while(0); \

bool Linked_List_Init(Linked_List_Node *node, size_t usize)
{
    if (node == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Init: the given node pointer is NULL.");
        return false;
    }
    node->flag_multithread = false;
    node->usize = usize;
    node->prev = node->next = NULL;
    if (usize){
        node->buf = malloc(usize);
        if (node->buf == NULL){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cannot allocate mem %llu.", __func__, usize);
            Madd_Error_Add(MADD_ERROR, error_info);
            return false;
        }
    }else{
        node->buf = NULL;
    }
    return true;
}

bool Linked_List_Enable_Multithread(Linked_List_Node *node)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (node->flag_multithread){
        Madd_Error_Add(MADD_WARNING, L"Linked_List_Enable_Multithread: linked list node already has read-write lock initialized.");
        return false;
    }
    RWLock_Init(&node->rwlock);
    node->flag_multithread = true;
    return true;
#else
    node->flag_multithread = false;
    Madd_Error_Add(MADD_WARNING, L"Linked_List_Enable_Multithread: Madd lib multithread wasn't enabled during compiling. Tried to enable Madd's multithread and re-compile Madd.");
    return false;
#endif
}

void Linked_List_Destroy(Linked_List_Node *node)
{
    if (node == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Destroy: node pointer is NULL.");
        return;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    LL_WRITE_LOCK(node)
#endif

    node->prev = node->next = NULL;
    if (node->buf != NULL && node->usize != 0){
        free(node->buf);
        node->buf = NULL;
    }
    node->usize = 0;
#ifdef MADD_ENABLE_MULTITHREAD
    if (node->flag_multithread){
        RWLock_Write_Unlock(&node->rwlock);
        RWLock_Destroy(&node->rwlock);
    }
#endif
    node->flag_multithread = false;
}

bool Linked_List_Link(Linked_List_Node *prev, Linked_List_Node *next, bool flag_bidirection)
{
    if (prev == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Link: prev node pointer is NULL.");
        return false;
    }
    if (next == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Link: next node pointer is NULL.");
        return false;
    }

    if (prev == next){
#ifdef MADD_ENABLE_MULTITHREAD
        LL_WRITE_LOCK(prev)
#endif
        prev->next = next;
        if (flag_bidirection){
            next->prev = prev;
        }
#ifdef MADD_ENABLE_MULTITHREAD
        LL_WRITE_UNLOCK(prev)
#endif
        return true;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    LL_WRITE_LOCK_NODES(prev, next)
#endif

    prev->next = next;
    if (flag_bidirection){
        next->prev = prev;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    LL_WRITE_UNLOCK_NODES(prev, next)
#endif
    return true;
}

static bool Deleta_ptr_compare(void *a, void *b, void *other_param)
{
    return a < b;
}

/* sort to avoid dead lock */
static void lock_nodes(Linked_List_Node **nodes, int count)
{
    Sort_Quicksort(count, sizeof(Linked_List_Node*), nodes, Deleta_ptr_compare, NULL);
    for (int i = 0; i < count; i++) {
        //printf("i=%d|%p", i, nodes[i]);
        if (nodes[i] != NULL) {
            LL_WRITE_LOCK(nodes[i]);
        }
        //printf("d\t");
    }
}

static void unlock_nodes(Linked_List_Node **nodes, int count)
{
    for (int i = count - 1; i >= 0; i--) {
        if (nodes[i] != NULL) {
            LL_WRITE_UNLOCK(nodes[i]);
        }
    }
}

bool Linked_List_Delete(Linked_List_Node *node)
{
    if (node == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Delete: node pointer is NULL.");
        return false;
    }

    Linked_List_Node *prev = node->prev, *next = node->next;
#ifdef MADD_ENABLE_MULTITHREAD
    /* avoid node self ptr */
    int n_node = 1;
    Linked_List_Node *nodes[3] = {node};
    if (prev != node){
        nodes[n_node] = prev;
        n_node ++;
    }
    if (next != node && next != prev){
        nodes[n_node] = next;
        n_node ++;
    }
    lock_nodes(nodes, n_node);
    if (node->prev != prev || node->next != next) {
        Linked_List_Node *locked_node = (node->prev != prev) ? prev : next;
        unlock_nodes(nodes, n_node);
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to delete node, ptr %p. The adjacent node (pointer %p) is locked by other thread.", __func__, node, locked_node);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
#endif

    if (prev != NULL){
        prev->next = (next == node) ? NULL : next;
    }
    if (next != NULL){
        next->prev = (prev == node) ? NULL : prev;
    }
    node->next = node->prev = NULL;

#ifdef MADD_ENABLE_MULTITHREAD
    unlock_nodes(nodes, n_node);
#endif
    return true;
}

bool Linked_List_Insert_After(Linked_List_Node *prev, Linked_List_Node *next, bool flag_bidirection)
{
    if (prev == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Link: prev node pointer is NULL.");
        return false;
    }
    if (next == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Link: next node pointer is NULL.");
        return false;
    }

    if (prev == next){
#ifdef MADD_ENABLE_MULTITHREAD
            LL_WRITE_LOCK(next)
#endif
        Linked_List_Node *old_next = prev->next;
        prev->next = next;
        if (flag_bidirection) next->prev = prev;
#ifdef MADD_ENABLE_MULTITHREAD
            LL_WRITE_UNLOCK(next)
#endif
        return true;
    }

    Linked_List_Node *old_next = prev->next;

    if (old_next == next){
        Madd_Error_Add(MADD_WARNING, L"Linked_List_Link: the next node had already linked.");
        if (flag_bidirection){
#ifdef MADD_ENABLE_MULTITHREAD
        LL_WRITE_LOCK(next)
#endif
            next->prev = prev;
#ifdef MADD_ENABLE_MULTITHREAD
        LL_WRITE_UNLOCK(next)
#endif
        }
        return true;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    Linked_List_Node *nodes[4] = {prev, next, old_next, NULL};
    int node_count = (old_next && old_next != next && prev != old_next) ? 3 : 2;
    //printf("node_count:%d\t", node_count);
    lock_nodes(nodes, node_count);
    if (prev->next != old_next) {
        unlock_nodes(nodes, node_count);
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Insert_After: list structure changed during locking");
        return false;
    }
#endif

    prev->next = next;
    next->prev = flag_bidirection ? prev : NULL;
    next->next = old_next;
    
    if (old_next != NULL) {
        if (flag_bidirection){
            old_next->prev = next;
        }
    }
    
#ifdef MADD_ENABLE_MULTITHREAD
    if (node_count == 3){
        unlock_nodes(nodes, node_count);
    }else{
        LL_WRITE_UNLOCK_NODES(prev, next)
    }
#endif
    return true;
}