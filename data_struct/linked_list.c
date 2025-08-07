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

static bool try_write_lock_node(Linked_List_Node *node, uint64_t n_try)
{
    if (!node->flag_multithread) return true;
    uint64_t i_try;
    bool res_trylock = false;
    if (n_try){
        for (i_try=0; i_try<n_try; i_try++){
            res_trylock = RWLock_Try_Write_Lock(&node->rwlock);
            if (res_trylock) return true;
        }
        return false;
    }else{
        while (!res_trylock){
            res_trylock = RWLock_Try_Write_Lock(&node->rwlock);
        }
        return true;
    }
}

bool Linked_List_Init(Linked_List_Node *node, size_t usize)
{
    if (node == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Init: the given node pointer is NULL.");
        return false;
    }
    node->flag_multithread = false;
    node->usize = usize;
    node->prev = node->next = NULL;
    node->max_trylock = 30;
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

bool Linked_List_Destroy(Linked_List_Node *node)
{
    if (node == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Destroy: node pointer is NULL.");
        return false;
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
    return true;
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

#ifdef MADD_ENABLE_MULTITHREAD
        LL_WRITE_LOCK(prev)
#endif

    if (prev == next){
        prev->next = next;
        if (flag_bidirection){
            next->prev = prev;
        }
#ifdef MADD_ENABLE_MULTITHREAD
        LL_WRITE_UNLOCK(prev)
#endif
        // prev->next = node->next, so no need to consider next node
        return true;
    }

    // the case "prev = next" had already been dealed above, so the following next != prev
    Linked_List_Node *prev_next = prev->next;
    bool res_change_prev_next = false;
    if (prev_next != NULL && flag_bidirection){
        if (prev_next->prev != prev){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"Linked_List_Link: the prev.next node (ptr %p) is not bidirectional linked.", prev_next);
            Madd_Error_Add(MADD_ERROR, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
            LL_WRITE_UNLOCK(prev)
#endif
            return false;
        }else{
            res_change_prev_next = true;
        }
    }
#ifdef MADD_ENABLE_MULTITHREAD
    if (flag_bidirection){
        bool res_locked_next = try_write_lock_node(next, prev->max_trylock);
        if (!res_locked_next){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"Linked_List_Link: unable to lock the next node (ptr %p was write-locked).", next);
            Madd_Error_Add(MADD_ERROR, error_info);
            LL_WRITE_UNLOCK(prev);
            return false;
        }
    }
    if (res_change_prev_next){
        bool res_locked_prev_next = try_write_lock_node(prev_next, prev->max_trylock);
        if (!res_locked_prev_next){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"Linked_List_Link: unable to lock the prev.next node (ptr %p was write-locked).", prev_next);
            Madd_Error_Add(MADD_ERROR, error_info);
            LL_WRITE_UNLOCK(next)
            LL_WRITE_UNLOCK(prev)
            return false;
        }
    }
#endif

    prev->next = next;
    if (flag_bidirection){
        next->prev = prev;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    LL_WRITE_UNLOCK(next)
    LL_WRITE_UNLOCK(prev)
#endif

    return true;
}

bool Linked_List_Delete(Linked_List_Node *node)
{
    if (node == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Delete: node pointer is NULL.");
        return false;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    LL_WRITE_LOCK(node)
#endif
    Linked_List_Node *prev = node->prev, *next = node->next;

#ifdef MADD_ENABLE_MULTITHREAD
    // try to lock prev & next
    bool res_trylock;
    if (prev != NULL && prev != node){
        //printf("locking prev:%p\t", prev);
        //fflush(stdout);
        res_trylock = try_write_lock_node(prev, node->max_trylock);
        if (!res_trylock){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"Linked_List_Delete: unable to lock the prev node (ptr %p was write-locked).", prev);
            Madd_Error_Add(MADD_ERROR, error_info);
            LL_WRITE_UNLOCK(node)
            return false;
        }
    }
    if (next != NULL && next != node && next != prev){
        //printf("locking next:%p\t", next);
        //fflush(stdout);
        res_trylock = try_write_lock_node(next, node->max_trylock);
        if (!res_trylock){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"Linked_List_Delete: unable to lock the next node (ptr %p was write-locked).", next);
            Madd_Error_Add(MADD_ERROR, error_info);
            LL_WRITE_UNLOCK(node)
            if (prev != NULL){
                LL_WRITE_UNLOCK(prev)
            }
            return false;
        }
    }
#endif
    if (prev != NULL){
        prev->next = (next == node) ? NULL : next;
#ifdef MADD_ENABLE_MULTITHREAD
        if (prev != node){
            LL_WRITE_UNLOCK(prev)
        }
#endif
    }
    if (next != NULL){
        next->prev = (prev == node) ? NULL : prev;
#ifdef MADD_ENABLE_MULTITHREAD
        if (next != node && next != prev){
            LL_WRITE_UNLOCK(next)
        }
#endif
    }
    node->next = node->prev = NULL;

#ifdef MADD_ENABLE_MULTITHREAD
    LL_WRITE_UNLOCK(node)
#endif
    return true;
}

bool Linked_List_Insert_After(Linked_List_Node *prev, Linked_List_Node *node, bool flag_bidirection)
{
    if (prev == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Link: prev node pointer is NULL.");
        return false;
    }
    if (node == NULL){
        Madd_Error_Add(MADD_ERROR, L"Linked_List_Link: inserting node pointer is NULL.");
        return false;
    }

    if (prev == node){
#ifdef MADD_ENABLE_MULTITHREAD
        LL_WRITE_LOCK(prev)
#endif
        prev->next = node;
        if (flag_bidirection){
            node->prev = prev;
        }
#ifdef MADD_ENABLE_MULTITHREAD
        LL_WRITE_UNLOCK(prev)
#endif
        return true;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    printf("locking prev:%p\t", prev);
    fflush(stdout);
    LL_WRITE_LOCK(prev)
    printf("locked\t");
    fflush(stdout);
#endif
    Linked_List_Node *next = prev->next;

#ifdef MADD_ENABLE_MULTITHREAD
    bool res_trylock_node = try_write_lock_node(node, prev->max_trylock), res_trylock_next;
    // case "prev = next" has already been checked above
    if (!res_trylock_node){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Linked_List_Delete: unable to lock the inserting node (ptr %p was write-locked).", node);
        Madd_Error_Add(MADD_ERROR, error_info);
        LL_WRITE_UNLOCK(prev)
        return false;
    }
    if (next != NULL && next != prev && next != node){
        res_trylock_next = try_write_lock_node(next, prev->max_trylock);
        if (!res_trylock_next){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"Linked_List_Delete: unable to lock the next node (ptr %p was write-locked).", next);
            Madd_Error_Add(MADD_ERROR, error_info);
            LL_WRITE_UNLOCK(node)
            LL_WRITE_UNLOCK(prev)
            return false;
        }else if (next != prev->next){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"Linked_List_Delete: the next node (ptr %p was write-locked) was changed externally (maybe by other threads, or modified by yourself before).", next);
            Madd_Error_Add(MADD_ERROR, error_info);
            LL_WRITE_UNLOCK(next)
            LL_WRITE_UNLOCK(node)
            LL_WRITE_UNLOCK(prev)
            return false;
        }
    }
#endif

    printf("insert after:\t%p\t%p\t%p\t", prev, node, next);
    prev->next = node;
    if (flag_bidirection){
        node->prev = prev;
    }
    node->next = next;
    if (next != NULL && flag_bidirection){
        next->prev = node;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    LL_WRITE_UNLOCK(node)
    LL_WRITE_UNLOCK(prev)
    if (next != NULL && next != node && next != prev){
        LL_WRITE_UNLOCK(next)
    }
#endif
    return true;
}