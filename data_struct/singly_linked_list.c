/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/singly_linked_list.c
*/
#include<stdlib.h>
#include<stdint.h>
#include"data_struct.h"
#include"../basic/basic.h"
#include"../thread_base/thread_base.h"

static inline void read_lock(Singly_Linked_List *list)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (list->flag_multithread){
        RWLock_Read_Lock(&list->rwlock);
    }
#endif
}

static inline void read_unlock(Singly_Linked_List *list)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (list->flag_multithread){
        RWLock_Read_Unlock(&list->rwlock);
    }
#endif
}

static inline void write_lock(Singly_Linked_List *list)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (list->flag_multithread){
        RWLock_Write_Lock(&list->rwlock);
    }
#endif
}

static inline void write_unlock(Singly_Linked_List *list)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (list->flag_multithread){
        RWLock_Write_Unlock(&list->rwlock);
    }
#endif
}

void Singly_Linked_List_Init(Singly_Linked_List *list)
{
    list->flag_multithread = false;
    list->head = list->tail = NULL;
}

bool Singly_Linked_List_Enable_Multithread(Singly_Linked_List *list)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (list->flag_multithread){
        Madd_Error_Add(MADD_ERROR, L"Singly_Linked_List_Enable_Multithread: singly linked list already has read-write lock initialized.");
        return false;
    }
    RWLock_Init(&list->rwlock);
    list->flag_multithread = true;
    return true;
#else
    list->flag_multithread = false;
    Madd_Error_Add(MADD_WARNING, L"Singly_Linked_List_Enable_Multithread: Madd lib multithread wasn't enabled during compiling. Tried to enable Madd's multithread and re-compile Madd.");
    return false;
#endif
}

bool Singly_Linked_List_Empty(Singly_Linked_List *list)
{
    read_lock(list);
    bool res = list->head == NULL;
    read_unlock(list);
    return res;
}

bool Singly_Linked_List_Insert_Tail(Singly_Linked_List *list, Singly_Linked_List_Node *node)
{
    if (list == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Insert_Tail: list is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (node == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Insert_Tail: node is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    write_lock(list);
    Singly_Linked_List_Node *tail_next = (list->tail == NULL) ? NULL : list->tail->next;
    if (list->head == NULL){
        list->head = list->tail = node;
        node->next = NULL;
    }else{
        list->tail->next = node;
        list->tail = node;
        node->next = tail_next;
    }
    write_unlock(list);
    return true;
}

bool Singly_Linked_List_Insert_After(Singly_Linked_List *list, Singly_Linked_List_Node *prev, Singly_Linked_List_Node *node)
{
    if (list == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Insert_After: list is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (prev == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Insert_After: prev is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (node == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Insert_After: node is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    write_lock(list);
    Singly_Linked_List_Node *tail=list->tail, *next=prev->next;
    prev->next = node;
    node->next = next;
    if (tail == prev){
        list->tail = node;
    }
    write_unlock(list);
    return true;
}

bool Singly_Linked_List_Delete(Singly_Linked_List *list, Singly_Linked_List_Node *node)
{
    if (list == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete: list is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (node == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete: node is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    write_lock(list);
    /* find the previous node */
    Singly_Linked_List_Node *prev = NULL, *cur = list->head, *head = list->head;
    if (list->head == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete: singly linked list at %p is empty.", list);
        Madd_Error_Add(MADD_ERROR, error_info);
        write_unlock(list);
        return false;
    }
    if (list->head == node){
        list->head = node->next;
        if (list->tail == node){
            list->tail = NULL;
        }
        node->next = NULL;
        write_unlock(list);
        return true;
    }
    do {
        prev = cur;
        cur = cur->next;
    } while (cur != node && cur != list->head && cur != NULL);
    if (cur != node){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete: node %p is not in singly linked list %p.", node, list);
        Madd_Error_Add(MADD_ERROR, error_info);
        write_unlock(list);
        return false;
    }
    if (list->tail == node){
        list->tail = prev;
    }
    prev->next = node->next;
    node->next = NULL;

    write_unlock(list);
    return true;
}

bool Singly_Linked_List_Delete_After(Singly_Linked_List *list, Singly_Linked_List_Node *prev)
{
    if (list == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete_After: list is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (prev == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete_After: prev is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    write_lock(list);
    Singly_Linked_List_Node *node = prev->next, *tail=list->tail, *tail_next;
    if (node == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete_After: no node after prev.");
        Madd_Error_Add(MADD_ERROR, error_info);
        write_unlock(list);
        return false;
    }
    if (list->head == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete_After: no node in singly linked list.");
        Madd_Error_Add(MADD_ERROR, error_info);
        write_unlock(list);
        return false;
    }
    if (tail == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Delete_After: the singly linked list %p has node(s), but the tail is NULL.", list);
        Madd_Error_Add(MADD_ERROR, error_info);
        write_unlock(list);
        return false;
    }
    tail_next = tail->next;
    prev->next = node->next;
    if (list->head == node){
        list->head = node->next;
    }
    if (list->tail == node){
        list->tail = prev;
    }
    if (list->tail && list->tail->next == node){
        list->tail->next = node->next;
    }
    node->next = NULL;
    write_unlock(list);
    return true;
}

bool Singly_Linked_List_Find_Loop(Singly_Linked_List_Node *head,
                                  Singly_Linked_List_Node **loop_start,
                                  Singly_Linked_List_Node **tail)
{
    if (loop_start != NULL){
        *loop_start = NULL;
    }
    if (tail != NULL){
        *tail = NULL;
    }
    if (head == NULL){
        return false;
    }
    Singly_Linked_List_Node *slow=head, *fast=head;
    bool flag_loop = false;
    while (fast && fast->next){
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast){
            flag_loop = true;
            break;
        }
    }

    if (flag_loop){
        Singly_Linked_List_Node *p1=head, *p2=fast;
        while (p1 != p2){
            p1 = p1->next;
            p2 = p2->next;
        }
        if (loop_start != NULL){
            *loop_start = slow;
        }
        if (tail != NULL){
            if (p1 == head){
                Singly_Linked_List_Node *p = p1;
                while (p->next != p1){
                    p = p->next;
                }
                *tail = p;
            }else{
                Singly_Linked_List_Node *p = head;
                while (p->next != p1){
                    p = p->next;
                }
                *tail = p;
            }
        }
        return true;
    }else{
        if (tail != NULL){
            *tail = (fast != NULL) ? fast : slow;
            if (*tail && (*tail)->next){
                while ((*tail)->next){
                    *tail = (*tail)->next;
                }
            }
        }
        return false;
    }
}

bool Singly_Linked_List_Has_Loop(Singly_Linked_List *list)
{
    if (list == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Has_Loop: list is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    read_lock(list);
    Singly_Linked_List_Node *lap;
    bool res = Singly_Linked_List_Find_Loop(list->head, &lap, NULL);
    read_unlock(list);
    return res;
}

Singly_Linked_List_Node *Singly_Linked_List_Loop_Start_Node(Singly_Linked_List *list)
{
    if (list == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Singly_Linked_List_Loop_Start_Node: list is NULL.");
        Madd_Error_Add(MADD_ERROR, error_info);
        return NULL;
    }

    read_lock(list);
    Singly_Linked_List_Node *lap = NULL;
    bool flag_has_loop;
    flag_has_loop = Singly_Linked_List_Find_Loop(list->head, &lap, NULL);
    if (!flag_has_loop){
        read_unlock(list);
        return NULL;
    }
    Singly_Linked_List_Node *p1 = list->head, *p2 = lap;
    while (p1 != p2){
        p1 = p1->next;
        p2 = p2->next;
    }
    read_unlock(list);
    return p1;
}

bool Singly_Linked_List_Link_Node(Singly_Linked_List_Node *prev, Singly_Linked_List_Node *next)
{
    if (prev == NULL){
        Madd_Error_Add(MADD_ERROR, L"Singly_Linked_List_Link_Node: prev is NULL.");
        return false;
    }
    prev->next = next;
    return true;
}

bool Singly_Linked_List_Unlink_Node(Singly_Linked_List_Node *prev)
{
    if (prev == NULL){
        Madd_Error_Add(MADD_ERROR, L"Singly_Linked_List_Unlink_Node: prev is NULL.");
        return false;
    }
    prev->next = NULL;
    return true;
}

bool Singly_Linked_List_Reverse(Singly_Linked_List *list)
{
    if (list == NULL){
        Madd_Error_Add(MADD_ERROR, L"Singly_Linked_List_Reverse: list is NULL.");
        return false;
    }

    write_lock(list);
    if (list->head == NULL || list->head == list->tail){ /* 0 or 1 node in list */
        write_unlock(list);
        return true;
    }

    bool flag_cyclic = (list->tail->next == list->head);
    if (flag_cyclic){
        list->tail->next = NULL;
    }

    Singly_Linked_List_Node *prev=NULL, *curr=list->head, *next=NULL;
    /* the next of head will be processed at the end of while loop */
    while (curr != NULL){
        next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }

    Singly_Linked_List_Node *old_head = list->head;
    list->head = prev;
    list->tail = old_head;

    if (flag_cyclic){
        list->tail->next = list->head;
    }
    write_unlock(list);
    return true;
}

void Singly_Linked_List_Reverse_Nodes(Singly_Linked_List_Node *head)
{
    if (head == NULL || head->next == head || head->next == NULL) return;

    Singly_Linked_List_Node *curr = head, *tail = NULL;
    bool flag_circular = false;
    while (curr->next){
        if (curr->next != NULL){
            flag_circular = true;
            tail = curr;
            break;
        }
        curr = curr->next;
    }
    if (flag_circular){
        tail->next = NULL;
    }

    Singly_Linked_List_Node *prev = NULL;
    curr = head;
    while (curr != NULL){
        Singly_Linked_List_Node *next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    if (flag_circular){
        head->next = prev;
    }
}