/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/queue.c
*/
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include"data_struct.h"
#include"../basic/basic.h"

#define QUEUE_READ_LOCK(Q) \
{ \
    if ((Q)->flag_multithread){ \
        RWLock_Read_Lock(&(Q)->rwlock); \
    } \
} \

#define QUEUE_READ_UNLOCK(Q) \
{ \
    if ((Q)->flag_multithread){ \
        RWLock_Read_Unlock(&(Q)->rwlock); \
    } \
} \

#define QUEUE_WRITE_LOCK(Q) \
{ \
    if ((Q)->flag_multithread){ \
        RWLock_Write_Lock(&(Q)->rwlock); \
    } \
} \

#define QUEUE_WRITE_UNLOCK(Q) \
{ \
    if ((Q)->flag_multithread){ \
        RWLock_Write_Unlock(&(Q)->rwlock); \
    } \
} \

static uint64_t madd_queue_default_len = 1024;

bool Queue_Init(Queue *queue, uint64_t unit_capacity, size_t usize)
{
    if (queue == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Init: queue pointer is NULL.");
        return false;
    }
    size_t size_required = sizeof(Queue_Node)+queue->unit_capacity*queue->usize;
    queue->unit_capacity = (unit_capacity) ? unit_capacity : madd_queue_default_len;
    queue->usize = (usize) ? usize : sizeof(void*);
    queue->flag_multithread = false;
    Queue_Node *node = queue->head = (Queue_Node*)malloc(size_required);
    if (node == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%s: unable to allocate mem %llu bytes.", __func__, size_required);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    node->start = node->end = node->n_element = 0;
    node->prev = node->next = node;
    node->buf = (unsigned char*)(node + 1);
    return true;
}

bool Queue_Enable_Multithread(Queue *queue)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (queue->flag_multithread){
        Madd_Error_Add(MADD_WARNING, L"Queue_Enable_Multithread: queue already has read-write lock initialized.");
        return false;
    }
    RWLock_Init(&queue->rwlock);
    queue->flag_multithread = true;
    return true;
#else
    queue->flag_multithread = false;
    Madd_Error_Add(MADD_WARNING, L"Queue_Enable_Multithread: Madd lib multithread wasn't enabled during compiling. Tried to enable Madd's multithread and re-compile Madd.");
    return false;
#endif
}

void Queue_Destroy(Queue *queue)
{
    if (queue == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Init: queue pointer is NULL.");
        return;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_WRITE_LOCK(queue)
#endif

    Queue_Node *root_node = queue->head, *last_node = root_node->prev, *next_node = root_node->next, *current_node = root_node;
    do {
        if (current_node == NULL){
            Madd_Error_Add(MADD_ERROR, L"Queue_Destroy: one node pointer of queue is NULL.");
            return;
        }
        next_node = next_node->next;
        free(current_node);
        current_node = next_node;
    } while (current_node != root_node);

    queue->unit_capacity = queue->usize = 0;
    queue->head = NULL;

#ifdef MADD_ENABLE_MULTITHREAD
    if (queue->flag_multithread){
        RWLock_Write_Unlock(&queue->rwlock);
        RWLock_Destroy(&queue->rwlock);
    }
#endif
    queue->flag_multithread = false;
}

static bool Queue_Insert_New_Node(Queue *queue)
{
    Queue_Node *head = queue->head, *last = head->prev;
    size_t size_required = sizeof(Queue_Node)+queue->unit_capacity*queue->usize;
    Queue_Node *node = (Queue_Node*)malloc(size_required);
    if (node == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%s: unable to allocate mem %llu for new node. See Madd source %s line %d.", __func__, size_required, __FILE__, __LINE__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    node->start = node->end = node->n_element = 0;
    node->buf = (unsigned char*)(node + 1);
    node->prev = last;
    node->next = head;
    head->prev = last->next = node;
    return true;
}

bool Queue_Enqueue(Queue *queue, void *element)
{
    if (queue == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Enqueue: queue pointer is NULL.");
        return false;
    }
    if (element == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Enqueue: element pointer is NULL.");
        return false;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_WRITE_LOCK(queue)
#endif

    Queue_Node *head = queue->head, *last;
    if (head == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Enqueue: corrupted queue, the head node of queue is NULL.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_WRITE_UNLOCK(queue)
#endif
        return false;
    }
    last = queue->head->prev;
    if (last == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Enqueue: corrupted queue, the last node of queue is NULL.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_WRITE_UNLOCK(queue)
#endif
        return false;
    }
    if (last->n_element && last->end == queue->unit_capacity){
        bool res_insert = Queue_Insert_New_Node(queue);
        if (!res_insert){
            Madd_Error_Add(MADD_ERROR, L"Queue_Enqueue: unable to insert new element, because not allocate mem. See info from Queue_Insert_New_Node.");
#ifdef MADD_ENABLE_MULTITHREAD
            QUEUE_WRITE_UNLOCK(queue)
#endif
            return false;
        }
    }
    last = queue->head->prev;
    memcpy(last->buf+queue->usize*last->end, element, queue->usize);
    last->end ++;
    last->n_element ++;
    queue->n_element ++;

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_WRITE_UNLOCK(queue)
#endif
    return true;
}

static bool Queue_Delete_Head_Node(Queue *queue)
{
    Queue_Node *head = queue->head, *last = head->prev, *next = head->next;
    if (last == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%s: corrupted queue, the last node of queue is NULL. See Madd source %s line %d.", __func__, __FILE__, __LINE__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (next == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%s: corrupted queue, the next-to-head node of queue is NULL. See Madd source %s line %d.", __func__, __FILE__, __LINE__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    queue->head = next;
    next->prev = last;
    last->next = next;
    free(head);
    return true;
}

bool Queue_Dequeue(Queue *queue, void *element)
{
    if (queue == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Dequeue: queue pointer is NULL.");
        return false;
    }
    if (element == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Dequeue: element pointer is NULL.");
        return false;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_WRITE_LOCK(queue)
#endif

    Queue_Node *head = queue->head, *last;
    if (head == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Dequeue: corrupted queue, the head node of queue is NULL.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_WRITE_UNLOCK(queue)
#endif
        return false;
    }
    last = head->prev;
    if (last == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Dequeue: corrupted queue, the last node of queue is NULL.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_WRITE_UNLOCK(queue)
#endif
        return false;
    }
    if (head->n_element == 0){
        Madd_Error_Add(MADD_WARNING, L"Queue_Dequeue: queue has no element.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_WRITE_UNLOCK(queue)
#endif
        return false;
    }
    memcpy(element, head->buf+head->start*queue->usize, queue->usize);
    head->start ++;
    head->n_element --;
    queue->n_element --;
    if (head->n_element == 0){
        if (head->next == head){ /* only one node in queue */
            head->start = head->end = head->n_element = 0;
        }else{
            bool res_delete = Queue_Delete_Head_Node(queue);
            if (!res_delete){
                Madd_Error_Add(MADD_ERROR, L"Queue_Dequeue: error when deleting head node of queue. See error info from Queue_Delete_Head_Node.");
#ifdef MADD_ENABLE_MULTITHREAD
                QUEUE_WRITE_UNLOCK(queue)
#endif
                return false;
            }
        }
    }

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_WRITE_UNLOCK(queue);
#endif
    return true;
}

bool Queue_Get_Head(Queue *queue, void *element)
{
    if (queue == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Get_Head: queue pointer is NULL.");
        return false;
    }
    if (element == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Get_Head: element pointer is NULL.");
        return false;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_READ_LOCK(queue)
#endif

    Queue_Node *head = queue->head;
    if (head == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Get_Head: corrupted queue, the head node of queue is NULL.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_READ_UNLOCK(queue)
#endif
        return false;
    }
    if (head->n_element == 0){
        Madd_Error_Add(MADD_WARNING, L"Queue_Get_Head: queue has no element.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_READ_UNLOCK(queue)
#endif
        return false;
    }
    memcpy(element, head->buf + head->start * queue->usize, queue->usize);

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_READ_UNLOCK(queue)
#endif
    return true;
}

bool Queue_Get_Last(Queue *queue, void *element)
{
    if (queue == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Get_Last: queue pointer is NULL.");
        return false;
    }
    if (element == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Get_Last: element pointer is NULL.");
        return false;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_READ_LOCK(queue)
#endif

    Queue_Node *head = queue->head, *last;
    if (head == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Get_Last: corrupted queue, the head node of queue is NULL.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_READ_UNLOCK(queue)
#endif
        return false;
    }
    last = head->prev;
    if (last == NULL){
        Madd_Error_Add(MADD_ERROR, L"Queue_Get_Last: corrupted queue, the last node of queue is NULL.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_READ_UNLOCK(queue)
#endif
        return false;
    }
    if (last->n_element == 0){
        Madd_Error_Add(MADD_WARNING, L"Queue_Get_Last: queue has no element.");
#ifdef MADD_ENABLE_MULTITHREAD
        QUEUE_READ_UNLOCK(queue)
#endif
        return false;
    }
    memcpy(element, last->buf + (last->end-1) * queue->usize, queue->usize);

#ifdef MADD_ENABLE_MULTITHREAD
    QUEUE_READ_UNLOCK(queue)
#endif
    return true;
}