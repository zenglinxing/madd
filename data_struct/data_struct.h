/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/data_struct.h
*/
#ifndef _DATA_STRUCT_H
#define _DATA_STRUCT_H

#include<stdint.h>
#include<stddef.h>
#include<stdbool.h>
#include"binary_search_tree.h"
#include"Fibonacci_Heap.h"
#include"RB_tree.h"
#include"../thread_base/thread_base.h"

/* stack */
typedef struct{
    bool flag_multithread, auto_shrink;
    void *buf;
    size_t capacity, n_element, unit_capacity, usize;
    RWLock rwlock;
} Stack;

bool Stack_Init(Stack *stack, uint64_t unit_capacity, size_t usize /* element size */);
bool Stack_Enable_Multithread(Stack *stack);
void Stack_Destroy(Stack *stack);
void Stack_Shrink(Stack *stack);
void Stack_Expand(Stack *stack, size_t new_capacity);
void Stack_Resize(Stack *stack, size_t new_capacity);
bool Stack_Push(Stack *stack, void *element);
bool Stack_Pop(Stack *stack, void *element);
bool Stack_Top(Stack *stack, void *element);
bool Stack_Empty(Stack *stack);
size_t Stack_Size(Stack *stack);

/* linked list */
struct _Linked_List_Node{
    bool flag_multithread;
    void *buf;
    size_t usize;
    struct _Linked_List_Node *prev, *next;
    RWLock rwlock;
};
typedef struct _Linked_List_Node Linked_List_Node;

bool Linked_List_Init(Linked_List_Node *node, size_t usize);
bool Linked_List_Enable_Multithread(Linked_List_Node *node);
void Linked_List_Destroy(Linked_List_Node *node);
bool Linked_List_Link(Linked_List_Node *prev, Linked_List_Node *next, bool flag_bidirection);
bool Linked_List_Delete(Linked_List_Node *node);
bool Linked_List_Insert_After(Linked_List_Node *prev, Linked_List_Node *next, bool flag_bidirection);

/* queue */
struct _Queue_Node{
    uint64_t start, end, n_element; /* convention: start + n_element = end */
    struct _Queue_Node *prev, *next;
    unsigned char *buf;
};
typedef struct _Queue_Node Queue_Node;

typedef struct{
    bool flag_multithread;
    uint64_t unit_capacity, n_element;
    size_t usize;
    Queue_Node *head;
    RWLock rwlock;
} Queue;

bool Queue_Init(Queue *queue, uint64_t unit_capacity, size_t usize);
bool Queue_Enable_Multithread(Queue *queue);
void Queue_Destroy(Queue *queue);
bool Queue_Enqueue(Queue *queue, void *element);
bool Queue_Dequeue(Queue *queue, void *element);
bool Queue_Get_Head(Queue *queue, void *element);
bool Queue_Get_Last(Queue *queue, void *element);

#endif /* _DATA_STRUCT_H */