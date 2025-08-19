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

/* singly linked list */
struct _Singly_Linked_List_Node{
    void *key;
    struct _Singly_Linked_List_Node *next;
};
typedef struct _Singly_Linked_List_Node Singly_Linked_List_Node;

typedef struct{
    bool flag_multithread;
    Singly_Linked_List_Node *head, *tail;
    RWLock rwlock;
} Singly_Linked_List;

void Singly_Linked_List_Init(Singly_Linked_List *list);
bool Singly_Linked_List_Enable_Multithread(Singly_Linked_List *list);
bool Singly_Linked_List_Empty(Singly_Linked_List *list);
bool Singly_Linked_List_Insert_Tail(Singly_Linked_List *list, Singly_Linked_List_Node *node);
bool Singly_Linked_List_Insert_After(Singly_Linked_List *list, Singly_Linked_List_Node *prev, Singly_Linked_List_Node *node);
bool Singly_Linked_List_Delete(Singly_Linked_List *list, Singly_Linked_List_Node *node);
bool Singly_Linked_List_Delete_After(Singly_Linked_List *list, Singly_Linked_List_Node *prev);
bool Singly_Linked_List_Find_Loop(Singly_Linked_List_Node *head,
                                  Singly_Linked_List_Node **loop_start,
                                  Singly_Linked_List_Node **tail);
bool Singly_Linked_List_Has_Loop(Singly_Linked_List *list);
Singly_Linked_List_Node *Singly_Linked_List_Loop_Start_Node(Singly_Linked_List *list);
bool Singly_Linked_List_Link_Node(Singly_Linked_List_Node *prev, Singly_Linked_List_Node *next);
bool Singly_Linked_List_Unlink_Node(Singly_Linked_List_Node *prev);
bool Singly_Linked_List_Reverse(Singly_Linked_List *list);
void Singly_Linked_List_Reverse_Nodes(Singly_Linked_List_Node *head);

#endif /* _DATA_STRUCT_H */