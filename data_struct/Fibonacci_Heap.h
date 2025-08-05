/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./data_struct/Fibonacci_Heap.h
*/
#ifndef _FIBONACCI_HEAP_H
#define _FIBONACCI_HEAP_H

#include<stdint.h>
#include<stdlib.h>
#include<stdbool.h>
#include"../thread_base/thread_base.h"

#define FIBONACCI_HEAP_DECREASE_KEY_SUCCESS 0
#define FIBONACCI_HEAP_DECREASE_KEY_FAIL 1

struct _Fibonacci_Heap_Node{
    uint8_t mark;
    uint64_t degree;
    struct _Fibonacci_Heap_Node *left,*right,*p,*child;
    void *key;
};

typedef struct _Fibonacci_Heap_Node Fibonacci_Heap_Node;

typedef struct{
    bool flag_multithread;
    uint64_t n;
    Fibonacci_Heap_Node *min;
    RWLock rwlock;
} Fibonacci_Heap;

Fibonacci_Heap Fibonacci_Heap_Make(void);
bool Fibonacci_Heap_Enable_Multithread(Fibonacci_Heap *H);
void Fibonacci_Heap_Insert(Fibonacci_Heap *H, Fibonacci_Heap_Node *x,
                           char func(void *key1,void *key2,void *other_param), void *other_param);
Fibonacci_Heap Fibonacci_Heap_Union(Fibonacci_Heap *H1, Fibonacci_Heap *H2,
                                    char func(void *key1,void *key2,void *other_param), void *other_param);
void Fibonacci_Heap_Link(/*Fibonacci_Heap H, */Fibonacci_Heap_Node *y, Fibonacci_Heap_Node *x,
                         char func(void *key1,void *key2,void *other_param), void *other_param);
void Fibonacci_Heap_Consolidate(Fibonacci_Heap *H,
                                char func(void *key1,void *key2,void *other_param), void *other_param);
Fibonacci_Heap_Node *Fibonacci_Heap_Extract_Min(Fibonacci_Heap *H,
                                                char func(void *key1,void *key2,void *other_param), void *other_param);
void Fibonacci_Heap_Cut(Fibonacci_Heap *H, Fibonacci_Heap_Node *x, Fibonacci_Heap_Node *y,
                        char func(void *key1,void *key2,void *other_param), void *other_param);
void Fibonacci_Heap_Cascading_Cut(Fibonacci_Heap *H, Fibonacci_Heap_Node *y,
                                  char func(void *key1,void *key2,void *other_param), void *other_param);
char Fibonacci_Heap_Decrease_Key(Fibonacci_Heap *H, Fibonacci_Heap_Node *x, void *k,
                                 char func(void *key1,void *key2,void *other_param), void *other_param/*, char purpose*/);
char Fibonacci_Heap_Delete__Func(void *key1, void *key2, void *other_delete_param_);
void Fibonacci_Heap_Delete(Fibonacci_Heap *H, Fibonacci_Heap_Node *x,
                           char func(void *key1,void *key2,void *other_param), void *other_param);

#endif /* _FIBONACCI_HEAP_H */
