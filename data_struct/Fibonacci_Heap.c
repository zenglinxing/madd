/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./data_struct/Fibonacci_Heap.c

The Fibonacci_Heap_Delete is mainly contributed by Deepseek
*/
#include<stdint.h>
#include<stdlib.h>
#include"Fibonacci_Heap.h"
#include"../basic/basic.h"
/*#include"../thread_base/thread_base.h"*/

#define FH_READ_LOCK(H) \
    if ((H) != NULL && (H)->flag_multithread){ \
        RWLock_Read_Lock(&(H)->rwlock); \
    } \

#define FH_READ_UNLOCK(H) \
    if ((H) != NULL && (H)->flag_multithread){ \
        RWLock_Read_Unlock(&(H)->rwlock); \
    } \

#define FH_WRITE_LOCK(H) \
    if ((H) != NULL && (H)->flag_multithread){ \
        RWLock_Write_Lock(&(H)->rwlock); \
    } \

#define FH_WRITE_UNLOCK(H) \
    if ((H) != NULL && (H)->flag_multithread){ \
        RWLock_Write_Unlock(&(H)->rwlock); \
    } \

bool Fibonacci_Heap_Init(Fibonacci_Heap *H)
{
    if (H == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Init: Fibonacci heap pointer is NULL.");
        return false;
    }
    H->n = 0;
    H->min = NULL;
    H->flag_multithread = false;
    return true;
}

bool Fibonacci_Heap_Enable_Multithread(Fibonacci_Heap *H)
{
#ifdef MADD_ENABLE_MULTITHREAD
    if (H->flag_multithread){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Enable_Multithread: heap already has read-write lock initialized.");
        return false;
    }
    RWLock_Init(&H->rwlock);
    H->flag_multithread = true;
    return true;
#else
    H->flag_multithread = false;
    Madd_Error_Add(MADD_WARNING, L"Fibonacci_Heap_Enable_Multithread: Madd lib multithread wasn't enabled during compiling. Tried to enable Madd's multithread and re-compile Madd.");
    return false;
#endif
}

typedef char (*Fibonacci_Heap_Delete_Func_Param)(void *,void *,void *);

typedef struct{
    Fibonacci_Heap_Delete_Func_Param func;
    void *other_param;
    void *useless_var;
} Fibonacci_Heap_Delete_Param;

void Fibonacci_Heap_Insert(Fibonacci_Heap *H, Fibonacci_Heap_Node *x, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    if (H == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Insert: the heap pointer is NULL.");
        return;
    }
    if (x == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Insert: the heap node pointer is NULL.");
        return;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Insert: the func pointer is NULL.");
        return;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    FH_WRITE_LOCK(H)
#endif

    x->degree = 0;
    x->p = x->child = NULL;
    x->mark = 0;
    if (H->min == NULL){
        H->min = x;
        x->left = x->right = x;
    }
    else{
        /* insert x into H's root list */
        x->left = H->min->left;
        x->right = H->min;
        H->min->left->right = x;
        H->min->left = x;
        if (func(x->key, H->min->key, other_param)==MADD_LESS){ /* x->key < H->min->key */
            H->min = x;
        }
    }
    H->n ++;

#ifdef MADD_ENABLE_MULTITHREAD
    FH_WRITE_UNLOCK(H)
#endif
}

Fibonacci_Heap Fibonacci_Heap_Union(Fibonacci_Heap *H1, Fibonacci_Heap *H2, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    Fibonacci_Heap H = {.min = NULL, .flag_multithread=false, .n=0};
    if (H1 == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Union: the 1st heap pointer is NULL.");
        return H;
    }
    if (H2 == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Union: the 2nd heap pointer is NULL.");
        return H;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Union: the func pointer is NULL.");
        return H;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    FH_READ_LOCK(H1)
    FH_READ_LOCK(H2)
#endif

    H.min = H1->min;
    /* concatenate the root list of H2 with the root list of H */
    if (H1->min != NULL && H2->min != NULL){
        H2->min->right->left = H.min->left;
        H.min->left->right = H2->min->right;
        H.min->left = H2->min;
        H2->min->right = H.min;
    }
    /*else if (H1.min != NULL);*/ /* nothing to do */
    /*else if (H2.min != NULL);*/ /* nothing to do */
    if (H1->min == NULL || ( H2->min != NULL && func(H2->min->key, H1->min->key, other_param)==MADD_LESS )){
        H.min = H2->min;
    }
    H.n = H1->n + H2->n;

#ifdef MADD_ENABLE_MULTITHREAD
    FH_READ_UNLOCK(H1)
    FH_READ_UNLOCK(H2)
#endif

    return H;
}

void Fibonacci_Heap_Link(/*Fibonacci_Heap H, */Fibonacci_Heap_Node *y, Fibonacci_Heap_Node *x, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    if (y == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Link: the 1st heap node pointer is NULL.");
        return;
    }
    if (x == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Link: the 2nd heap node pointer is NULL.");
        return;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Link: the func pointer is NULL.");
        return;
    }

    /* remove y from root list */
    y->left->right=y->right;
    y->right->left=y->left;
    /* make y a child of x, and insert it at right of x.child */
    y->p=x;
    if (x->degree>0){
        y->left=x->child;
        y->right=x->child->right;
        x->child->right->left=y;
        x->child->right=y;
        if (func(y->key, x->child->key, other_param)==MADD_LESS){
            x->child=y;
        }
    }else{ /* y is the only child of x */
        x->child=y;
        y->left=y->right=y;
    }
    /* incrementing x.degree */
    x->degree++;
    y->mark=0;
}

void Fibonacci_Heap_Consolidate(Fibonacci_Heap *H, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    if (H == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Union: the heap pointer is NULL.");
        return;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Union: the func pointer is NULL.");
        return;
    }

    uint64_t max_degree = (uint64_t)(1.5 * log2(H->n + 1)) + 1;
    Fibonacci_Heap_Node **A = malloc((max_degree + 1) * sizeof(*A));
    register uint64_t i,d;
    for (i=0;i<H->n;i++){
        A[i]=NULL;
    }
    register Fibonacci_Heap_Node /**H_first=H->min,*/*H_last=H->min->left,*w=H->min,*x,*y,*next,*temp;
    /* Iterate each node in root list */
    i=0;
    /* Here is optimized, since siblings won't point to the same address */
    while (1){
        /*printf("consolidate\ti=%llu\n",i);*/
        x=w;
        next=w->right;
        d=x->degree;
        while (A[d]!=NULL){
            /*printf("consolidate\td=%llu\n",d);*/
            y=A[d]; /* another node with the same degree as x */
            if (func(y->key, x->key, other_param)==MADD_LESS){
            /*if (x->unit.weight>y->unit.weight){*/ /* swap x y */
                temp=x;
                x=y;
                y=temp;
            }
            Fibonacci_Heap_Link(/* *H, */y,x,func,other_param);
            A[d]=NULL;
            d++;
        }
        A[d]=x;
        if (w==H_last) break;
        w=next;
        i++;
    }
    H->min=NULL;
    for (i=0;i<H->n;i++){
        w=A[i];
        if (w==NULL) continue;
        /* insert w into root list */
        if (H->min==NULL){
            H->min=w;
            H->min->left = H->min->right = H->min;
        }else{
            w->left=H->min;
            w->right=H->min->right;
            H->min->right->left=w;
            H->min->right=w;
            if (func(w->key, H->min->key, other_param)==MADD_LESS){
            /*if (w->unit.weight<H->min->unit.weight){*/
                H->min=w;
            }
        }
    }
    free(A);
}

static Fibonacci_Heap_Node *Fibonacci_Heap_Extract_Min_Internal(Fibonacci_Heap *H, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    if (H == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Extract_Min_Internal: the heap pointer is NULL.");
        return NULL;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Extract_Min_Internal: the func pointer is NULL.");
        return NULL;
    }

    register Fibonacci_Heap_Node *first,*last,*z=H->min,*temp;
    if (z!=NULL){
        /* Add z's children to root list */
        if (z->child!=NULL){
            first=z->child; /* first is z.child */
            last=first->left; /* last is z.child.left */
            /* Add the list [first -> last] between z and z.right */
            first->left=z;
            last->right=z->right;
            z->right->left=last;
            z->right=first;
            /* Remove parents of [first -> last] */
            temp=first;
            while (temp!=last){
                temp->p=NULL;
                temp=temp->right;
            }
            temp->p=NULL;
            z->degree=0;
            z->child=NULL;
        }
        /* remove z from the root list of H */
        z->left->right=z->right;
        z->right->left=z->left;
        if (z==z->right){ /* if z WAS the only one in root list */
            H->min=NULL;
        }else{
            H->min=z->right;
            /*printf("ExtractMin,H.min\t%f\t%llu\t%llu\tn=%llu\n",H->min->unit.weight,H->min->unit.id,H->min->id,H->n);*/
            Fibonacci_Heap_Consolidate(H,func,other_param); /* H.min will be determined here */
            /*printf("ExtractMin,H.min\t%f\t%llu\t%llu\tn=%llu\n",H->min->unit.weight,H->min->unit.id,H->min->id,H->n);*/
        }
        H->n--;
    }
    return z;
}

Fibonacci_Heap_Node *Fibonacci_Heap_Extract_Min(Fibonacci_Heap *H, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    if (H == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Extract_Min: the heap pointer is NULL.");
        return NULL;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Extract_Min: the func pointer is NULL.");
        return NULL;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    FH_WRITE_LOCK(H)
#endif

    Fibonacci_Heap_Node *z = Fibonacci_Heap_Extract_Min_Internal(H, func, other_param);

#ifdef MADD_ENABLE_MULTITHREAD
    FH_WRITE_UNLOCK(H)
#endif

    return z;
}

void Fibonacci_Heap_Cut(Fibonacci_Heap *H, Fibonacci_Heap_Node *x, Fibonacci_Heap_Node *y, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    register void *min;
    register uint64_t i;
    register Fibonacci_Heap_Node *p,*pmin,*last;
    /* remove x from the child list of y, decrementing y.degree */
    if (y->child==x){
        if (x->right==x){ /* x is the only child of y */
            y->child=NULL;
        }else{
            x->right->left=x->left;
            x->left->right=x->right;
            y->child=x->right;
            /* find min of y's children */
            min=y->child->key;
            pmin=y->child;
            p=y->child;
            last=y->child->left;
            i=0;
            while (p!=last){
                if (func(p->key, min, other_param)==MADD_LESS){
                /*if (min > p->unit.weight){*/
                    pmin=p;
                    min=p->key;
                }
                p=p->right;
                i++;
            }
            if (func(p->key, min, other_param)==MADD_LESS){
            /*if (min > p->unit.weight){*/ /* check the last one */
                pmin=p;
                min=p->key;
            }
            y->child=pmin;
        }
    }else{
        x->right->left=x->left;
        x->left->right=x->right;
    }
    y->degree--;
    /* Add x to the root list of H */
    x->right=H->min->right;
    x->left=H->min;
    H->min->right->left=x;
    H->min->right=x;
    x->p=NULL;
    x->mark=0;
}

void Fibonacci_Heap_Cascading_Cut(Fibonacci_Heap *H, Fibonacci_Heap_Node *y, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    Fibonacci_Heap_Node *current = y;
    while (current && current->p) {
        if (!current->mark) {
            current->mark = 1;
            break;
        }
        Fibonacci_Heap_Node *parent = current->p;
        Fibonacci_Heap_Cut(H, current, parent, func, other_param);
        current = parent;
    }
}
/*{
    register Fibonacci_Heap_Node *z=y->p;
    if (z!=NULL){
        if (y->mark==0){
            y->mark=1;
        }else{
            Fibonacci_Heap_Cut(H,y,z,func,other_param);
            Fibonacci_Heap_Cascading_Cut(H,z,func,other_param);
        }
    }
}*/

void Fibonacci_Heap_Decrease_Key_Internal(Fibonacci_Heap *H, Fibonacci_Heap_Node *x, void *k, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    x->key=k;
    Fibonacci_Heap_Node *y=x->p;
    if (y!=NULL){
        if (func(x->key, y->key, other_param)==MADD_LESS){
        /*if (x->key < y->key){*/
            Fibonacci_Heap_Cut(H,x,y,func,other_param);
            Fibonacci_Heap_Cascading_Cut(H,y,func,other_param);
        }
    }
    if (func(x->key, H->min->key, other_param)==MADD_LESS){
    /*if (x->key < H->min->key){*/
        H->min=x;
    }
}

char Fibonacci_Heap_Decrease_Key(Fibonacci_Heap *H, Fibonacci_Heap_Node *x, void *k, char func(void *key1,void *key2,void *other_param), void *other_param/*, char purpose*/)
{
    if (H == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Decrease_Key: the heap pointer is NULL.");
        return FIBONACCI_HEAP_DECREASE_KEY_FAIL;
    }
    if (k == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Decrease_Key: the key pointer is NULL.");
        return FIBONACCI_HEAP_DECREASE_KEY_FAIL;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Decrease_Key: the func pointer is NULL.");
        return FIBONACCI_HEAP_DECREASE_KEY_FAIL;
    }

    if (func(x->key, k, other_param)==MADD_LESS/* && purpose == FIBONACCI_HEAP_NORMAL*/){
    /*if (k > x->key){*/
        /*printf("new key is greater than current key");*/
        return FIBONACCI_HEAP_DECREASE_KEY_FAIL;
    }
    Fibonacci_Heap_Decrease_Key_Internal(H, x, k, func, other_param);
    return FIBONACCI_HEAP_DECREASE_KEY_SUCCESS;
}

char Fibonacci_Heap_Delete__Func(void *key1, void *key2, void *other_delete_param_)
{
    Fibonacci_Heap_Delete_Param *other_delete_param=(Fibonacci_Heap_Delete_Param*)other_delete_param_;
    if (key1 == other_delete_param->useless_var){
        return MADD_LESS;
    }
    else if (key2 == other_delete_param->useless_var){
        return MADD_GREATER;
    }
    else{
        return other_delete_param->func(key1,key2,other_delete_param->other_param);
    }
}

void Fibonacci_Heap_Delete(Fibonacci_Heap *H, Fibonacci_Heap_Node *x, char func(void *key1,void *key2,void *other_param), void *other_param)
{
    if (H == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Delete: the heap pointer is NULL.");
        return;
    }
    if (x == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Delete: the key pointer is NULL.");
        return;
    }
    if (func == NULL){
        Madd_Error_Add(MADD_ERROR, L"Fibonacci_Heap_Delete: the func pointer is NULL.");
        return;
    }

#ifdef MADD_ENABLE_MULTITHREAD
    FH_WRITE_LOCK(H)
#endif

    if (x == H->min) {
        // 如果要删除的是最小节点，直接调用提取最小节点函数
        Fibonacci_Heap_Extract_Min_Internal(H, func, other_param);
    } else {
        // 1. 将节点从父节点中移除（如果有父节点）
        if (x->p != NULL) {
            Fibonacci_Heap_Node *parent = x->p;
            
            // 从父节点的子链表中移除x
            if (x->right == x) { // x是唯一子节点
                parent->child = NULL;
            } else {
                x->left->right = x->right;
                x->right->left = x->left;
                
                // 如果父节点指向x，更新父节点的child指针
                if (parent->child == x) {
                    // 找到新的最小子节点
                    Fibonacci_Heap_Node *child = x->right;
                    Fibonacci_Heap_Node *minChild = child;
                    void *minKey = child->key;
                    Fibonacci_Heap_Node *current = child->right;
                    
                    while (current != x->right) {
                        if (func(current->key, minKey, other_param) == MADD_LESS) {
                            minKey = current->key;
                            minChild = current;
                        }
                        current = current->right;
                    }
                    parent->child = minChild;
                }
            }
            parent->degree--;
            
            // 触发级联切断
            Fibonacci_Heap_Cascading_Cut(H, parent, func, other_param);
        }
        
        // 2. 将节点的所有子节点提升到根链表
        if (x->child != NULL) {
            Fibonacci_Heap_Node *child = x->child;
            Fibonacci_Heap_Node *firstChild = child;
            
            do {
                Fibonacci_Heap_Node *nextChild = child->right;
                
                // 清除父指针和标记
                child->p = NULL;
                child->mark = 0;
                
                // 插入到根链表
                child->left = H->min->left;
                child->right = H->min;
                H->min->left->right = child;
                H->min->left = child;
                
                child = nextChild;
            } while (child != firstChild);
        }
        
        // 3. 从根链表移除节点
        x->left->right = x->right;
        x->right->left = x->left;
        
        // 4. 更新节点计数
        H->n--;
        
        // 5. 检查是否需要更新min指针
        if (func(x->key, H->min->key, other_param) == MADD_LESS) {
            // 如果被删除的节点比当前最小节点小（虽然理论上不应该发生）
            // 需要遍历根链表找到新的最小节点
            Fibonacci_Heap_Node *minNode = H->min;
            Fibonacci_Heap_Node *current = H->min->right;
            
            while (current != H->min) {
                if (func(current->key, minNode->key, other_param) == MADD_LESS) {
                    minNode = current;
                }
                current = current->right;
            }
            H->min = minNode;
        }
    }

#ifdef MADD_ENABLE_MULTITHREAD
    FH_WRITE_UNLOCK(H)
#endif
}