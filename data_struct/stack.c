#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#include"data_struct.h"
#include"../basic/basic.h"
#include"../thread_base/thread_base.h"

static size_t Stack_Default_Unit_Capacity = 1<<10;

bool Stack_Init(Stack *stack, uint64_t unit_capacity, size_t usize_ /* element size */)
{
    size_t usize = (usize_) ? usize_ : sizeof(void*);
    if (unit_capacity == 0){
        unit_capacity = Stack_Default_Unit_Capacity;
    }
    stack->capacity = unit_capacity;
    stack->n_element = 0;
    stack->unit_capacity = unit_capacity;
    stack->usize = usize;
    stack->auto_shrink = true;
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Init(&stack->rwlock);
#endif
    stack->buf = (void*)malloc(unit_capacity*usize);
    if (stack->buf == NULL){
        stack->capacity = stack->unit_capacity = 0;
        /* unable to allocate mem for Stack */
        Madd_Error_Add(MADD_ERROR, L"Stack_Init: unable to allocate mem for Stack");
        return false;
    }
    return true;
}

void Stack_Destroy(Stack *stack)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Lock(&stack->rwlock);
#endif
    free(stack->buf);
    stack->buf = NULL;
    stack->capacity = stack->n_element = stack->unit_capacity = 0;
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
    RWLock_Destroy(&stack->rwlock);
#endif
}

void Stack_Shrink(Stack *stack)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Lock(&stack->rwlock);
#endif
    size_t n_unit = stack->capacity/stack->unit_capacity, n_rest = stack->capacity%stack->unit_capacity, new_capacity;
    if (n_rest){
        n_unit ++;
    }
    new_capacity = n_unit * stack->unit_capacity;
    if (new_capacity == stack->capacity){
        /* do not shrink */
        Madd_Error_Add(MADD_ERROR, L"Stack_Shrink: stack size is equal to the given, does not shrink");
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return;
    }
    void *new_buf = realloc(stack->buf, new_capacity*stack->usize);
    if (new_buf == NULL){
        /* unable to realloc */
        Madd_Error_Add(MADD_ERROR, L"Stack_Shrink: unable to re-allocate new mem for stack");
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return;
    }
    stack->buf = new_buf;
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
}

void Stack_Expand(Stack *stack, size_t new_capacity)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Lock(&stack->rwlock);
#endif
    if (new_capacity <= stack->capacity){
        /* new capacity less than current capacity */
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Expand: the given size %llu is less than current capacity %llu", new_capacity, stack->capacity);
        Madd_Error_Add(MADD_ERROR, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return;
    }
    void *new_buf = realloc(stack->buf, new_capacity*stack->usize);
    if (new_buf == NULL){
        /* unable to realloc */
        Madd_Error_Add(MADD_ERROR, L"Stack_Expand: unable to re-allocate new mem for stack");
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return;
    }
    stack->buf = new_buf;
    stack->capacity = new_capacity;
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
}

void Stack_Resize(Stack *stack, size_t new_capacity)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Lock(&stack->rwlock);
#endif
    if (new_capacity < stack->n_element){
        /* new capacity not enough */
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Resize: given size %llu is not enough for current elements %llu", new_capacity, stack->n_element);
        Madd_Error_Add(MADD_ERROR, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return;
    }
    if (new_capacity == stack->capacity){
        /* new capacity same as current capacity */
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Resize: new capacity same as current capacity %llu", new_capacity);
        Madd_Error_Add(MADD_ERROR, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return;
    }
    if (new_capacity % stack->unit_capacity){
        /* new capacity should be the multiple of stack.unit_capacity */
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Resize: new capacity %llu should be the multiple of stack.unit_capacity %llu", new_capacity, stack->unit_capacity);
        Madd_Error_Add(MADD_ERROR, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return;
    }
    void *new_buf = realloc(stack->buf, new_capacity*stack->usize);
    if (new_buf == NULL){
        /* unable to realloc */
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Resize: unable to re-allocate size %llu", new_capacity);
        Madd_Error_Add(MADD_ERROR, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return;
    }
    stack->buf = new_buf;
    stack->capacity = new_capacity;
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
}

bool Stack_Push(Stack *stack, void *element)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Lock(&stack->rwlock);
#endif
    if (stack->capacity == stack->n_element){
        /* expand stack */
        size_t new_capacity = stack->capacity + stack->unit_capacity;
        void *new_buf = realloc(stack->buf, new_capacity*stack->usize);
        if (new_buf == NULL){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Push: unable to push new element due to cannot expand stack for a new capacity %llu", new_capacity);
            Madd_Error_Add(MADD_ERROR, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
            RWLock_Write_Unlock(&stack->rwlock);
#endif
            return false;
        }
        stack->buf = new_buf;
        stack->capacity = new_capacity;
    }
    unsigned char *buf = stack->buf;
    buf += stack->n_element * stack->usize;
    memcpy(buf, element, stack->usize);
    stack->n_element ++;
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
    return true;
}

bool Stack_Pop(Stack *stack, void *element)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Lock(&stack->rwlock);
#endif
    if (stack->n_element == 0){
        /* no element */
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Pop: no element in stack now");
        Madd_Error_Add(MADD_ERROR, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
        return false;
    }
    unsigned char *buf = stack->buf;
    buf += (stack->n_element-1) * stack->usize;
    memcpy(element, buf, stack->usize);
    stack->n_element --;
    /* check if shrink is needed */
    if (stack->auto_shrink && stack->n_element%stack->unit_capacity==0 && stack->capacity > stack->unit_capacity){
        /* shrink stack */
        size_t new_capacity = stack->capacity - stack->unit_capacity;
        void *new_buf = realloc(stack->buf, new_capacity*stack->usize);
        if (new_buf == NULL){
            /* unable to re-allocate */
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Pop: unable to shrink because of cannot re-allocate new mem %llu", new_capacity);
            Madd_Error_Add(MADD_WARNING, error_info);
        }else{
            stack->buf = new_buf;
            stack->capacity = new_capacity;
        }
    }
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Write_Unlock(&stack->rwlock);
#endif
    return true;
}

bool Stack_Top(Stack *stack, void *element)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Lock(&stack->rwlock);
#endif
    if (stack->n_element == 0){
        /* no element */
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Stack_Top: no element in stack now");
        Madd_Error_Add(MADD_WARNING, error_info);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&stack->rwlock);
#endif
        return false;
    }
    unsigned char *buf = stack->buf;
    buf += (stack->n_element-1) * stack->usize;
    memcpy(element, buf, stack->usize);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&stack->rwlock);
#endif
    return true;
}

bool Stack_Empty(Stack *stack)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Lock(&stack->rwlock);
#endif
    bool res = (stack->n_element==0);
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&stack->rwlock);
#endif
    return res;
}

size_t Stack_Size(Stack *stack)
{
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Lock(&stack->rwlock);
#endif
    size_t n = stack->n_element;
#ifdef MADD_ENABLE_MULTITHREAD
    RWLock_Read_Unlock(&stack->rwlock);
#endif
    return n;
}