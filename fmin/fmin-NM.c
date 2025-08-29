/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/fmin-NM.c
Nelder-Mead Search
*/
#include<stdlib.h>
#include<stdint.h>
#include<string.h>
#include<wchar.h>
#include"fmin.h"
#include"../basic/basic.h"
#include"../data_struct/RB_tree.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define FMIN_NM_PRINT_LEN 100

#define FMIN_NM_PARAM(num_type, Fmin_NM_Element, Fmin_NM_Param) \
struct Fmin_NM_Element{ \
    uint64_t id; \
    num_type y, *x; \
}; \
typedef struct{ \
    uint64_t np, nx; \
    struct Fmin_NM_Element *element; \
} Fmin_NM_Param; \

#define FMIN_NM_COMPARE__ALGORITHM(num_type, Fmin_NM_Element, PRINT_INFO) \
{ \
    struct Fmin_NM_Element *key1=key1_, *key2=key2_; \
    if (key1->y < key2->y)      return MADD_LESS; \
    else if (key1->y > key2->y) return MADD_GREATER; \
    else if (key1->y == key2->y){ \
        if (key1->id < key2->id)        return MADD_LESS; \
        else if (key1->id > key2->id)   return MADD_GREATER; \
        else{ \
            Madd_Error_Add(MADD_ERROR, PRINT_INFO); \
            return MADD_SAME; \
        } \
    } \
} \

#define FMIN_NM__ALGORITHM(num_type, num_print_type, Fmin_NM_Element, Fmin_NM_Param, Fmin_NM_Compare) \
{ \
    size_t size_cpy=sizeof(num_type)*n_param /* n_param *//*, Size_cpy=size_cpy*n_param+size_cpy*/ /* n_cpy*(n_param+1) */; \
    uint64_t ix, i_param, nx=(uint64_t)n_param+1, nn_param=nx*n_param; \
    size_t total_size = nx * (sizeof(struct Fmin_NM_Element) + sizeof(RB_Tree_Node)) \
                   + 6 * size_cpy; \
    void *space=(void*)malloc(total_size); \
    if (space == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to allocate mem %llu bytes.", __func__, total_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return FMIN_NM_FAIL; \
    } \
 \
    /* NM parameter */ \
    struct Fmin_NM_Element *pnme; \
    Fmin_NM_Param nm={.np=n_param, .nx=nx}; \
    RB_Tree T; \
    RB_Tree_Init(&T); \
    RB_Tree_Node *nodes=(RB_Tree_Node*)space; \
    /* parameter print info */ \
    wchar_t *print_info; \
    size_t print_len, print_where; \
    int print_temp_len; \
    if (print_step){ \
        print_len = (50 + (n_param*15 + 20) * (n_param + 1))*sizeof(wchar_t); \
        print_info = (wchar_t*)malloc(print_len); \
        if (print_info == NULL){ \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to allocate mem %llu bytes for printing.", __func__, print_len); \
        } \
    } \
 \
    /* sort the x according to y */ \
    nm.element = (struct Fmin_NM_Element*)(nodes + nx); \
 \
    /* initialize NM parameter */ \
    for (ix=0,pnme=nm.element /*nm.element*/; ix<nx; ix++,pnme++){ \
        /* pnme=nm.element+ix; */ \
        pnme->id = ix; \
        pnme->x = start[ix]; \
        pnme->y = func(pnme->x,other_param); /* function value */ \
        /* RB Tree nodes */ \
        nodes[ix].key = (void*)pnme; \
        RB_Tree_Insert(&T, nodes+ix, Fmin_NM_Compare, NULL, RB_TREE_DISABLE_SAME_KEY); \
    } \
    /* x mean & x sum */ \
    num_type *x_list=(num_type*)(nm.element+nx); \
    num_type *x_mean=x_list, *x_sum=x_list+n_param, *xr=x_list+2*n_param /* 2*x_mean+x_h */, yr; /* xh is nm.max->x */ \
    num_type *xe=x_list+3*n_param /* 3*x_mean-2*xh */, ye, *xoc=x_list+4*n_param /* 1.5*x_mean-0.5*xh */, yoc; \
    num_type *xic=x_list+5*n_param, yic /* 1.5*x_mean+0.5*xh */; \
    /* Search */ \
    uint64_t i_step, print_next_step=print_start, id_max; \
    struct Fmin_NM_Element *p_search, *pe_min, *pe_max, *pe_max_less; \
    RB_Tree_Node *node_min, *node_max, *node_max_less; \
    num_type *accept_x,accept_y; \
    int flag_accept=0; /* flag_accept may not only be 0 or 1 */ \
    for (i_step=0; i_step<n_step; i_step++){ \
        node_max = RB_Tree_Maximum(&T, T.root); \
        node_min = RB_Tree_Minimum(&T, T.root); \
        RB_Tree_Delete(&T, node_max); \
        node_max_less = RB_Tree_Maximum(&T, T.root); \
        pe_max = (struct Fmin_NM_Element*)(node_max->key); \
        pe_min = (struct Fmin_NM_Element*)(node_min->key); \
        pe_max_less = (struct Fmin_NM_Element*)(node_max_less->key); \
        id_max = pe_max->id; \
        /* cal x_mean */ \
        /* set 0 */ \
        for (i_param=0; i_param<n_param; i_param++){ \
            x_sum[i_param] = 0; \
        } \
        for (ix=0,pnme=nm.element; ix<nx; ix++,pnme++){ \
            if (pnme == pe_max) continue; \
            for (i_param=0; i_param<n_param; i_param++){ \
                x_sum[i_param] += pnme->x[i_param]; \
            } \
        } \
        for (i_param=0; i_param<n_param; i_param++){ \
            x_mean[i_param] = x_sum[i_param] / (num_type)n_param; \
        } \
        /* x_r = 2*x_mean-xh */ \
        for (i_param=0; i_param<n_param; i_param++){ \
            xr[i_param] = 2 * x_mean[i_param] - pe_max->x[i_param]; \
        } \
        yr = func(xr, other_param); \
        /* y[n] is the max of y, y[0] is the min of y, in the following comments */ \
        flag_accept = 0; /* marks whether a new x is accepted */ \
        if (yr < pe_max_less->y){ /* yr < y[n-1] */ \
            if (yr < pe_min->y){ /* yr < y[0] */ \
                /* xe = 3*x_mean-2*xh */ \
                for (i_param=0; i_param<n_param; i_param++){ \
                    xe[i_param] = 3 * x_mean[i_param] - 2 * pe_max->x[i_param]; \
                } \
                ye = func(xe, other_param); \
                if (ye < yr){ /* ye < yr < y[0] */ \
                    /* accept xe */ \
                    accept_x = xe; \
                    accept_y = ye; \
                    flag_accept = 1; \
                }else{ \
                    /* accept xr */ \
                    accept_x = xr; \
                    accept_y = yr; \
                    flag_accept = 2; \
                } \
            }else{ /* y[0] <= yr < y[n-1] */ /* don't know where yr exactly is now */ \
                /* accept xr */ \
                accept_x = xr; \
                accept_y = yr; \
                flag_accept = 3; \
            } \
        }else{ /* yr >= y[n-1] */ \
            if (yr < pe_max->y){ /* yr < y[n] */ \
                /* xoc = 1.5*x_mean-0.5*xh */ \
                for (i_param=0; i_param<n_param; i_param++){ \
                    xoc[i_param] = 1.5 * x_mean[i_param] - .5 * pe_max->x[i_param]; \
                } \
                yoc=func(xoc,other_param); \
                if (yoc<yr){ /* don't know where yoc exactly is */ \
                    /* accept xoc */ \
                    accept_x = xoc; \
                    accept_y = yoc; \
                    flag_accept = 4; \
                }else{ \
                    /* shrink to x[0]: x[i]=(x[0]+x[i])/2 */ \
                    flag_accept = 0; \
                    ; \
                } \
            }else{ \
                /* xic = (x_mean+xh)/2 */ \
                for (i_param=0; i_param<n_param; i_param++){ \
                    xic[i_param] = (x_mean[i_param] + pe_max->x[i_param]) * .5; \
                } \
                yic = func(xic, other_param); \
                if (yic < pe_max->y){ /* yic < y[n] */ \
                    /* accept xic */ \
                    accept_x = xic; \
                    accept_y = yic; \
                    flag_accept = 5; \
                }else{ \
                    /* shrink to x[0]: x[i]=(x[0]+x[i])/2 */ \
                    ; \
                } \
            } \
        } \
        /* if accept a new x, then update; else shrink x[1:n] to x[0] */ \
        /* Notice: if a new x is accepted, then y < y[n] according to the code above */ \
        if (flag_accept){ \
            RB_Tree_Delete(&T, nodes+id_max); \
            memcpy(pe_max->x, accept_x, size_cpy); \
            pe_max->y = accept_y; \
            RB_Tree_Insert(&T, node_max, Fmin_NM_Compare, NULL, RB_TREE_DISABLE_SAME_KEY); \
        }else{ /* shrink to x[0]: x[i]=(x[0]+x[i])/2. */ \
            for (ix=0,p_search=nm.element; ix<nx; ix++,p_search++){ \
                if (p_search == pe_min) continue; \
                for (i_param=0; i_param<n_param; i_param++){ \
                    p_search->x[i_param] = (pe_min->x[i_param] + p_search->x[i_param])*.5; \
                } \
                p_search->y=func(p_search->x,other_param); \
            } \
            /* re-allocate the Heap */ \
            RB_Tree_Init(&T); \
            for (ix=0; ix<nx; ix++){ \
                nodes[ix].key = (void*)&nm.element[ix]; \
                RB_Tree_Insert(&T, nodes+ix, Fmin_NM_Compare, NULL, RB_TREE_DISABLE_SAME_KEY); \
            } \
        } \
        if (print_step && print_next_step==i_step){ \
            print_where = 0; \
            print_temp_len = swprintf(print_info + print_where, FMIN_NM_PRINT_LEN, L"Step %llu: accept new value %d\n",i_step,flag_accept); \
            print_where += print_temp_len; \
            node_max = RB_Tree_Maximum(&T, T.root); \
            node_min = RB_Tree_Minimum(&T, T.root); \
            for (ix=0,pnme=nm.element; ix<nx; ix++,pnme++){ \
                /* print x */ \
                for (i_param=0; i_param<n_param; i_param++){ \
                    print_temp_len = swprintf(print_info + print_where, FMIN_NM_PRINT_LEN, L"%e\t", (num_print_type)pnme->x[i_param]); \
                    print_where += print_temp_len; \
                } \
                /* print y */ \
                if (pnme == (struct Fmin_NM_Element*)(node_min->key)){ \
                    print_temp_len = swprintf(print_info + print_where, FMIN_NM_PRINT_LEN, L"| %e MIN\n", (num_print_type)pnme->y); \
                }else if (pnme == (struct Fmin_NM_Element*)(node_max->key)){ \
                    print_temp_len = swprintf(print_info + print_where, FMIN_NM_PRINT_LEN, L"| %e MAX\n", (num_print_type)pnme->y); \
                }else{ \
                    print_temp_len = swprintf(print_info + print_where, FMIN_NM_PRINT_LEN, L"| %e\n", (num_print_type)pnme->y); \
                } \
                print_where += print_temp_len; \
            } \
            print_info[print_where] = 0; \
            Madd_Print(print_info); \
            print_next_step += print_step; \
        } \
    } \
    /* free */ \
    free(space); \
    if (print_step){ \
        free(print_info); \
    } \
    return flag_accept; \
} \

/* uint64_t & double */
FMIN_NM_PARAM(double, Fmin_NM_Element, Fmin_NM_Param)

static char Fmin_NM_Compare(void *key1_, void *key2_, void *other_param)
FMIN_NM_COMPARE__ALGORITHM(double, Fmin_NM_Element, L"Fmin_NM_Compare: encountered same key.")

int Fmin_NM(uint64_t n_param, double **start,
            double func(double *params,void *other_param), void *other_param,
            uint64_t n_step, uint64_t print_start, uint64_t print_step)
FMIN_NM__ALGORITHM(double, double, Fmin_NM_Element, Fmin_NM_Param, Fmin_NM_Compare)

/* uint64_t & float */
FMIN_NM_PARAM(float, Fmin_NM_Element_f32, Fmin_NM_Param_f32)

static char Fmin_NM_Compare_f32(void *key1_, void *key2_, void *other_param)
FMIN_NM_COMPARE__ALGORITHM(float, Fmin_NM_Element_f32, L"Fmin_NM_Compare_f32: encountered same key.")

int Fmin_NM_f32(uint64_t n_param, float **start,
                float func(float *params,void *other_param), void *other_param,
                uint64_t n_step, uint64_t print_start, uint64_t print_step)
FMIN_NM__ALGORITHM(float, float, Fmin_NM_Element_f32, Fmin_NM_Param_f32, Fmin_NM_Compare_f32)

/* uint64_t & long double */
FMIN_NM_PARAM(long double, Fmin_NM_Element_fl, Fmin_NM_Param_fl)

static char Fmin_NM_Compare_fl(void *key1_, void *key2_, void *other_param)
FMIN_NM_COMPARE__ALGORITHM(long double, Fmin_NM_Element_fl, L"Fmin_NM_Compare_fl: encountered same key.")

int Fmin_NM_fl(uint64_t n_param, long double **start,
               long double func(long double *params,void *other_param), void *other_param,
               uint64_t n_step, uint64_t print_start, uint64_t print_step)
FMIN_NM__ALGORITHM(long double, double, Fmin_NM_Element_fl, Fmin_NM_Param_fl, Fmin_NM_Compare_fl)

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
FMIN_NM_PARAM(__float128, Fmin_NM_Element_f128, Fmin_NM_Param_f128)

static char Fmin_NM_Compare_f128(void *key1_, void *key2_, void *other_param)
FMIN_NM_COMPARE__ALGORITHM(__float128, Fmin_NM_Element_f128, L"Fmin_NM_Compare_f128: encountered same key.")

int Fmin_NM_f128(uint64_t n_param, __float128 **start,
                 __float128 func(__float128 *params,void *other_param), void *other_param,
                 uint64_t n_step, uint64_t print_start, uint64_t print_step)
FMIN_NM__ALGORITHM(__float128, double, Fmin_NM_Element_f128, Fmin_NM_Param_f128, Fmin_NM_Compare_f128)

#endif