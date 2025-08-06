/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/fmin-NM.c
Nelder-Mead Search
*/
#ifndef _FMIN_NM_C
#define _FMIN_NM_C

#include<stdlib.h>
#include<stdint.h>
#include<string.h>
#include<stdio.h>
#include<wchar.h>
#include"../basic/basic.h"
#include"../data_struct/RB_tree.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define FMIN_NM_FAIL 100
#define FMIN_NM_COMPARE_FAIL 3
#define FMIN_NM_PRINT_LEN 100

/* uint32_t & double */
struct Fmin_NM_Element{ 
    uint64_t id; 
    double y,*x; 
}; 
typedef struct{ 
    uint64_t np, nx; 
    struct Fmin_NM_Element *element; 
} Fmin_NM_Param; 

static char Fmin_NM_Compare(void *key1_, void *key2_, void *other_param)
{
    struct Fmin_NM_Element *key1=key1_, *key2=key2_;
    if (key1->y < key2->y)      return MADD_LESS;
    else if (key1->y > key2->y) return MADD_GREATER;
    else if (key1->y == key2->y){
        if (key1->id < key2->id)        return MADD_LESS;
        else if (key1->id > key2->id)   return MADD_GREATER;
        else{
            Madd_Error_Add(MADD_ERROR, L"Fmin_NM_Compare: encountered same key.");
            return MADD_SAME;
        }
    }
}

int Fmin_NM(uint32_t n_param, double **start,
            double func(double *params,void *other_param), void *other_param,
            uint32_t n_step, uint32_t print_start, uint32_t print_step)
{ 
    size_t size_cpy=sizeof(double)*n_param /* n_param *//*, Size_cpy=size_cpy*n_param+size_cpy*/ /* n_cpy*(n_param+1) */;
    uint64_t ix, i_param, nx=(uint64_t)n_param+1;
    size_t total_size = nx * (sizeof(struct Fmin_NM_Element) + sizeof(RB_Tree_Node)) 
                   + 6 * size_cpy;
    void *space=(void*)malloc(total_size);
    if (space == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Fmin_NM: unable to allocate mem %llu bytes.", total_size);
        Madd_Error_Add(MADD_ERROR, error_info);
        return FMIN_NM_FAIL;
    }

    /* NM parameter */ 
    struct Fmin_NM_Element *pnme;
    Fmin_NM_Param nm={.np=n_param, .nx=nx};
    RB_Tree T;
    RB_Tree_Init(&T);
    RB_Tree_Node *nodes=(RB_Tree_Node*)space;

    /* sort the x according to y */ 
    nm.element = (struct Fmin_NM_Element*)(nodes + nx);

    /* initialize NM parameter */ 
    for (ix=0,pnme=nm.element /*nm.element*/; ix<nx; ix++,pnme++){ 
        /* pnme=nm.element+ix; */ 
        pnme->id = ix; 
        pnme->x = start[ix]; 
        pnme->y = func(pnme->x,other_param); /* function value */ 
        /* RB Tree nodes */ 
        nodes[ix].key = (void*)pnme; 
        RB_Tree_Insert(&T, nodes+ix, Fmin_NM_Compare, NULL, RB_TREE_DISABLE_SAME_KEY);
    } 
    /* x mean & x sum */ 
    double *x_list=(double*)(nm.element+nx); 
    double *x_mean=x_list, *x_sum=x_list+n_param, *xr=x_list+2*n_param /* 2*x_mean+x_h */, yr; /* xh is nm.max->x */ 
    double *xe=x_list+3*n_param /* 3*x_mean-2*xh */, ye, *xoc=x_list+4*n_param /* 1.5*x_mean-0.5*xh */, yoc; 
    double *xic=x_list+5*n_param, yic /* 1.5*x_mean+0.5*xh */; 
    /* Search */ 
    uint64_t i_step, print_next_step=print_start, id_max; 
    struct Fmin_NM_Element *p_search, *pe_min, *pe_max, *pe_max_less; 
    RB_Tree_Node *node_min, *node_max, *node_max_less; 
    double *accept_x,accept_y; 
    int flag_accept=0; /* flag_accept may not only be 0 or 1 */
    for (i_step=0; i_step<n_step; i_step++){
        node_max = RB_Tree_Maximum(&T, T.root); 
        node_min = RB_Tree_Minimum(&T, T.root);
        RB_Tree_Delete(&T, node_max);
        node_max_less = RB_Tree_Maximum(&T, T.root);
        pe_max = (struct Fmin_NM_Element*)(node_max->key); 
        pe_min = (struct Fmin_NM_Element*)(node_min->key); 
        pe_max_less = (struct Fmin_NM_Element*)(node_max_less->key); 
        id_max = pe_max->id; 
        /* cal x_mean */ 
        memset(x_sum, 0, size_cpy); 
        for (ix=0,pnme=nm.element; ix<nx; ix++,pnme++){ 
            if (pnme == pe_max) continue; 
            for (i_param=0; i_param<n_param; i_param++){ 
                x_sum[i_param] += pnme->x[i_param]; 
            } 
        } 
        for (i_param=0; i_param<n_param; i_param++){ 
            x_mean[i_param] = x_sum[i_param] / (double)n_param;
        } 
        /* x_r = 2*x_mean-xh */ 
        for (i_param=0; i_param<n_param; i_param++){ 
            xr[i_param] = 2 * x_mean[i_param] - pe_max->x[i_param];
        } 
        yr = func(xr, other_param); 
        /* y[n] is the max of y, y[0] is the min of y, in the following comments */ 
        flag_accept = 0; /* marks whether a new x is accepted */ 
        if (yr < pe_max_less->y){ /* yr < y[n-1] */ 
            if (yr < pe_min->y){ /* yr < y[0] */ 
                /* xe = 3*x_mean-2*xh */ 
                for (i_param=0; i_param<n_param; i_param++){ 
                    xe[i_param] = 3 * x_mean[i_param] - 2 * pe_max->x[i_param]; 
                } 
                ye = func(xe, other_param);
                if (ye < yr){ /* ye < yr < y[0] */ 
                    /* accept xe */ 
                    accept_x = xe; 
                    accept_y = ye; 
                    flag_accept = 1; 
                }else{ 
                    /* accept xr */ 
                    accept_x = xr; 
                    accept_y = yr; 
                    flag_accept = 2; 
                } 
            }else{ /* y[0] <= yr < y[n-1] */ /* don't know where yr exactly is now */ 
                /* accept xr */ 
                accept_x = xr; 
                accept_y = yr; 
                flag_accept = 3; 
            } 
        }else{ /* yr >= y[n-1] */ 
            if (yr < pe_max->y){ /* yr < y[n] */ 
                /* xoc = 1.5*x_mean-0.5*xh */ 
                for (i_param=0; i_param<n_param; i_param++){ 
                    xoc[i_param] = 1.5 * x_mean[i_param] - .5 * pe_max->x[i_param]; 
                } 
                yoc=func(xoc,other_param); 
                if (yoc<yr){ /* don't know where yoc exactly is */ 
                    /* accept xoc */ 
                    accept_x = xoc; 
                    accept_y = yoc; 
                    flag_accept = 4; 
                }else{ 
                    /* shrink to x[0]: x[i]=(x[0]+x[i])/2 */ 
                    ; 
                } 
            }else{ 
                /* xic = (x_mean+xh)/2 */ 
                for (i_param=0; i_param<n_param; i_param++){ 
                    xic[i_param] = (x_mean[i_param] + pe_max->x[i_param]) * .5; 
                } 
                yic = func(xic, other_param); 
                if (yic < pe_max->y){ /* yic < y[n] */ 
                    /* accept xic */ 
                    accept_x = xic; 
                    accept_y = yic; 
                    flag_accept = 5; 
                }else{ 
                    /* shrink to x[0]: x[i]=(x[0]+x[i])/2 */ 
                    ; 
                } 
            } 
        } 
        /* if accept a new x, then update; else shrink x[1:n] to x[0] */ 
        /* Notice: if a new x is accepted, then y < y[n] according to the code above */
        if (flag_accept){ 
            RB_Tree_Delete(&T, nodes+id_max);
            memcpy(pe_max->x, accept_x, size_cpy); 
            pe_max->y = accept_y;
            RB_Tree_Insert(&T, node_max, Fmin_NM_Compare, NULL, RB_TREE_DISABLE_SAME_KEY);
        }else{ /* shrink to x[0]: x[i]=(x[0]+x[i])/2. */ 
            for (ix=0,p_search=nm.element; ix<nx; ix++,p_search++){ 
                if (p_search == pe_min) continue; 
                for (i_param=0; i_param<n_param; i_param++){ 
                    p_search->x[i_param] = (pe_min->x[i_param] + p_search->x[i_param])*.5; 
                } 
                p_search->y=func(p_search->x,other_param);
            }
            /* re-allocate the Heap */
            RB_Tree_Init(&T);
            for (ix=0; ix<nx; ix++){
                nodes[ix].key = (void*)&nm.element[ix];
                RB_Tree_Insert(&T, nodes+ix, Fmin_NM_Compare, NULL, RB_TREE_DISABLE_SAME_KEY);
            }
        }
        if (print_step && print_next_step==i_step){
            wchar_t print_info[FMIN_NM_PRINT_LEN];
            swprintf(print_info, FMIN_NM_PRINT_LEN, L"Step %llu: accept new value %d\n",i_step,flag_accept);
            Madd_Print(print_info);
            node_max = RB_Tree_Maximum(&T, T.root);
            node_min = RB_Tree_Minimum(&T, T.root);
            for (ix=0,pnme=nm.element; ix<nx; ix++,pnme++){
                /* print x */
                for (i_param=0; i_param<n_param; i_param++){
                    swprintf(print_info, FMIN_NM_PRINT_LEN, L"%e\t", (double)pnme->x[i_param]);
                    Madd_Print(print_info);
                }
                /* print y */
                if (pnme == (struct Fmin_NM_Element*)(node_min->key)){
                    swprintf(print_info, FMIN_NM_PRINT_LEN, L"| %e MIN\n", (double)pnme->y);
                    Madd_Print(print_info);
                }else if (pnme == (struct Fmin_NM_Element*)(node_max->key)){
                    swprintf(print_info, FMIN_NM_PRINT_LEN, L"| %e MAX\n", (double)pnme->y);
                    Madd_Print(print_info);
                }else{
                    swprintf(print_info, FMIN_NM_PRINT_LEN, L"| %e\n", (double)pnme->y);
                    Madd_Print(print_info);
                }
            }
            print_next_step += print_step;
        }
    }
    /* free */
    free(space);
    return flag_accept;
}


#endif /* _FMIN_NM_C */
