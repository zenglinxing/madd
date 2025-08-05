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
#include<stdio.h>
#include"../data_struct/RB_Tree.c"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define FMIN_NM_COMPARE_FAIL 3

#define FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element,Fmin_NM_Param,num_type) \
struct Fmin_NM_Element{ \
    uint64_t id; \
    num_type y,*x; \
}; \
typedef struct{ \
    uint64_t np,nx; \
    struct Fmin_NM_Element *element; \
} Fmin_NM_Param; \

#define FMIN_NM_COMPARE_MAX__ALGORITHM(Fmin_NM_Element) \
{ \
    if (other_param!=NULL) return FMIN_NM_COMPARE_FAIL; \
    register struct Fmin_NM_Element *key1=(struct Fmin_NM_Element*)key1_, *key2=(struct Fmin_NM_Element*)key2_; \
    if (key1->y > key2->y) return FIBONACCI_HEAP_LESS; \
    else if (key1->y < key2->y) return FIBONACCI_HEAP_GREATER; \
    else return FIBONACCI_HEAP_SAME; \
} \

#define FMIN_NM_COMPARE_MIN__ALGORITHM(Fmin_NM_Element) \
{ \
    if (other_param!=NULL) return FMIN_NM_COMPARE_FAIL; \
    register struct Fmin_NM_Element *key1=(struct Fmin_NM_Element*)key1_, *key2=(struct Fmin_NM_Element*)key2_; \
    if (key1->y < key2->y) return FIBONACCI_HEAP_LESS; \
    else if (key1->y > key2->y) return FIBONACCI_HEAP_GREATER; \
    else return FIBONACCI_HEAP_SAME; \
} \

/*
n_param is the length of x, nx in the function is n_param+1 (how many x in **start)
*/
#define FMIN_NM__ALGORITHM(num_type,Fmin_NM_Element,Fmin_NM_Param,Fmin_NM_Compare_Min,Fmin_NM_Compare_Max,num_type_print) \
{ \
    size_t size_cpy=sizeof(num_type)*n_param /* n_param *//*, Size_cpy=size_cpy*n_param+size_cpy*/ /* n_cpy*(n_param+1) */; \
    register uint64_t ix; \
    register uint64_t i_param; \
    uint64_t nx=(uint64_t)n_param+1; \
    void *space=(void*)malloc(nx*(sizeof(struct Fmin_NM_Element)+2*sizeof(Fibonacci_Heap_Node)+sizeof(struct Fmin_NM_Element))+6*size_cpy); \
    /* NM parameter */ \
    register struct Fmin_NM_Element *pnme; \
    struct Fmin_NM_Element *sort_temp=(struct Fmin_NM_Element*)space; /* sort_temp nm.element */ \
    register Fmin_NM_Param nm/*={.np=n_param, .nx=nx}*/; \
    nm.np=n_param; \
    nm.nx=nx; \
    Fibonacci_Heap Hmax=Fibonacci_Heap_Make(), Hmin=Fibonacci_Heap_Make(); \
    Fibonacci_Heap_Node *nodes_min=(Fibonacci_Heap_Node*)(sort_temp+nx), *nodes_max=nodes_min+nx; \
    /* sort the x according to y */ \
    nm.element=(struct Fmin_NM_Element*)(nodes_max+nx); \
    /* initialize NM parameter */ \
    for (ix=0,pnme=nm.element /*nm.element*/; ix<nx; ix++,pnme++){ \
        /* pnme=nm.element+ix; */ \
        pnme->id = ix; \
        pnme->x=start[ix]; \
        pnme->y=func(pnme->x,other_param); /* function value */ \
        /* Fibonacci nodes */ \
        nodes_min[ix].key = nodes_max[ix].key = (void*)pnme; \
        Fibonacci_Heap_Insert(&Hmin, nodes_min+ix, Fmin_NM_Compare_Min, NULL); \
        Fibonacci_Heap_Insert(&Hmax, nodes_max+ix, Fmin_NM_Compare_Max, NULL); \
    } \
    /* x mean & x sum */ \
    num_type *x_list=(num_type*)(nm.element+nx); \
    register num_type *x_mean=x_list, *x_sum=x_list+n_param, *xr=x_list+2*n_param /* 2*x_mean+x_h */,yr; /* xh is nm.max->x */ \
    register num_type *xe=x_list+3*n_param /* 3*x_mean-2*xh */,ye, *xoc=x_list+4*n_param /* 1.5*x_mean-0.5*xh */,yoc; \
    register num_type *xic=x_list+5*n_param,yic /* 1.5*x_mean+0.5*xh */; \
    /* Search */ \
    register uint64_t i_step,print_next_step=print_start, id_max; \
    register struct Fmin_NM_Element *p_search, *pe_min, *pe_max, *pe_max_less; \
    register Fibonacci_Heap_Node *node_min, *node_max, *node_max_less; \
    register num_type *accept_x,accept_y; \
    register int8_t flag_accept=0; /* flag_accept may not only be 0 or 1 */ \
    num_type n_param_d=n_param; \
    for (i_step=0; i_step<n_step; i_step++){ \
        node_max = Fibonacci_Heap_Extract_Min(&Hmax, Fmin_NM_Compare_Max, NULL); \
        node_min = Hmin.min; \
        node_max_less = Hmax.min; \
        pe_max = (struct Fmin_NM_Element*)(node_max->key); \
        pe_min = (struct Fmin_NM_Element*)(node_min->key); \
        pe_max_less = (struct Fmin_NM_Element*)(node_max_less->key); \
        id_max = pe_max->id; \
        /* cal x_mean */ \
        memset(x_sum,0,size_cpy); \
        for (ix=0,pnme=nm.element; ix<nx; ix++,pnme++){ \
            if (pnme == pe_max) continue; \
            for (i_param=0; i_param<n_param; i_param++){ \
                x_sum[i_param]+=pnme->x[i_param]; \
            } \
        } \
        for (i_param=0; i_param<n_param; i_param++){ \
            x_mean[i_param]=x_sum[i_param]/n_param_d; \
        } \
        /* x_r = 2*x_mean-xh */ \
        for (i_param=0; i_param<n_param; i_param++){ \
            xr[i_param]=2*x_mean[i_param]-pe_max->x[i_param]; \
        } \
        yr=func(xr,other_param); \
        /* y[n] is the max of y, y[0] is the min of y, in the following comments */ \
        flag_accept=0; /* marks whether a new x is accepted */ \
        if (yr < pe_max_less->y){ /* yr < y[n-1] */ \
            if (yr < pe_min->y){ /* yr < y[0] */ \
                /* xe = 3*x_mean-2*xh */ \
                for (i_param=0; i_param<n_param; i_param++){ \
                    xe[i_param]=3*x_mean[i_param]-2*pe_max->x[i_param]; \
                } \
                ye=func(xe,other_param); \
                if (ye < yr){ /* ye < yr < y[0] */ \
                    /* accept xe */ \
                    accept_x=xe; \
                    accept_y=ye; \
                    flag_accept=1; \
                }else{ \
                    /* accept xr */ \
                    accept_x=xr; \
                    accept_y=yr; \
                    flag_accept=2; \
                } \
            }else{ /* y[0] <= yr < y[n-1] */ /* don't know where yr exactly is now */ \
                /* accept xr */ \
                accept_x=xr; \
                accept_y=yr; \
                flag_accept=3; \
            } \
        }else{ /* yr >= y[n-1] */ \
            if (yr < pe_max->y){ /* yr < y[n] */ \
                /* xoc = 1.5*x_mean-0.5*xh */ \
                for (i_param=0; i_param<n_param; i_param++){ \
                    xoc[i_param]=1.5*x_mean[i_param]-.5*pe_max->x[i_param]; \
                } \
                yoc=func(xoc,other_param); \
                if (yoc<yr){ /* don't know where yoc exactly is */ \
                    /* accept xoc */ \
                    accept_x=xoc; \
                    accept_y=yoc; \
                    flag_accept=4; \
                }else{ \
                    /* shrink to x[0]: x[i]=(x[0]+x[i])/2 */ \
                    ; \
                } \
            }else{ \
                /* xic = (x_mean+xh)/2 */ \
                for (i_param=0; i_param<n_param; i_param++){ \
                    xic[i_param]=(x_mean[i_param]+pe_max->x[i_param])*.5; \
                } \
                yic=func(xic,other_param); \
                if (yic < pe_max->y){ /* yic < y[n] */ \
                    /* accept xic */ \
                    accept_x=xic; \
                    accept_y=yic; \
                    flag_accept=5; \
                }else{ \
                    /* shrink to x[0]: x[i]=(x[0]+x[i])/2 */ \
                    ; \
                } \
            } \
        } \
        /* if accept a new x, then update; else shrink x[1:n] to x[0] */ \
        /* Notice: if a new x is accepted, then y < y[n] according to the code above */ \
        if (flag_accept){ \
            Fibonacci_Heap_Delete(&Hmin, nodes_min+id_max, Fmin_NM_Compare_Min, NULL); \
            memcpy(pe_max->x, accept_x, size_cpy); \
            pe_max->y=accept_y; \
            Fibonacci_Heap_Insert(&Hmax, node_max, Fmin_NM_Compare_Max, NULL); \
            Fibonacci_Heap_Insert(&Hmin, nodes_min+id_max, Fmin_NM_Compare_Min, NULL); \
        }else{ /* shrink to x[0]: x[i]=(x[0]+x[i])/2. */ \
            for (ix=0,p_search=nm.element; ix<nx; ix++,p_search++){ \
                if (p_search == pe_min) continue; \
                for (i_param=0; i_param<n_param; i_param++){ \
                    p_search->x[i_param] = (pe_min->x[i_param] + p_search->x[i_param])*.5; \
                } \
                p_search->y=func(p_search->x,other_param); \
            } \
            /* re-allocate the Heap */ \
            Hmin = Hmax = Fibonacci_Heap_Make(); \
            for (ix=0; ix<nx; ix++){ \
                Fibonacci_Heap_Insert(&Hmin, nodes_min+ix, Fmin_NM_Compare_Min, NULL); \
                Fibonacci_Heap_Insert(&Hmax, nodes_max+ix, Fmin_NM_Compare_Max, NULL); \
            } \
        } \
        if (print_step && print_next_step==i_step){ \
            printf("Step %llu: accept new value %d\n",i_step,flag_accept); \
            for (ix=0,pnme=nm.element; ix<nx; ix++,pnme++){ \
                for (i_param=0; i_param<n_param; i_param++){ \
                    printf("%e\t",(num_type_print)pnme->x[i_param]); \
                } \
                if (pnme == (struct Fmin_NM_Element*)(Hmin.min->key)){ \
                    printf("%e MIN\n",(num_type_print)pnme->y); \
                }else if (pnme == (struct Fmin_NM_Element*)(Hmax.min->key)){ \
                    printf("%e MAX\n",(num_type_print)pnme->y); \
                }else{ \
                    printf("%e\n",(num_type_print)pnme->y); \
                } \
            } \
            print_next_step+=print_step; \
        } \
    } \
    /* free */ \
    free(space); \
    return flag_accept; \
} \

/* uint32_t & double */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element,Fmin_NM_Param,double)

char Fmin_NM_Compare_Max(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MAX__ALGORITHM(Fmin_NM_Element)

char Fmin_NM_Compare_Min(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MIN__ALGORITHM(Fmin_NM_Element)

uint8_t Fmin_NM(uint32_t n_param,double func(double *params,void *other_param),double **start,void *other_param,uint32_t n_step,uint32_t print_start,uint32_t print_step)
FMIN_NM__ALGORITHM(double,Fmin_NM_Element,Fmin_NM_Param,Fmin_NM_Compare_Min,Fmin_NM_Compare_Max,double)

/* uint32_t & float */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element_F,Fmin_NM_Param_F,float)

char Fmin_NM_Compare_Max_F(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MAX__ALGORITHM(Fmin_NM_Element_F)

char Fmin_NM_Compare_Min_F(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MIN__ALGORITHM(Fmin_NM_Element_F)

uint8_t Fmin_NM_F(uint32_t n_param,float func(float *params,void *other_param),float **start,void *other_param,uint32_t n_step,uint32_t print_start,uint32_t print_step)
FMIN_NM__ALGORITHM(float,Fmin_NM_Element_F,Fmin_NM_Param_F,Fmin_NM_Compare_Min_F,Fmin_NM_Compare_Max_F,float)

/* uint64_t & double */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element_L,Fmin_NM_Param_L,double)

char Fmin_NM_Compare_Max_L(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MAX__ALGORITHM(Fmin_NM_Element_L)

char Fmin_NM_Compare_Min_L(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MIN__ALGORITHM(Fmin_NM_Element_L)

uint8_t Fmin_NM_L(uint64_t n_param,double func(double *params,void *other_param),double **start,void *other_param,uint64_t n_step,uint64_t print_start,uint64_t print_step)
FMIN_NM__ALGORITHM(double,Fmin_NM_Element_L,Fmin_NM_Param_L,Fmin_NM_Compare_Min_L,Fmin_NM_Compare_Max_L,double)

/* uint64_t & long double */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element_LD,Fmin_NM_Param_LD,long double)

char Fmin_NM_Compare_Max_LD(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MAX__ALGORITHM(Fmin_NM_Element_LD)

char Fmin_NM_Compare_Min_LD(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MIN__ALGORITHM(Fmin_NM_Element_LD)

uint8_t Fmin_NM_LD(uint64_t n_param,long double func(long double *params,void *other_param),long double **start,void *other_param,uint64_t n_step,uint64_t print_start,uint64_t print_step)
FMIN_NM__ALGORITHM(long double,Fmin_NM_Element_LD,Fmin_NM_Param_LD,Fmin_NM_Compare_Min_LD,Fmin_NM_Compare_Max_LD,double)

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element_QD,Fmin_NM_Param_QD,__float128)

char Fmin_NM_Compare_Max_QD(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MAX__ALGORITHM(Fmin_NM_Element_QD)

char Fmin_NM_Compare_Min_QD(void *key1_,void *key2_,void *other_param)
FMIN_NM_COMPARE_MIN__ALGORITHM(Fmin_NM_Element_QD)

uint8_t Fmin_NM_QD(uint64_t n_param,__float128 func(__float128 *params,void *other_param),__float128 **start,void *other_param,uint64_t n_step,uint64_t print_start,uint64_t print_step)
FMIN_NM__ALGORITHM(__float128,Fmin_NM_Element_QD,Fmin_NM_Param_QD,Fmin_NM_Compare_Min_QD,Fmin_NM_Compare_Max_QD,double)
#endif /* ENABLE_QUADPRECISION */

#endif /* _FMIN_NM_C */
