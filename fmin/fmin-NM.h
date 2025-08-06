/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/fmin-NM.h
Nelder-Mead Search
*/
#ifndef _FMIN_NM_H
#define _FMIN_NM_H

#include<stdlib.h>
#include<stdint.h>

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define FMIN_NM_FAIL 100
#define FMIN_NM_COMPARE_FAIL 3

#define FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element, Fmin_NM_Param, num_type) \
struct Fmin_NM_Element{ \
    uint64_t id; \
    num_type y,*x; \
}; \
typedef struct{ \
    uint64_t np, nx; \
    struct Fmin_NM_Element *element; \
} Fmin_NM_Param; \

/*
n_param is the length of x, nx in the function is n_param+1 (how many x in **start)
*/
/* uint64_t & double */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element, Fmin_NM_Param, double)

char Fmin_NM_Compare_Max(void *key1_, void *key2_, void *other_param);
char Fmin_NM_Compare_Min(void *key1_, void *key2_, void *other_param);

uint8_t Fmin_NM(uint64_t n_param, double **start,
                double func(double *params,void *other_param), void *other_param,
                uint32_t n_step, uint32_t print_start, uint32_t print_step);

/* uint64_t & float */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element_f32, Fmin_NM_Param_f32, float)

char Fmin_NM_Compare_Max_f32(void *key1_, void *key2_, void *other_param);
char Fmin_NM_Compare_Min_f32(void *key1_, void *key2_, void *other_param);

uint8_t Fmin_NM_f32(uint64_t n_param, float **start,
                    float func(float *params,void *other_param), void *other_param,
                    uint64_t n_step, uint64_t print_start, uint64_t print_step);

/* uint64_t & long double */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element_fl, Fmin_NM_Param_fl, long double)

char Fmin_NM_Compare_Max_fl(void *key1_, void *key2_, void *other_param);
char Fmin_NM_Compare_Min_fl(void *key1_, void *key2_, void *other_param);

uint8_t Fmin_NM_fl(uint64_t n_param, long double **start,
                   long double func(long double *params,void *other_param), void *other_param,
                   uint64_t n_step, uint64_t print_start, uint64_t print_step);

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
FMIN_NM_ELEMENT_AND_PARAM(Fmin_NM_Element_f128, Fmin_NM_Param_f128, __float128)

char Fmin_NM_Compare_Max_f128(void *key1_, void *key2_, void *other_param);
char Fmin_NM_Compare_Min_f128(void *key1_, void *key2_, void *other_param);

uint8_t Fmin_NM_f128(uint64_t n_param, __float128 **start,
                     __float128 func(__float128 *params, void *other_param), void *other_param,
                     uint64_t n_step, uint64_t print_start, uint64_t print_step);
#endif /* ENABLE_QUADPRECISION */

#endif /* _FMIN_NM_H */
