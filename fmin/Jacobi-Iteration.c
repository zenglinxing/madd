/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/Jacobi-Iteration.c
*/
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#include"fmin.h"
#include"../basic/basic.h"
#include"../sort/sort.h"

static char jacobi_compare(void *a_, void *b_, void *other_param)
{
    Fmin_Jacobi_Iteration_Param *a = (Fmin_Jacobi_Iteration_Param*)a_, *b = (Fmin_Jacobi_Iteration_Param*)b_;
    if (a->x < b->x) return MADD_LESS;
    else if (a->x > b->x) return MADD_GREATER;
    else if (a->y < b->y) return MADD_LESS;
    else if (a->y > b->y) return MADD_GREATER;
    else return MADD_SAME;
}

bool Fmin_Jacobi_Iteration_Scarse(uint64_t n, uint64_t n_param, Fmin_Jacobi_Iteration_Param *param, double *b, double *solution, uint64_t n_step)
{
    if (n == 0){
        return false;
    }
    if (n_param == 0){
        return false;
    }
    if (param == NULL){
        return false;
    }
    if (b == NULL){
        return false;
    }
    if (solution == NULL){
        return false;
    }

    double *diag = (Fmin_Jacobi_Iteration_Param*)malloc((uint64_t)n*2*sizeof(double)), *x_new = diag + n;
    if (diag == NULL){
        return false;
    }
    Sort(n_param, sizeof(Fmin_Jacobi_Iteration_Param), param, jacobi_compare, NULL);
    for (uint64_t i=0; i<n; i++){
        diag[i] = Inf;
    }
    for (uint64_t i=0; i<n_param; i++){
        Fmin_Jacobi_Iteration_Param *p = param + i;
        if (p->x == p->y){
            diag[p->x] = 1 / p->value;
        }
    }

    size_t size_cpy = (uint64_t)n * sizeof(double);
    for (uint64_t i_step=0; i_step<n_step; i_step++){
        Fmin_Jacobi_Iteration_Param *p = param;
        uint64_t i_param = 0;
        while (p->x == p->y){
            p ++;
            i_param ++;
            if (i_param == n_param) break;
        }
        for (uint64_t ix=0; ix<n; ix++){
            double sum = 0;
            if (p->x > ix) continue;
            while (i_param < n_param && p->x == ix){
                if (p->y == ix){
                    p ++;
                    i_param ++;
                    continue;
                }
                sum += p->value * solution[p->y];
                p ++;
                i_param ++;
            }
            x_new[ix] = diag[ix] * (b[ix] - sum);
        }
        memcpy(solution, x_new, size_cpy);
    }

    free(diag);
    return true;
}