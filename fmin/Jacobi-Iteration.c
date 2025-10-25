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
#include"../linalg/linalg.h"
#include"../sort/sort.h"

static char jacobi_compare(void *a_, void *b_, void *other_param)
{
    Sparse_Matrix_COO_Unit *a = (Sparse_Matrix_COO_Unit*)a_, *b = (Sparse_Matrix_COO_Unit*)b_;
    if (a->x < b->x) return MADD_LESS;
    else if (a->x > b->x) return MADD_GREATER;
    else if (a->y < b->y) return MADD_LESS;
    else if (a->y > b->y) return MADD_GREATER;
    else return MADD_SAME;
}

bool Fmin_Jacobi_Iteration_Sparse(Sparse_Matrix_COO *param, double *b, double *solution, uint64_t n_step)
{
    if (param->dim == 0){
        return false;
    }
    if (param->n_unit == 0){
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

    double *diag = (double*)malloc((uint64_t)param->dim*2*sizeof(double)), *x_new = diag + param->dim;
    if (diag == NULL){
        return false;
    }
    Sort(param->n_unit, sizeof(Sparse_Matrix_COO_Unit), param, jacobi_compare, NULL);
    for (uint64_t i=0; i<param->dim; i++){
        diag[i] = Inf;
    }
    for (uint64_t i=0; i<param->n_unit; i++){
        Sparse_Matrix_COO_Unit *p = param->unit + i;
        if (p->x == p->y){
            diag[p->x] = 1 / p->value;
        }
    }

    size_t size_cpy = (uint64_t)param->dim * sizeof(double);
    for (uint64_t i_step=0; i_step<n_step; i_step++){
        Sparse_Matrix_COO_Unit *p = param->unit;
        uint64_t i_unit = 0;
        while (i_unit < param->n_unit && p->x == p->y){
            p ++;
            i_unit ++;
        }
        for (uint64_t ix=0; ix<param->dim; ix++){
            double sum = 0;
            if (p->x > ix) continue;
            while (i_unit < param->n_unit && p->x == ix){
                if (p->y == ix){
                    p ++;
                    i_param ++;
                    continue;
                }
                sum += p->value * solution[p->y];
                p ++;
                i_unit ++;
            }
            x_new[ix] = diag[ix] * (b[ix] - sum);
        }
        memcpy(solution, x_new, size_cpy);
    }

    free(diag);
    return true;
}