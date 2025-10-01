/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/eigen.c
*/
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#include<lapacke.h>

#include"linalg.h"
#include"../basic/basic.h"

bool Eigen(int n, double *matrix, Cnum *eigenvalue,
           bool flag_left, Cnum *eigenvector_left,
           bool flag_right, Cnum *eigenvector_right)
{
    if (n == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (matrix == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (eigenvalue == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: eigenvalue is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (flag_left && eigenvector_left == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_left is true, but eigenvector_left is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (flag_right && eigenvector_right == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_right is true, but eigenvector_right is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    double *wr = (double*)eigenvalue, *wi = (double*)wr + n;
    char jobvl = (flag_left) ? 'V' : 'N', jobvr = (flag_right) ? 'V' : 'N';
    uint64_t nn = (uint64_t)n * n;
    size_t size_nn = nn*sizeof(double);
    double *vl = (double*)malloc(2*size_nn), *vr = vl + nn;
    if (vl == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for vl & vr.", __func__, 2*size_nn);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    /* eigen */
    lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, jobvl, jobvr, n,
                                    matrix, n,
                                    wr, wi,
                                    vl, n,
                                    vr, n);
    if (info != 0){
        free(vl);
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        if (info < 0){
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value.", __func__, -info);
        }else{
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the QR algorithm failed to compute all the eigenvalues, and no eigenvectors have been computed; elements %d+1:N of WR and WI contain eigenvalues which have converged.", __func__, info);
        }
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    Matrix_Transpose(2, n, (double*)eigenvalue);

    uint64_t i, j;
    for (i=0; i<n; i++){
        Cnum *le = eigenvector_left + i, *re = eigenvector_right + i;
        double *lv = vl + i, *rv = vr + i;
        if (eigenvalue[i].imag == 0){
            for (j=0; j<n; j++,lv+=n,rv+=n,le+=n,re+=n){
                le->real = *lv;
                re->real = *rv;
                le->imag = re->imag = 0;
            }
        }else{
            for (j=0; j<n; j++,lv+=n,rv+=n,le+=n,re+=n){
                le->real = le[1].real = *lv;
                re->real = re[1].real = *rv;
                le->imag = lv[1];
                le[1].imag = -lv[1];
                re->imag = rv[1];
                re[1].imag = -rv[1];
            }
            i ++;
            continue;
        }
    }

    free(vl);
    return true;
}