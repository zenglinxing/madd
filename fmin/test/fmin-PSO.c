/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

double func(double *x,void *other_param)
{
    return pow(x[0]-5,2.)+pow(x[1]+2,2.)+3;
}

int main(int argc,char *argv[])
{
    uint32_t n_bird=32,i,j;
    RNG_Param rng = RNG_Init(20, RNG_XOSHIRO256SS);
    double **x, x_init[2]={300, -300}, x_step[2]={1.e3, 1.e3,}, **velocity, v_init[2]={0., 0.}, v_step[2]={2.e2, 2.e2};
    double *x_list=(double*)malloc(n_bird*2*sizeof(double)),*v_list=(double*)malloc(n_bird*2*sizeof(double));
    x=(double**)malloc(n_bird*sizeof(double*));
    velocity=(double**)malloc(n_bird*sizeof(double*));
    for (i=0;i<n_bird;i++){
        x[i]=x_list+2*i;
        velocity[i]=v_list+2*i;
        for (j=0;j<2;j++){
            x[i][j]=x_init[j]+x_step[j]*(2*Rand(&rng)-1);
            velocity[i][j]=v_init[j]+v_step[j]*(2*Rand(&rng)-1);
        }
    }
    Fmin_PSO(2, n_bird, x,
             func, NULL,
             1e6, .9, .4, 2., 2., velocity, 1.e-1, &rng);
    for (i=0;i<n_bird;i++){
        for (j=0;j<2;j++){
            printf("%f\t",x[i][j]);
        }
        printf("%f\n",func(x[i],NULL));
    }
    free(x_list);
    free(v_list);
    free(x);
    free(velocity);
    return 0;
}
