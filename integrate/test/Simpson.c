/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

double func(double x,void *other_param)
{
    return x+x*x;
}

int main(int argc,char *argv[])
{
    double xmin=0,xmax=20,res=Integrate_Simpson(func,xmin,xmax,2000,NULL);
    printf("res\t%f\n",res);
    return 0;
}
