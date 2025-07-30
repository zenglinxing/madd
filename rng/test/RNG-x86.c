/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

int main(int argc,char *argv[])
{
    int i,n=20;
    float rand;
    for (i=0; i<n; i++){
        rand = Rand_x86_f32();
        printf("%d\t%f\n", i, rand);
    }
    return 0;
}
