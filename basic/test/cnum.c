/* coding: utf-8 */
#include<stdio.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    double a1=1, a2=-2, b1=3, b2=4;
    Cnum a=Cnum_Value(a1, a2), a_same=Cnum_Value(a1, a2), b=Cnum_Value(b1, b2), c;
    printf("a = %f + %f i\n", a.real, a.imag);
    printf("b = %f + %f i\n", b.real, b.imag);
    c = Cnum_Add(a, b);
    printf("a+b = %f + %f i\n", c.real, c.imag);
    c = Cnum_Sub(a, b);
    printf("a-b = %f + %f i\n", c.real, c.imag);
    c = Cnum_Mul(a, b);
    printf("a*b = %f + %f i\n", c.real, c.imag);
    c = Cnum_Div(a, b);
    printf("a/b = %f + %f i\n", c.real, c.imag);
    printf("a=a? %d\n", Cnum_Eq(a, a_same));
    printf("a=b? %d\n", Cnum_Eq(a, b));
    c = Cnum_Div_Real(a, a2);
    printf("a/%f = %f + %f i\n", a2, c.real, c.imag);
    c = Real_Div_Cnum(a2, a);
    printf("%f/a = %f + %f i\n", a2, c.real, c.imag);
    double mod = Cnum_Mod2(a);
    printf("mod2 a = %f + %f i\n", c.real, c.imag);
    c = Cnum_Dot(a, b);
    printf("(a,b) = %f + %f i\n", c.real, c.imag);
    c = Cnum_Conj(a);
    printf("a* = %f + %f i\n", c.real, c.imag);
    double radius = Cnum_Radius(a);
    printf("r(a) = %f\n", radius);
    double angle = Cnum_Angle(a);
    printf("theta(a) = %f\n", angle);
    return 0;
}