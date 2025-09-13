Complex Number
===

Complex number is defined as `struct`, `Cnum` in Madd. The following `inline` functions are defined for `Cnum` type.

Additioin, subtraction, multiply, multiply with a real number, division, a complex number divided by a real number, a real number divided by a complex number.
```C
inline Cnum Cnum_Add(Cnum a, Cnum b);
inline Cnum Cnum_Sub(Cnum a, Cnum b);
inline Cnum Cnum_Mul(Cnum a, Cnum b);
inline Cnum Cnum_Mul_Real(Cnum a, double b);
inline Cnum Cnum_Div(Cnum a, Cnum b);
inline Cnum Cnum_Div_Real(Cnum a, double b);
inline Cnum Real_Div_Cnum(double a, Cnum b);
```

Value a complex number with its real and imagine part.
```C
inline Cnum Cnum_Value(double real, double imag);
```

Check if two complex numbers are equal.
```C
inline bool Cnum_Eq(Cnum a, Cnum b);
```

The square of the modulus of the complex number.
```C
inline double Cnum_Mod2(Cnum a);
```

$a'\times b$
```C
inline Cnum Cnum_Dot(Cnum a, Cnum b);
```

returns conjugate of $a$.
```C
inline Cnum Cnum_Conj(Cnum a);
```

In polar coordinate
```C
inline double Cnum_Radius(Cnum a);
inline double Cnum_Angle(Cnum a);
inline Cnum Cnum_Pole(double radius, double angle);
```