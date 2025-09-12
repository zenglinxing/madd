Complex Number
===

Complex number is defined as `struct`, `Cnum` in Madd. The following `inline` functions are defined for `Cnum` type.

```C
inline Cnum Cnum_Add(Cnum a, Cnum b);
inline Cnum Cnum_Sub(Cnum a, Cnum b);
inline Cnum Cnum_Mul(Cnum a, Cnum b);
inline Cnum Cnum_Mul_Real(Cnum a, double b);
inline Cnum Cnum_Div(Cnum a, Cnum b);
inline Cnum Cnum_Value(double real, double imag);
inline bool Cnum_Eq(Cnum a, Cnum b);
inline Cnum Cnum_Div_Real(Cnum a, double b);
inline Cnum Real_Div_Cnum(double a, Cnum b);
// square of the modulus
inline double Cnum_Mod2(Cnum a);
inline Cnum Cnum_Dot(Cnum a, Cnum b);
inline Cnum Cnum_Conj(Cnum a);
inline double Cnum_Radius(Cnum a);
inline double Cnum_Angle(Cnum a);
inline Cnum Cnum_Pole(double radius, double angle);
```