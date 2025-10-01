DCT Functions
===

Discrete Cosine Transform (DCT) and Inverse Discrete Cosine Transform (IDCT) are common in signaling and used in Clenshaw-Curtis quadrature, etc.
DCT could be accelerated via FFT.
Madd has realized DCT-II and IDCT-II, which are the most prevalent in DCT.

DCT Function
---

```C
bool Discrete_Cosine_Transform_2(uint64_t n, double *arr);
bool Inverse_Discrete_Cosine_Transform_2(uint64_t n, double *arr);
```