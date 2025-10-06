DCT Functions
===

Discrete Cosine Transform (DCT) and Inverse Discrete Cosine Transform (IDCT) are common in signaling and used in Clenshaw-Curtis quadrature, etc.
DCT could be accelerated via FFT.
Madd has realized DCT-II and IDCT-II, which are the most prevalent in DCT.

DCT-II

$$
F[i] = \sum_{i=0}^{N-1} a_{i} \sqrt{\frac{2}{N}} T[j] cos\frac{i(2j+1)\pi}{2N}
$$

where $a_{i} = 1/\sqrt{2}$ when $i=0$, and $a_{i} = 1$ elsewise.

IDCT-II

$$
T[i] = \sum_{i=0}^{N-1} a_{j} \sqrt{\frac{2}{N}} F[j] cos\frac{(2i+1)j\pi}{2N}
$$

where $a_{j} = 1/\sqrt{2}$ when $j=0$, and $a_{j} = 1$ elsewise.

DCT Function
---

```C
bool Discrete_Cosine_Transform_2(uint64_t n, double *arr);
bool Inverse_Discrete_Cosine_Transform_2(uint64_t n, double *arr);
```