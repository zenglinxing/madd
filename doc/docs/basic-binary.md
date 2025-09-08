Binary Number
===

Madd is developed for binary 64-bit platform. It provides multiple functions for processing binary numbers (integer & float).

Binary Union Definition
---

Madd provides `union` to save integer and float in the same memory place. Madd functions resolving binary problems will input/return these `union` for consistency.

```C
union _union8{
    uint8_t u;
    int8_t i;
};

union _union16{
    uint16_t u;
    int16_t i;
    uint8_t u8[2];
    int8_t i8[2];
};

union _union32{
    uint32_t u;
    int32_t i;
    float f;
    uint8_t u8[4];
    int8_t i8[4];
    uint16_t u16[2];
    uint16_t i16[2];
};

union _union64{
    uint64_t u;
    int64_t i;
    double f;
    uint8_t u8[8];
    int8_t i8[8];
    uint16_t u16[4];
    int16_t i16[4];
    uint32_t u32[2];
    int32_t i32[2];
    float f32[2];
};
```

Number of 1
---

Counting number of 1 in a binary number.

```C
uint8_t Binary_Number_of_1_8bit(union _union8 u8);
uint8_t Binary_Number_of_1_16bit(union _union16 u16);
uint8_t Binary_Number_of_1_32bit(union _union32 u32);
uint8_t Binary_Number_of_1_64bit(union _union64 u64);
```

$\log_{2}$ Integer
---

The result of `Log2_Floor` should be equal to or less than $\log_{2} x$. The result of `Log2_Ceil` should be equal to or larger than $\log_{2} x$.

```C
uint64_t Log2_Floor(uint64_t x);
uint64_t Log2_Ceil(uint64_t x);
void Log2_Full(uint64_t x, uint64_t *lower, uint64_t *upper);
```

Bit Reverse
---

```C
inline union _union8 Bit_Reverse_8(union _union8 x);
inline union _union16 Bit_Reverse_16(union _union16 x);
inline union _union32 Bit_Reverse_32(union _union32 x);
inline union _union64 Bit_Reverse_64(union _union64 x);
```

Byte Reverse
---

```C
union _union16 Byte_Reverse_16(union _union16 u);
union _union32 Byte_Reverse_32(union _union32 u);
union _union64 Byte_Reverse_64(union _union64 u);
void Byte_Reverse_Allocated(uint64_t n, void *arr, void *narr);
void *Byte_Reverse(uint64_t n, void *arr);
```

`Byte_Reverse` return a pointer with allocated space, and `Byte_Reverse_Allocated` supposes you have allocate enough space for `narr`.

Endian Type
---

The endian type on different platforms may be varied. The function `Endian_Type` returns 1 if your machine is big endian, or returns 0 if little endian.

```C
inline bool Endian_Type(void);
```

Hash IEEE-754 Float to Integer
---

If the float number on your machine obeys IEEE-754, you can map your float number to an integer via the following 2 functions. The corresponding integers still suffice the relationship of float numbers' sizes. It could be helpful to sort the float numbers when you are applying sorting integer algorithms.

```C
inline uint64_t Hash_IEEE754_double_to_uint64(double x);
inline uint32_t Hash_IEEE754_float_to_uint32(float x);
```