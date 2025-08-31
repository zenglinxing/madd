Random Number Generator (RNG)
===

Generally speaking, RNG is such a method that generates a float number in [0, 1) (in most numerical library, this function is named *rand*). Many applications and algorithms are based on random number. For most of the RNG algorithms, when a random number seed is set to initialized the state of RNG, the random number is generated recursively. But there are also some techniques to generate random number, such as intrinsics of x86 CPU which does not accept seed to initialized.

One underlying step, is that most of RNG generate an integer *i* between [0, M), and the float number is *i*/M, where M is usually set as $2^{64}$ or $2^{32}$. Madd observes this step. For each RNG, Madd provides a function to generate the (random) integer, and the float number is generated correspondingly.

RNG Algorithms & Functions
---

All the RNG algorithms have such struct and functions are in the form of follows. \<algorithm\> should be replaced by the algorithm's name, as listed in the next section. But do note some algorithms may have different functions, which we will introduce in the following context.

```C
/* RNG parameters */
typedef struct ... RNG_<algorithm>_Param;

/* RNG parameters initialization */
RNG_<algorithm>_Param RNG_<algorithm>_Init(uint64_t seed);

/* generate random integer */
/* whether U64 or U32, depends on the algorithm */
uint64_t/uint32_t RNG_<algorithm>_U64/U32(RNG_<algorithm>_Param *rng);

/* generate float number in [0, 1)] */
double Rand_<algorithm>(RNG_<algorithm>_Param *rng);
float Rand_<algorithm>_f32(RNG_<algorithm>_Param *rng);
long double Rand_<algorithm>_fl(RNG_<algorithm>_Param *rng);

/* read RNG parameters from file */
RNG_<algorithm>_Param RNG_<algorithm>_Read_BE(FILE *fp);
RNG_<algorithm>_Param RNG_<algorithm>_Read_LE(FILE *fp);

/* write RNG parameters to file */
void RNG_<algorithm>_Write_BE(RNG_<algorithm>_Param *rng, FILE *fp);
void RNG_<algorithm>_Write_LE(RNG_<algorithm>_Param *rng, FILE *fp);
```

# Optional Algorithms

Here lists the algorithms and the corresponding \<algorithm\> in the struct and functions.

* **Mersenne Twister**: \<algorithm\>=MT. Here only apply the MT19937-64 parameters. 

* **C library**: \<algorithm\>=Clib. The default RNG in C standard library. It has no Read and Write functions.

* **x86**: \<algorithm\>=x86. The RNG by x86 CPU. So only when Madd is built on x86 platform could it be used. It has no Read and Write functions.

* **Xorshift64** / **Xorshift64\***: \<algorithm\>=Xorshift64/Xorshift64s. They share the same RNG\_Xorshift64\_Param.

* **Xorshift1024\***: \<algorithm\>=Xorshift1024s.

* **Xoshiro256+** / **Xoshiro256\*\***: \<algorithm\>=Xoshiro256p/Xoshiro256ss.

* **Xorwow**: \<algorithm\>=Xorwow. Note its function name RNG_Xorwow_U32.

# Example

Take MT algorithm as an example.

```C
#include<stdio.h>
#include<madd.h>

int main(int argc, char *argv[])
{
    uint64_t seed = 10;
    RNG_MT_Param rng = RNG_MT_Init(seed);
    
    // generate random number
    double array[8];
    uint64_t i;
    for (i=0; i<8; i++){
        array[i] = Rand_MT(&rng);
    }
    
    // save RNG
    FILE *fp = fopen("RNG_MT-BE", "wb");
    RNG_MT_Write_BE(&rng);
    fclose(fp);
    
    // load RNG
    RNG_MT_Param mt;
    fp = fopen("RNG_MT-BE", "rb");
    RNG_MT_Read_BE(&mt);
    fclose(fp);
    return 0;
}
```

General RNG
---

Other Madd functions that rely on RNG only accept the unified RNG interfaces as follows.

```C
/* General RNG parameter */
typedef struct ... RNG_Param;

/* initialize RNG parameters */
RNG_Param RNG_Init(uint64_t seed, uint32_t rng_type);

uint64_t Rand_Uint(RNG_Param *rng);

double Rand(RNG_Param *rng);
float Rand_f32(RNG_Param *rng);
long double Rand_fl(RNG_Param *rng);

RNG_Param RNG_Read_BE(FILE *fp);
RNG_Param RNG_Read_LE(FILE *fp);
void RNG_Write_BE(RNG_Param *rng, FILE *fp);
void RNG_Write_LE(RNG_Param *rng, FILE *fp);
```

`rng_type` of `RNG_Init` specifies the RNG algorithm. The following table provides the macro you could input for `rng_type`. If you have no idea which algorithm to choose, just input 0, which set Xoshiro256**.

| algorithm | `rng_type` |
| --------- | ---------- |
| Xoshiro256** | RNG_XOSHIRO256SS |
| MT | RNG_MT |
| C library | RNG_CLIB |
| Xorshift64 | RNG_XORSHIFT64 |
| Xorshift64* | RNG_XORSHIFT64S |
| Xorshift1024* | RNG_XORSHIFT1024S |
| Xoshiro256+ | RNG_XOSHIRO256P |
| Xorwow | RNG_XORWOW |
| x86 | RNG_X86 |

The unified RNG interfaces are much convenient since you don't need to concern about the details of algorithm. However, `RNG_Param` consumes more memory space than any specific algorithm's parameters.

One thing to note is that not every algorithm return a random integer ranging from 0 to BIN64.

# Example

```C
#include<stdio.h>
#include<madd.h>

int main(int argc, char *argv[])
{
    uint64_t seed = 10;
    // choose Xoshiro256** algorithm
    RNG_Param rng = RNG_Init(seed, RNG_XOSHIRO256SS);
    
    // generate random number
    double array[8];
    uint64_t i;
    for (i=0; i<8; i++){
        array[i] = Rand(&rng);
    }
    
    // save RNG
    FILE *fp = fopen("RNG_MT-BE", "wb");
    RNG_Write_BE(&rng);
    fclose(fp);
    
    // load RNG
    RNG_Param rng2;
    fp = fopen("RNG_MT-BE", "rb");
    RNG_Read_BE(&rng2);
    fclose(fp);
    return 0;
}
```