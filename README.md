Math Addition (Madd)
===

Author: Lin-Xing Zeng

Email:  jasonphysics@outlook.com | zenglinxing@petalmail.com

Open Source License: MIT

Introduction
---

Madd is an open source C library for numerical computations. It integrates the numerical functions and error/warning logging internally.

To build it, you should guarantee that your C compiler supports C99 standard and is running on a binary 64-bit platform where 1 byte = 8 bit.

Build
---

```bash
cd src
mkdid build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<your install path> -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest
cmake --install .
```

Usage
---

If you had built the document madd-guide.pdf when configuring and building the library, you should find it at share/doc/madd in your install path. This document elucidate the functions of madd.

In C source

```C
#include<madd/madd.h>
```

Link with madd library `-lmadd`.