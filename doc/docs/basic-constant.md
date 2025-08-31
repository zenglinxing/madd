Constants
===

The constants defined in Madd are global variables. Some may have macros. The global variables may be changed by the program, thus result in unpredictable result. And the CUDA device code won't accept the host global variables. So the macros is defined in paralle with global variables.

Common Constants
---

| constant | global variable | macro |
| :------: | :-------------- | :---- |
| $\pi$ | `Pi` | _CONSTANT_PI |
| $e$ | `E_Nat` | _CONSTANT_E |
| $\infty$ | `Inf` |  |

Binary Mask
---

| global variable | macro |
| :-------------: | :---: |
| Bin4 | BIN4 |
| Bin5 |  |
| Bin6 |  |
| Bin7 | BIN7 |
| Bin8 | BIN8 |
| Bin15 | BIN15 |
| Bin16 | BIN16 |
| Bin31 | BIN31 |
| Bin32 | BIN32 |
| Bin63 | BIN63 |
| Bin64 | BIN64 |