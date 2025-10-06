Eigenvalues & Eigenvectors
===

| variable | explanation |
| -------- | ----------- |
| $A$ | matrix |
| $\lambda$ | eigenvalue |
| $v$ | eigenvector |

Left eigen problem

$$
v^{H}A = \lambda v^{H}
$$

where $v^{H}$ refers to the Hermitian transpose of eigenvector $v$.

Right eigen problem

$$
Av = \lambda v
$$

Besides, there is another kind of eigen: generailized eigen. Suppose $B$ is a $n\times n$matrix, the left eigen problem is described as 

$$
v^{H}A = \lambda v^{H} B
$$

Right eigen problem is described as 

$$
Av = \lambda B v
$$



Functions
---

```C
bool Eigen(int n, double *matrix, Cnum *eigenvalue,
           bool flag_left, Cnum *eigenvector_left,
           bool flag_right, Cnum *eigenvector_right);
bool Eigen_c64(int n, Cnum *matrix, Cnum *eigenvalue,
               bool flag_left, Cnum *eigenvector_left,
               bool flag_right, Cnum *eigenvector_right);
```

| parameter | explanation |
| --------- | ----------- |
| `n` | dimension of `matrix` (`n` x `n`) |
| `matrix` | matrix to be solved, must be `n` $\times$ `n` |
| `eigenvalue` | eigenvalues of `matrix`, must be dimension `n` |
| `flag_left` | if left eigenvectors should be solved, either `true` or `false` |
| `eigenvector_left` | left eigenvectors, must be dimension of `n` x `n` |
| `flag_right` | if right eigenvectors should be solved, either `true` or `false` |
| `eigenvector_right` | right eigenvectors, must be dimension of `n` x `n` |

If you only want the eigenvalues, you can set both `flag_left` and `flag_right` to be `false`.

The *i*-th column (not row) of `eigenvector_left` and `eigenvector_right` corresponds to the eigenvector of *i*-th eigenvalue.

```C
/* generalized eigen */
bool Generalized_Eigen(int n, double *matrix_A, double *matrix_B,
                       Cnum *eigenvalue,
                       bool flag_left, Cnum *eigenvector_left,
                       bool flag_right, Cnum *eigenvector_right);
bool Generalized_Eigen_c64(int n, Cnum *matrix_A, Cnum *matrix_B,
                           Cnum *eigenvalue,
                           bool flag_left, Cnum *eigenvector_left,
                           bool flag_right, Cnum *eigenvector_right);
```

| parameter | explanation |
| --------- | ----------- |
| `matrix_A` | the matrix $A$ in generalized eigen, must be `n` $\times$ `n` |
| `matrix_B` | the matrix $B$ in generalized eigen, must be `n` $\times$ `n` |

CUDA 64-bit Function
---

Since CUDA 12.6, a new function `cusolverDnXgeev` is introduced, and it was the first time to solve a general matrix eigen problem (*Not Generalized eigen*) via CUDA.

The parameter of `Eigen_cuda64` are the same as `Eigen`.

*NOTE: at present,* `cusolverDnXgeev` *does not support the left eigenvector problem. So if you set* `flag_left=true`*, you will probably get an error, and the function will stop.*

```C
bool Eigen_cuda64(int64_t n, double *matrix,
                  Cnum *eigenvalue,
                  bool flag_left, Cnum *eigenvector_left,
                  bool flag_right, Cnum *eigenvector_right);
bool Eigen_cuda64_f32(int64_t n, float *matrix,
                      Cnum32 *eigenvalue,
                      bool flag_left, Cnum32 *eigenvector_left,
                      bool flag_right, Cnum32 *eigenvector_right);
bool Eigen_cuda64_c64(int64_t n, Cnum *matrix,
                      Cnum *eigenvalue,
                      bool flag_left, Cnum *eigenvector_left,
                      bool flag_right, Cnum *eigenvector_right);
bool Eigen_cuda64_c32(int64_t n, Cnum32 *matrix,
                      Cnum32 *eigenvalue,
                      bool flag_left, Cnum32 *eigenvector_left,
                      bool flag_right, Cnum32 *eigenvector_right);
```