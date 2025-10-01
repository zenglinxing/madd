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

where $v^{H}$ refer to the Hermitian transpose of eigenvector $v$.

Right eigen problem

$$
Av = \lambda v
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
| `matrix` | matrix to be solved, must be `n` x `n` |
| `eigenvalue` | eigenvalues of `matrix`, must be dimension `n` |
| `flag_left` | if left eigenvectors should be solved, either `true` or `false` |
| `eigenvector_left` | left eigenvectors, must be dimension of `n` x `n` |
| `flag_right` | if right eigenvectors should be solved, either `true` or `false` |
| `eigenvector_right` | right eigenvectors, must be dimension of `n` x `n` |

If you only want the eigenvalues, you can set both `flag_left` and `flag_right` to be `false`.

The *i*-th column (not row) of `eigenvector_left` and `eigenvector_right` corresponds to the eigenvector of *i*-th eigenvalue.