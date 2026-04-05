from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from frax.utils.linalg_utils import random_spd_matrix, fast_spd_inverse


class StructuredSPDInverseResult(NamedTuple):
    """Stores the result of inverting a SPD matrix with the following structure:
    ```
    A  0  0    0  0
    0  B  0    0  0
    0  0  C    F  G
    0  0  F^T  D  0
    0  0  G^T  0  E
    ```
    The inverse has the following structure:
    ```
    11  0   0     0     0
    0   22  0     0     0
    0   0   33    34    35
    0   0   34^T  44    45
    0   0   35^T  45^T  55
    ```
    """

    # On-diagonal entries of the inverse
    inv_11: jax.Array
    inv_22: jax.Array
    inv_33: jax.Array
    inv_44: jax.Array
    inv_55: jax.Array
    # Off-diagonal entries
    inv_34: jax.Array
    inv_35: jax.Array
    inv_45: jax.Array


def structured_spd_inverse(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: jax.Array,
    E: jax.Array,
    F: jax.Array,
    G: jax.Array,
) -> jax.Array:
    """Computes the inverse of a symmetric positive definite (SPD) matrix with the following structure:
    ```
    A  0  0    0  0
    0  B  0    0  0
    0  0  C    F  G
    0  0  F^T  D  0
    0  0  G^T  0  E
    ```
    where all blocks along the diagonal (A, B, C, D, E) are also SPD
    """
    # Independent inversions
    A_inv = fast_spd_inverse(A)
    B_inv = fast_spd_inverse(B)
    D_inv = fast_spd_inverse(D)
    E_inv = fast_spd_inverse(E)

    # Coupling Terms (cached)
    V_F = F @ D_inv
    V_G = G @ E_inv

    # Schur complement: C - F(D^-1)F^T - G(E^-1)G^T
    # Note: V_F @ F.T is (F @ D^-1) @ F.T
    schur = C - V_F @ F.T - V_G @ G.T

    # Invert the schur complement (C block)
    # TODO: Any structure to take advantage here with inverse?
    inv_schur = jnp.linalg.inv(schur)

    # Off-Diagonal Coupling Blocks (F/G blocks)
    # H_12 = -inv_schur @ F @ D^-1
    inv_34 = -inv_schur @ V_F
    inv_35 = -inv_schur @ V_G

    # D and E blocks
    # inv_44 = D^-1 + D^-1 F^T inv_schur F D^-1
    # inv_44 = D^-1 - (D^-1 F^T) @ inv_34
    # likewise for inv_55 with E
    inv_44 = D_inv - V_F.T @ inv_34
    inv_55 = E_inv - V_G.T @ inv_35

    # Coupling between D and E blocks
    # inv_45 = D^-1 F^T inv_schur G E^-1
    # inv_45 = -V_F.T @ inv_35
    inv_45 = -V_F.T @ inv_35

    return StructuredSPDInverseResult(
        inv_11=A_inv,
        inv_22=B_inv,
        inv_33=inv_schur,
        inv_44=inv_44,
        inv_55=inv_55,
        inv_34=inv_34,
        inv_35=inv_35,
        inv_45=inv_45,
    )


def structured_spd_result_to_matrix(res: StructuredSPDInverseResult) -> jax.Array:
    n1 = res.inv_11.shape[0]
    n2 = res.inv_22.shape[0]
    n3 = res.inv_33.shape[0]
    n4 = res.inv_44.shape[0]
    n5 = res.inv_55.shape[0]

    Z_13 = jnp.zeros((n1, n3))
    Z_14 = jnp.zeros((n1, n4))
    Z_15 = jnp.zeros((n1, n5))
    Z_12 = jnp.zeros((n1, n2))
    Z_23 = jnp.zeros((n2, n3))
    Z_24 = jnp.zeros((n2, n4))
    Z_25 = jnp.zeros((n2, n5))

    res_row1 = jnp.hstack([res.inv_11, Z_12, Z_13, Z_14, Z_15])
    res_row2 = jnp.hstack([Z_12.T, res.inv_22, Z_23, Z_24, Z_25])
    res_row3 = jnp.hstack([Z_13.T, Z_23.T, res.inv_33, res.inv_34, res.inv_35])
    res_row4 = jnp.hstack([Z_14.T, Z_24.T, res.inv_34.T, res.inv_44, res.inv_45])
    res_row5 = jnp.hstack([Z_15.T, Z_25.T, res.inv_35.T, res.inv_45.T, res.inv_55])

    return jnp.vstack([res_row1, res_row2, res_row3, res_row4, res_row5])


def structured_spd_inv_mvp(res: StructuredSPDInverseResult, v: jax.Array) -> jax.Array:
    """
    Computes y = M^-1 @ v efficiently using the block structure.
    """
    nA = res.inv_11.shape[0]
    nB = res.inv_22.shape[0]
    nC = res.inv_33.shape[0]
    nD = res.inv_44.shape[0]

    # v corresponds to [a, b, c, d, e] stacked
    idx_a = nA
    idx_b = idx_a + nB
    idx_c = idx_b + nC
    idx_d = idx_c + nD

    v_a = v[0:idx_a]
    v_b = v[idx_a:idx_b]
    v_c = v[idx_b:idx_c]
    v_d = v[idx_c:idx_d]
    v_e = v[idx_d:]

    # 1. Independent Blocks
    res_a = res.inv_11 @ v_a
    res_b = res.inv_22 @ v_b

    # 2. Coupled Blocks (The "Arrowhead")
    # Rows for C: Inv_C*v_c + Inv_F*v_d + Inv_G*v_e
    res_c = res.inv_33 @ v_c + res.inv_34 @ v_d + res.inv_35 @ v_e

    # Rows for D: Inv_F.T*v_c + Inv_D*v_d + Inv_DE*v_e
    res_d = res.inv_34.T @ v_c + res.inv_44 @ v_d + res.inv_45 @ v_e

    # Rows for E: Inv_G.T*v_c + Inv_DE.T*v_d + Inv_E*v_e
    res_e = res.inv_35.T @ v_c + res.inv_45.T @ v_d + res.inv_55 @ v_e

    return jnp.concatenate([res_a, res_b, res_c, res_d, res_e])


@jax.jit
def structured_spd_inv_right_multiply(
    res: StructuredSPDInverseResult, X: jax.Array
) -> jax.Array:
    """
    Computes M^-1 @ X
    Maps sparse_mvp over the columns (axis 1) of X
    """
    # in_axes=(None, 1): Keep blocks fixed, map over col of X
    # out_axes=1: Stack results as columns
    return jax.vmap(structured_spd_inv_mvp, in_axes=(None, 1), out_axes=1)(res, X)


@jax.jit
def structured_spd_inv_left_multiply(res: StructuredSPDInverseResult, X) -> jax.Array:
    """
    Computes X @ M^-1
    Maps sparse_mvp over the rows (axis 0) of X
    """
    # in_axes=(None, 0): Keep blocks fixed, map over row of X
    # out_axes=0: Stack results as rows
    # Logic: row @ M^-1 = (M^-1 @ row.T).T
    return jax.vmap(structured_spd_inv_mvp, in_axes=(None, 0), out_axes=0)(res, X)


def random_structured_spd_matrix():
    nA, nB, nC, nD, nE = 2, 2, 3, 4, 5

    A = random_spd_matrix(nA)
    B = random_spd_matrix(nB)
    C = random_spd_matrix(nC)
    D = random_spd_matrix(nD)
    E = random_spd_matrix(nE)
    F = np.random.normal(size=(nC, nD))
    G = np.random.normal(size=(nC, nE))
    return A, B, C, D, E, F, G


def structured_spd_to_matrix(A, B, C, D, E, F, G):
    n1 = A.shape[0]
    n2 = B.shape[0]
    n3 = C.shape[0]
    n4 = D.shape[0]
    n5 = E.shape[0]

    Z_13 = jnp.zeros((n1, n3))
    Z_14 = jnp.zeros((n1, n4))
    Z_15 = jnp.zeros((n1, n5))
    Z_12 = jnp.zeros((n1, n2))
    Z_23 = jnp.zeros((n2, n3))
    Z_24 = jnp.zeros((n2, n4))
    Z_25 = jnp.zeros((n2, n5))
    Z_45 = jnp.zeros((n4, n5))

    res_row1 = jnp.hstack([A, Z_12, Z_13, Z_14, Z_15])
    res_row2 = jnp.hstack([Z_12.T, B, Z_23, Z_24, Z_25])
    res_row3 = jnp.hstack([Z_13.T, Z_23.T, C, F, G])
    res_row4 = jnp.hstack([Z_14.T, Z_24.T, F.T, D, Z_45])
    res_row5 = jnp.hstack([Z_15.T, Z_25.T, G.T, Z_45.T, E])

    return jnp.vstack([res_row1, res_row2, res_row3, res_row4, res_row5])
