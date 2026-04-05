import time
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.linalg import cho_factor, cho_solve

# Ensure 64-bit precision for fair comparison (matrix inversion is sensitive)
jax.config.update("jax_enable_x64", True)


def invert_sparse_robust(blocks):
    """
    Robust version: Uses Cholesky for Leaves (speed)
    but LU/Standard Solve for Hubs (stability).
    """

    # 1. Unpack
    A, B, C, D, E, F = (
        blocks["A"],
        blocks["B"],
        blocks["C"],
        blocks["D"],
        blocks["E"],
        blocks["F"],
    )
    G, H, I, J, K = blocks["G"], blocks["H"], blocks["I"], blocks["J"], blocks["K"]
    L, M = blocks["L"], blocks["M"]

    # 2. Factorize Leaves (Keep Cholesky here, it's safe for inputs)
    B_cho = cho_factor(B)
    C_cho = cho_factor(C)
    E_cho = cho_factor(E)
    F_cho = cho_factor(F)

    # 3. Eliminate Leaves (Condense)

    # Terms for A from B, C
    # Note: We use cho_solve(B_cho, G.T) which is equivalent to B^-1 @ G.T
    term_B = G @ cho_solve(B_cho, G.T)
    term_C = H @ cho_solve(C_cho, H.T)

    # Terms for A, D, I from E
    J_Einv = cho_solve(E_cho, J.T).T
    L_Einv = cho_solve(E_cho, L.T).T

    # Terms for A, D, I from F
    K_Finv = cho_solve(F_cho, K.T).T
    M_Finv = cho_solve(F_cho, M.T).T

    # Update Hubs
    # We do the subtractions.
    # Even if these drift slightly from SPD, the logic below handles it.
    A_mod = A - term_B - term_C - J_Einv @ J.T - K_Finv @ K.T
    D_mod = D - L_Einv @ L.T - M_Finv @ M.T
    I_mod = I - J_Einv @ L.T - K_Finv @ M.T

    # 4. Eliminate D (Secondary Hub)

    # CRITICAL CHANGE: Do not use Cholesky for D_mod.
    # Use standard solve (LU based) to handle potential numerical noise.
    # We need to compute: term_D = I_mod @ (D_mod^-1 @ I_mod.T)

    # D_mod_inv_I_T = D_mod^-1 @ I_mod.T
    D_inv_IT = jnp.linalg.solve(D_mod, I_mod.T)

    term_D = I_mod @ D_inv_IT
    A_final = A_mod - term_D

    # 5. Define Solve (Back-substitution)
    def solve(rhs):
        bA, bB, bC, bD, bE, bF = jnp.split(rhs, 6, axis=0)

        # Forward Elimination
        bA = bA - G @ cho_solve(B_cho, bB) - H @ cho_solve(C_cho, bC)

        xE_tmp = cho_solve(E_cho, bE)
        xF_tmp = cho_solve(F_cho, bF)

        bA = bA - J @ xE_tmp - K @ xF_tmp
        bD = bD - L @ xE_tmp - M @ xF_tmp

        # Eliminate D from A
        # We need D_mod^-1 @ bD. Use standard solve.
        xD_tmp = jnp.linalg.solve(D_mod, bD)
        bA = bA - I_mod @ xD_tmp

        # --- Solve Root (A) ---
        # CRITICAL CHANGE: Use standard solve for A_final as well
        xA = jnp.linalg.solve(A_final, bA)

        # --- Backward Substitution ---

        # Recover xD
        # D_mod @ xD = (bD - I_mod.T @ xA)
        rhs_D = bD - I_mod.T @ xA
        xD = jnp.linalg.solve(D_mod, rhs_D)

        # Recover Leaves (Cholesky is fine here)
        xB = cho_solve(B_cho, bB - G.T @ xA)
        xC = cho_solve(C_cho, bC - H.T @ xA)
        xE = cho_solve(E_cho, bE - J.T @ xA - L.T @ xD)
        xF = cho_solve(F_cho, bF - K.T @ xA - M.T @ xD)

        return jnp.concatenate([xA, xB, xC, xD, xE, xF], axis=0)

    return solve


# --- 1. The Sparse Solver (from previous step) ---
def invert_sparse_robust_wrapper(blocks, b):
    """
    Wraps the robust sparse solver to be JIT-compatible with a direct call.
    Returns x such that Ax = b.
    """
    # Instantiate the solver closure
    solver = invert_sparse_robust(blocks)
    # Solve for b
    return solver(b)


# --- 2. The Dense Solver (Baseline) ---
def dense_solve_spd(blocks, b):
    """
    Constructs the dense matrix and uses JAX's optimized SPD solver.
    """
    # Reconstruct the full dense matrix M
    # (Same logic as construct_dense, inlined for JIT)
    z = jnp.zeros_like(blocks["A"])
    r1 = jnp.hstack(
        [blocks["A"], blocks["G"], blocks["H"], blocks["I"], blocks["J"], blocks["K"]]
    )
    r2 = jnp.hstack([blocks["G"].T, blocks["B"], z, z, z, z])
    r3 = jnp.hstack([blocks["H"].T, z, blocks["C"], z, z, z])
    r4 = jnp.hstack([blocks["I"].T, z, z, blocks["D"], blocks["L"], blocks["M"]])
    r5 = jnp.hstack([blocks["J"].T, z, z, blocks["L"].T, blocks["E"], z])
    r6 = jnp.hstack([blocks["K"].T, z, z, blocks["M"].T, z, blocks["F"]])

    M = jnp.vstack([r1, r2, r3, r4, r5, r6])

    # Use assume_a='pos' to use Cholesky (fastest dense method)
    return jsp.linalg.solve(M, b, assume_a="pos")


# --- 3. Benchmarking Harness ---


def run_benchmark(N=200):
    print(
        f"--- Benchmarking with Block Size N={N} (Total Matrix Size {6 * N}x{6 * N}) ---"
    )

    # Generate Data
    key = jax.random.PRNGKey(42)
    blocks = generate_random_problem(N, key)
    b = jax.random.normal(key, (6 * N, 1))

    # JIT Compile both functions
    # We JIT compile to measure raw calculation speed, excluding Python overhead
    print("Compiling functions...")
    sparse_jit = jax.jit(invert_sparse_robust_wrapper)
    dense_jit = jax.jit(dense_solve_spd)

    # --- Warmup Run ---
    # (JAX compiles on the first run, so we ignore this time)
    _ = sparse_jit(blocks, b).block_until_ready()
    _ = dense_jit(blocks, b).block_until_ready()
    print("Warmup complete.\n")

    # --- Time Dense Method ---
    start = time.time()
    result_dense = dense_jit(blocks, b)
    result_dense.block_until_ready()  # Force synchronization
    end = time.time()
    time_dense = end - start
    print(f"Dense Solver Time:  {time_dense:.6f} s")

    # --- Time Sparse Method ---
    start = time.time()
    result_sparse = sparse_jit(blocks, b)
    result_sparse.block_until_ready()  # Force synchronization
    end = time.time()
    time_sparse = end - start
    print(f"Sparse Solver Time: {time_sparse:.6f} s")

    # --- Comparison ---
    speedup = time_dense / time_sparse
    print(f"\nSpeedup: {speedup:.2f}x faster")

    # Verify Accuracy
    diff = jnp.linalg.norm(result_dense - result_sparse)
    print(f"Agreement Error: {diff:.2e}")


# --- Helper to Generate Data (Same as before) ---
def generate_random_problem(N, key):
    keys = jax.random.split(key, 20)

    def rand_spd(k):
        X = jax.random.normal(k, (N, N))
        return (
            X.T @ X + jnp.eye(N) * 1.0
        )  # Stronger diagonal to prevent conditioning issues

    def rand_mat(k):
        return jax.random.normal(k, (N, N)) * 0.1

    return {
        "A": rand_spd(keys[0]),
        "B": rand_spd(keys[1]),
        "C": rand_spd(keys[2]),
        "D": rand_spd(keys[3]),
        "E": rand_spd(keys[4]),
        "F": rand_spd(keys[5]),
        "G": rand_mat(keys[6]),
        "H": rand_mat(keys[7]),
        "I": rand_mat(keys[8]),
        "J": rand_mat(keys[9]),
        "K": rand_mat(keys[10]),
        "L": rand_mat(keys[11]),
        "M": rand_mat(keys[12]),
    }


# Run the test
if __name__ == "__main__":
    # VERY small test
    run_benchmark(N=6)

    # Small scale test
    run_benchmark(N=100)

    # Large scale test (Benefits become massive here)
    run_benchmark(N=500)
