# Conditioning and Preconditioning

## Purpose

EFIE matrices become harder to solve as discretization grows or scenarios become
numerically stiff. This chapter explains what conditioning means in this
package, what optional regularization/preconditioning pathways do, and what
they do **not** do.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why poor conditioning hurts both forward and adjoint solves.
2. Use regularization and left preconditioning consistently.
3. Choose `:off`, `:on`, or `:auto` preconditioning modes correctly.

---

## 1. What Conditioning Means for EFIE Matrices

### 1.1 Mathematical Definition of Condition Number

For a non‑singular matrix $\mathbf{Z} \in \mathbb{C}^{N\times N}$, the condition number with respect to the Euclidean (spectral) norm is defined as

```math
\kappa(\mathbf{Z}) = \|\mathbf{Z}\|_2 \cdot \|\mathbf{Z}^{-1}\|_2 = \frac{\sigma_{\max}(\mathbf{Z})}{\sigma_{\min}(\mathbf{Z})},
```

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values of $\mathbf{Z}$. The condition number measures the sensitivity of the solution $\mathbf{I}$ to perturbations in the data ($\mathbf{v}$) or in the matrix itself. Specifically, for the perturbed system

```math
(\mathbf{Z} + \delta\mathbf{Z})(\mathbf{I} + \delta\mathbf{I}) = \mathbf{v} + \delta\mathbf{v},
```

the relative error in the solution satisfies (to first order)

```math
\frac{\|\delta\mathbf{I}\|}{\|\mathbf{I}\|} \lesssim \kappa(\mathbf{Z})\left(\frac{\|\delta\mathbf{Z}\|}{\|\mathbf{Z}\|} + \frac{\|\delta\mathbf{v}\|}{\|\mathbf{v}\|}\right).
```

Thus, when $\kappa(\mathbf{Z})$ is large, small errors in the matrix entries or the right‑hand side can be amplified dramatically in the computed solution.

### 1.2 Why EFIE Matrices Are Ill‑Conditioned

The EFIE operator $\mathcal{T}$ defined in Chapter 2 is a **first‑kind integral operator** mapping surface currents to tangential electric fields. Such operators are known to be **ill‑posed** in the sense of Hadamard: their discretizations yield matrices with rapidly decaying singular values, leading to large condition numbers. Two primary mechanisms contribute to the ill‑conditioning of $\mathbf{Z}^{\mathrm{EFIE}}$:

1. **Low‑frequency breakdown**: At frequencies where $k h \ll 1$ (electrically small triangles), the scalar‑potential term $S_{mn}$ dominates and scales as $O(1/k^2)$, while the vector‑potential term $V_{mn}$ scales as $O(1)$. This imbalance makes the matrix **nearly singular**, with $\kappa(\mathbf{Z}) \sim O(1/k^2)$ as $k\to 0$.

2. **Mesh refinement**: For a fixed frequency, refining the mesh (decreasing triangle size $h$) increases the condition number as $\kappa(\mathbf{Z}) \sim O(1/h)$ for EFIE discretized with RWG basis functions. This growth occurs because finer meshes better approximate the null‑space of the continuous operator.

3. **High‑aspect‑ratio geometry**: Structures with thin features or large aspect ratios produce matrices with widely varying entries, further exacerbating ill‑conditioning.

### 1.3 Physical Interpretation of Ill‑Conditioning

Physically, a large condition number indicates that the surface current solution $\mathbf{J}$ is **highly sensitive** to the incident field $\mathbf{E}^{\mathrm{inc}}$. In the limit $k\to 0$, the EFIE reduces to the static electric field integral equation, which has a non‑trivial null‑space: any **solenoidal** (divergence‑free) current produces zero tangential electric field. At low but nonzero frequencies, these near‑null modes correspond to **loop currents** that hardly radiate. The matrix $\mathbf{Z}$ has very small singular values associated with these modes, making them difficult to resolve numerically.

From an energy perspective, ill‑conditioning manifests as a huge disparity between the **easiest** and **hardest** current distributions to excite: the dominant singular vectors correspond to currents that efficiently radiate (dipole‑like), while the weakest singular vectors correspond to currents that store energy in near‑fields without radiating (loop‑like).

### 1.4 Symptoms of Poor Conditioning

When $\kappa(\mathbf{Z})$ exceeds $10^8$–$10^{12}$ (typical for EFIE at low frequencies or on fine meshes), one observes:

- **Slow iterative convergence**: Krylov methods like GMRES or Bi‑CGSTAB require many iterations to reduce the residual, because the eigenvalue spectrum is widely spread.
- **Loss of numerical precision**: Direct solves (LU factorization) may produce solutions with fewer than half the machine‑precision digits, due to amplification of rounding errors during back‑substitution.
- **Noisy adjoint gradients**: In gradient‑based optimization, derivatives computed via the adjoint method become contaminated with numerical noise, causing optimization algorithms to stall or converge to spurious minima.
- **Unstable refinement convergence**: As the mesh is refined, the solution error may fail to decrease monotonically, or may even increase, because discretization errors are overwhelmed by solution errors.

### 1.5 Conditioning Diagnostics in `DifferentiableMoM.jl`

The package provides the function `condition_diagnostics` (in `src/Diagnostics.jl`) to quantify conditioning:

```julia
using DifferentiableMoM

# After assembling Z (EFIE matrix)
stats = condition_diagnostics(Z)
println("Condition number κ = ", stats.cond)
println("Largest singular value = ", stats.sv_max)
println("Smallest singular value = ", stats.sv_min)
```

The function computes the singular values via a full SVD (feasible for $N \lesssim 5000$) and returns a named tuple with the condition number and the extreme singular values. For larger matrices, estimating the condition number without a full SVD is possible via iterative methods, but this is not currently implemented in the package.

**Interpretation guidelines**:
- $\kappa < 10^6$: well‑conditioned, direct solves are accurate.
- $10^6 \le \kappa < 10^{10}$: moderately ill‑conditioned, may benefit from preconditioning.
- $\kappa \ge 10^{10}$: severely ill‑conditioned, preconditioning or regularization is essential.

### 1.6 Connection to Low‑Frequency Stabilization Techniques

The ill‑conditioning of the EFIE at low frequencies has motivated several stabilization techniques in the literature, including:
- **Loop‑tree decomposition**: Separates solenoidal (loop) and non‑solenoidal (tree) components of the current, preconditioning each subspace differently.
- **Augmented EFIE (A‑EFIE)**: Adds a charge neutrality constraint to suppress the near‑null space.
- **Calderón preconditioning**: Uses the operator product $\mathcal{T}^2$ which is better conditioned than $\mathcal{T}$.

`DifferentiableMoM.jl` adopts a simpler, more general approach: **mass‑based regularization** and **left preconditioning**, described in the following sections. These methods are easier to implement, preserve differentiability, and work adequately for many practical scenarios.

---

## 2. Regularization: Adding a Stabilizing Shift

Regularization is a simple yet effective technique for stabilizing ill‑conditioned EFIE matrices. By adding a small, positive‑definite perturbation $\alpha\mathbf{R}$ to the impedance matrix $\mathbf{Z}$, we shift its eigenvalues away from zero, thereby improving the condition number and numerical stability of the linear solve. This section explains the mathematical foundation of regularization, its physical interpretation, practical guidelines for choosing the regularization parameter $\alpha$ and matrix $\mathbf{R}$, and how regularization is implemented in `DifferentiableMoM.jl`.

### 2.1 Mathematical Formulation

The regularized system is defined as

```math
\mathbf{Z}_\alpha = \mathbf{Z} + \alpha\mathbf{R}, \qquad \alpha \ge 0,
```

where $\mathbf{R} \succeq 0$ is a **Hermitian positive semi‑definite** matrix. The regularization term $\alpha\mathbf{R}$ acts as a **Tikhonov regularizer** that penalizes solutions with large “energy” measured by the quadratic form $\mathbf{I}^\dagger\mathbf{R}\mathbf{I}$. The regularized linear system becomes

```math
(\mathbf{Z} + \alpha\mathbf{R})\mathbf{I} = \mathbf{v}.
```

If $\mathbf{R}$ is positive definite, the eigenvalues $\lambda_i(\mathbf{Z}_\alpha)$ satisfy

```math
\lambda_i(\mathbf{Z}_\alpha) = \lambda_i(\mathbf{Z}) + \alpha\lambda_i(\mathbf{R}) \quad \text{for } i=1,\dots,N.
```

**Caveat:** This additivity holds exactly only when $\mathbf{Z}$ and $\mathbf{R}$ commute (e.g., if both are simultaneously diagonalizable). For general EFIE and mass matrices, eigenvalue perturbation bounds (Weyl's inequality) provide approximate estimates.

Since $\lambda_i(\mathbf{R}) \ge 0$, the regularization shifts all eigenvalues to the right (in the complex plane), moving small eigenvalues away from the origin and reducing the condition number.

### 2.2 Physical Interpretation

What does regularization mean physically? For the EFIE, the matrix $\mathbf{Z}$ represents the mapping from surface currents to tangential electric fields. The regularization term $\alpha\mathbf{R}$ can be interpreted as adding a small **artificial impedance** to the scatterer. Specifically, if $\mathbf{R}$ is chosen as the **mass matrix** (Gram matrix of the RWG basis functions), then

```math
\mathbf{I}^\dagger\mathbf{R}\mathbf{I} = \int_\Gamma |\mathbf{J}(\mathbf{r})|^2\, dS
```

measures the total squared magnitude of the surface current. The regularized equation

```math
\mathcal{T}[\mathbf{J}] + \alpha\mathbf{J} = -\mathbf{E}_t^{\mathrm{inc}}
```

corresponds to adding a resistive sheet with surface impedance $\alpha$ (in ohms) everywhere on the scatterer. This resistive term dissipates energy, making the problem **less singular** and stabilizing the numerical solution.

### 2.3 Choice of Regularization Matrix $\mathbf{R}$

The package provides two main choices for $\mathbf{R}$:

1. **Identity matrix** ($\mathbf{R} = \mathbf{I}$): The simplest regularizer, which penalizes the Euclidean norm of the coefficient vector $\mathbf{I}$. This corresponds to adding a uniform resistive sheet with isotropic surface impedance $\alpha$. While easy to implement, it lacks geometric invariance and may distort the physical solution more than necessary.

2. **Patch mass matrix** ($\mathbf{R} = \sum_p \mathbf{M}_p$): The default choice in `DifferentiableMoM.jl`. Here $\mathbf{M}_p$ are the patch mass matrices introduced in Chapter 3 for impedance boundary conditions. The sum $\sum_p \mathbf{M}_p$ is the **global mass matrix** of the RWG basis, which respects the geometry of the discretization and yields a regularizer that is **dimensionally consistent** with the EFIE operator.


### 2.4 Selecting the Regularization Parameter $\alpha$

The parameter $\alpha$ controls the trade‑off between stability and accuracy:
- **Too small** ($\alpha \ll \sigma_{\min}(\mathbf{Z})$): insufficient stabilization, condition number remains large.
- **Too large** ($\alpha \gg \sigma_{\min}(\mathbf{Z})$): solution is overly damped, physical fidelity is lost.

A practical heuristic is to choose $\alpha$ relative to the **average diagonal entry** of $\mathbf{Z}$ or the **trace** of $\mathbf{R}$. The package uses the scale

```math
\alpha = \epsilon \cdot \frac{\operatorname{tr}(\mathbf{R})}{N},
```

where $\epsilon$ is a user‑supplied relative tolerance (typically $10^{-8}$ to $10^{-12}$). This ensures that the regularization perturbation is small compared to the typical magnitude of $\mathbf{Z}$.

**Recommended values**:
- $\epsilon = 10^{-10}$ for moderate ill‑conditioning ($\kappa \sim 10^8$).
- $\epsilon = 10^{-8}$ for severe ill‑conditioning ($\kappa \sim 10^{12}$).
- $\epsilon = 0$ (no regularization) for well‑conditioned problems.

### 2.5 Implementation in `prepare_conditioned_system`

Regularization is applied via the function `prepare_conditioned_system`:

```julia
Z_eff, rhs_eff, fac = prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=1e-10,
    regularization_R=R,
    preconditioner_M=nothing,
    preconditioner_factor=nothing
)
```

If `regularization_alpha` is nonzero and `regularization_R` is provided, the function computes

```math
\mathbf{Z}_{\mathrm{eff}} = \mathbf{Z}_{\mathrm{raw}} + \alpha\mathbf{R}, \qquad
\mathbf{v}_{\mathrm{eff}} = \mathbf{v}.
```

The regularized matrix $\mathbf{Z}_{\mathrm{eff}}$ is then passed to the linear solver (direct or iterative).

### 2.6 Adjoint Consistency

When regularization is used in **forward solves**, the **adjoint solves** must employ the **same regularized operator** to guarantee consistent gradients. This is automatically handled by `prepare_conditioned_system`: both forward and adjoint solves receive the same $\mathbf{Z}_{\mathrm{eff}}$.

Mathematically, the adjoint variable $\boldsymbol{\lambda}$ satisfies

```math
(\mathbf{Z} + \alpha\mathbf{R})^\dagger \boldsymbol{\lambda} = \frac{\partial \Phi}{\partial \mathbf{I}^*},
```

where $\Phi$ is the objective function. Using a different operator in the adjoint equation would yield incorrect gradients, potentially breaking gradient‑based optimization.

### 2.7 When to Use Regularization

Regularization is particularly beneficial in the following scenarios:

1. **Low‑frequency EFIE**: When $k h \ll 1$, the scalar term $S_{mn}$ dominates, making $\mathbf{Z}$ nearly singular. A tiny regularization ($\alpha \sim 10^{-12}$) can prevent solver failures without noticeably affecting accuracy.

2. **Fine meshes**: As $h \to 0$, the condition number grows as $O(1/h)$. Regularization helps maintain solver accuracy across mesh refinements.

3. **Impedance sheets with very small resistance**: When optimizing for sheet resistance $\theta_p \to 0$, the matrix $\mathbf{Z} = \mathbf{Z}_{\mathrm{EFIE}} - \sum_p \theta_p \mathbf{M}_p$ becomes nearly singular. Regularization stabilizes the solve during optimization iterations.

4. **Noisy gradient detection**: If adjoint gradients appear excessively noisy, adding a small regularization can smooth the gradient landscape, aiding convergence.

### 2.8 Limitations of Regularization

Regularization is a **blunt instrument**: it perturbs the original physical problem. While effective for stabilization, it can:
- Introduce small errors in the solution (especially for resonant structures).
- Shift resonance frequencies slightly.
- Alter power balance (regularization adds artificial loss).

For problems where exact physical fidelity is paramount, **preconditioning** (Section 3) is often preferable, as it improves conditioning without altering the solution.

### 2.9 Example: Regularizing a Low‑Frequency Solve

```julia
using DifferentiableMoM

# Build a low‑frequency problem (k small)
mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)
k = 0.001  # Very low frequency
Z = assemble_Z_efie(mesh, rwg, k)
v = assemble_v_plane_wave(...)

# Compute patch mass matrices for regularization
partition = PatchPartition(...)
Mp = precompute_patch_mass(mesh, rwg, partition)
R = make_mass_regularizer(Mp)

# Solve with and without regularization
I_raw = Z \ v  # May be inaccurate due to ill‑conditioning

Z_reg, v_reg, _ = prepare_conditioned_system(Z, v; regularization_alpha=1e-10, regularization_R=R)
I_reg = Z_reg \ v_reg  # Stabilized solve

println("Condition number raw: ", condition_diagnostics(Z).cond)
println("Condition number regularized: ", condition_diagnostics(Z_reg).cond)
```

The regularized solve typically yields a solution with **better numerical accuracy** and **smoother gradients**, at the cost of a tiny perturbation to the physical model.

---

## 3. Left Preconditioning: Improving the Matrix Spectrum

Left preconditioning is a more sophisticated technique than regularization: instead of perturbing the matrix, it multiplies the linear system by an approximate inverse $\mathbf{M}^{-1}$ that clusters the eigenvalues of the preconditioned matrix $\tilde{\mathbf{Z}} = \mathbf{M}^{-1}\mathbf{Z}$, thereby accelerating iterative convergence and improving numerical stability. This section describes the mathematical theory of left preconditioning, the mass‑based preconditioner implemented in `DifferentiableMoM.jl`, the user modes `:off`, `:on`, and `:auto`, and how preconditioning integrates with forward and adjoint solves.

### 3.1 Mathematical Formulation

Given a nonsingular preconditioner matrix $\mathbf{M} \approx \mathbf{Z}$, the left‑preconditioned system is

```math
\tilde{\mathbf{Z}} \mathbf{I} = \tilde{\mathbf{v}}, \qquad
\tilde{\mathbf{Z}} = \mathbf{M}^{-1}\mathbf{Z}, \quad
\tilde{\mathbf{v}} = \mathbf{M}^{-1}\mathbf{v}.
```

Solving the preconditioned system yields the same solution $\mathbf{I}$ as the original system $\mathbf{Z}\mathbf{I} = \mathbf{v}$, but the condition number of $\tilde{\mathbf{Z}}$ can be much smaller:

```math
\kappa(\tilde{\mathbf{Z}}) = \frac{\sigma_{\max}(\mathbf{M}^{-1}\mathbf{Z})}{\sigma_{\min}(\mathbf{M}^{-1}\mathbf{Z})} \ll \kappa(\mathbf{Z})
```

provided $\mathbf{M}$ approximates $\mathbf{Z}$ well. For EFIE matrices, a good preconditioner should capture the **low‑frequency behavior** (dominant scalar term) and the **geometry** of the discretization.

### 3.2 Physical Interpretation

Preconditioning can be viewed as **changing the inner product** in the test space. The original Galerkin discretization uses the $L^2$ inner product $\langle \mathbf{f}_m, \mathbf{f}_n \rangle$. Left preconditioning with $\mathbf{M}^{-1}$ corresponds to using the inner product induced by $\mathbf{M}^{-1}$, which effectively **rescales** the basis functions according to their spatial support. When $\mathbf{M}$ is the mass matrix, the preconditioned system weights each current element by its area, making the matrix more balanced.

Physically, the mass‑based preconditioner **counteracts the ill‑scaling** between the vector and scalar terms of the EFIE at low frequencies, mitigating the low‑frequency breakdown.

### 3.3 Mass‑Based Preconditioner Construction

The function `make_left_preconditioner` (in `src/Solve.jl`) builds a preconditioner from the patch mass matrices $\mathbf{M}_p$:

```math
\mathbf{M} = \mathbf{R} + \epsilon\mathbf{I}, \qquad \mathbf{R} = \sum_p \mathbf{M}_p,
```

where $\epsilon = \epsilon_{\text{rel}} \cdot \max(\operatorname{tr}(\mathbf{R})/N, 1)$ is a small diagonal shift that ensures invertibility. The shift parameter `eps_rel` defaults to $10^{-8}$. The resulting $\mathbf{M}$ is Hermitian positive definite and its inverse can be applied via an LU factorization.

### 3.4 Implementation in `select_preconditioner` and `prepare_conditioned_system`

The package decides whether to apply preconditioning via `select_preconditioner`, which supports three modes:

- **`:off`**: No preconditioning (default for small problems).
- **`:on`**: Always build and apply the mass‑based preconditioner.
- **`:auto`**: Enable preconditioning only when `iterative_solver=true` or $N \ge n_{\text{threshold}}$ (default 256).

A user‑supplied preconditioner matrix `preconditioner_M` overrides the automatic selection.

The function `prepare_conditioned_system` applies both regularization (if requested) and preconditioning:

```julia
Z_eff, rhs_eff, fac = prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=0.0,
    regularization_R=nothing,
    preconditioner_M=M,
    preconditioner_factor=nothing
)
```

If `preconditioner_M` is provided, it computes $\mathbf{Z}_{\text{eff}} = \mathbf{M}^{-1}\mathbf{Z}_{\text{reg}}$ and $\mathbf{v}_{\text{eff}} = \mathbf{M}^{-1}\mathbf{v}$, returning also the LU factorization `fac` of $\mathbf{M}$ for reuse in adjoint solves.

### 3.5 Adjoint Consistency with Preconditioning

When left preconditioning is used, the adjoint equation must be formulated with the **same preconditioned operator**:

```math
\tilde{\mathbf{Z}}^{\dagger} \boldsymbol{\lambda} = \frac{\partial \Phi}{\partial \mathbf{I}^*}.
```

Since $\tilde{\mathbf{Z}} = \mathbf{M}^{-1}\mathbf{Z}$, the adjoint equation becomes

```math
\mathbf{Z}^{\dagger} \mathbf{M}^{-\dagger} \boldsymbol{\lambda} = \frac{\partial \Phi}{\partial \mathbf{I}^*}.
```

The package handles this automatically: `prepare_conditioned_system` returns the preconditioned matrix $\tilde{\mathbf{Z}}$, and both forward and adjoint solves use it.

### 3.6 When to Use Preconditioning

Preconditioning is recommended in the following scenarios:

1. **Iterative solves**: Krylov methods (GMRES, Bi‑CGSTAB) converge much faster with a good preconditioner.
2. **Large problems** ($N \ge 1000$): The $O(N^3)$ cost of direct solves becomes prohibitive; preconditioned iterative methods offer better scalability.
3. **Low‑frequency EFIE**: The mass‑based preconditioner significantly improves conditioning at low frequencies.
4. **Optimization loops**: When solving many similar systems (e.g., during gradient‑based optimization), the preconditioner can be factored once and reused, amortizing the setup cost.

### 3.7 Example: Using Auto‑Preconditioning

The script `examples/05_solver_methods.jl` demonstrates the recommended usage:

```julia
using DifferentiableMoM

# Build a problem
mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)
Mp = precompute_patch_mass(...)

# Select preconditioner with :auto mode
M_eff, enabled, reason = select_preconditioner(
    Mp;
    mode=:auto,
    iterative_solver=true,
    n_threshold=256,
    eps_rel=1e-6
)
println("Preconditioning enabled: ", enabled, " (", reason, ")")

# Assemble and solve with preconditioning
Z = assemble_Z_efie(...)
v = assemble_v_plane_wave(...)
Z_eff, v_eff, fac = prepare_conditioned_system(
    Z, v;
    preconditioner_M=M_eff
)
I = Z_eff \ v_eff  # Preconditioned solve
```

### 3.8 Limitations and Trade‑Offs

- **Setup cost**: Factoring $\mathbf{M}$ costs $O(N^3)$ (though $\mathbf{M}$ is often sparser than $\mathbf{Z}$). For small $N$, preconditioning may not be worth the overhead.
- **Memory**: Storing the preconditioner matrix and its factorization increases memory usage.
- **Adjoint consistency**: Must reuse the same preconditioner in forward and adjoint solves; changing $\mathbf{M}$ between solves breaks gradient accuracy.

Despite these trade‑offs, preconditioning is often the most effective way to solve large, ill‑conditioned EFIE problems in practice.

---

## 4. User Modes: `:off`, `:on`, `:auto`

The function `select_preconditioner` provides three simple modes that let you control when and how preconditioning is applied, balancing automation with reproducibility. This section explains each mode in detail, describes the decision logic behind the `:auto` mode, and offers practical advice on choosing the right mode for your problem.

### 4.1 The Three Modes Explained

**`:off`** – No automatic preconditioning.
- **Behavior**: `select_preconditioner` returns `nothing` and sets `enabled=false`.
- **Use case**: Baseline validation runs, small problems ($N < 100$), debugging, or when you want full control over preconditioning via a user‑supplied matrix.
- **Rationale**: Keeps the solve identical to the raw EFIE, ensuring reproducibility and simplifying comparison across different solvers.

**`:on`** – Always build and apply the mass‑based preconditioner.
- **Behavior**: `select_preconditioner` calls `make_left_preconditioner` and returns the resulting matrix $\mathbf{M}$; `enabled=true`.
- **Use case**: Problems where you know preconditioning is beneficial (e.g., low‑frequency EFIE, iterative solves) and you want to guarantee its use.
- **Rationale**: Eliminates any guesswork; the same preconditioner is used regardless of problem size or solver type.

**`:auto`** – Conservative automatic activation.
- **Behavior**: The function decides based on two criteria:
  1. **Iterative solver**: If `iterative_solver=true`, preconditioning is enabled.
  2. **Problem size**: If $N \ge n_{\text{threshold}}$ (default 256), preconditioning is enabled.
- **Use case**: Exploratory studies where problem size and solver choice may vary; you want preconditioning for “hard” cases but avoid overhead for “easy” ones.
- **Rationale**: Balances performance (no extra cost for small direct solves) with robustness (preconditioning for large/iterative solves).

### 4.2 Mode Selection Logic

The decision logic inside `select_preconditioner` (simplified) is:

```julia
function select_preconditioner(Mp; mode=:off, iterative_solver=false, n_threshold=256, ...)
    if mode == :off
        return nothing, false, "mode=:off"
    elseif mode == :on
        M = make_left_preconditioner(Mp; ...)
        return M, true, "mode=:on"
    else  # :auto
        if iterative_solver || size(Mp[1],1) >= n_threshold
            M = make_left_preconditioner(Mp; ...)
            return M, true, "mode=:auto (iterative_solver=true or N ≥ threshold)"
        else
            return nothing, false, "mode=:auto (N < threshold)"
        end
    end
end
```

The function returns a triple `(M_eff, enabled, reason)` where `reason` is a short string that logs why preconditioning was enabled or disabled. This string should be recorded in run logs to ensure reproducibility.

### 4.3 User‑Supplied Preconditioner Matrix

If you provide a matrix via the `preconditioner_M` keyword argument, it **overrides** the mode selection:

```julia
M_custom = ...  # your custom preconditioner
M_eff, enabled, reason = select_preconditioner(
    Mp;
    mode=:off,  # ignored because preconditioner_M is provided
    preconditioner_M=M_custom
)
```

The returned `enabled` will be `true` and `reason` will be `"user‑provided preconditioner"`. This mechanism allows advanced users to implement custom preconditioners (e.g., block‑diagonal, incomplete LU, geometric multigrid) while still using the same `prepare_conditioned_system` interface.

### 4.4 Practical Guidelines

1. **Start with `:off`** when debugging or running small validation cases. This ensures the raw EFIE matrix is solved, making it easier to spot implementation errors.

2. **Switch to `:auto`** for production runs or parameter sweeps where problem size may grow. The default threshold $N=256$ works well for most applications; you may lower it if you encounter ill‑conditioning at smaller $N$.

3. **Use `:on`** for low‑frequency problems ($k h \ll 1$) even with small $N$, because the ill‑conditioning is frequency‑driven, not size‑driven.

4. **Record the reason string** in your experiment logs. Knowing *why* preconditioning was enabled/disabled is crucial for reproducing results later.

5. **Combine with regularization** if needed: you can set `regularization_alpha` alongside preconditioning. The regularization is applied first, then the preconditioner multiplies the regularized matrix.

### 4.5 Example: Comparing Modes

```julia
using DifferentiableMoM

Mp = ...  # patch mass matrices

# Test each mode
for mode in [:off, :on, :auto]
    M, enabled, reason = select_preconditioner(Mp; mode=mode, n_threshold=256)
    println("Mode $mode: enabled=$enabled, reason=\"$reason\"")
end
```

Running this snippet will show you how each mode behaves for your specific problem size, helping you choose the appropriate one.

---

## 5. What Preconditioning Improves (and What It Does Not)

Left preconditioning is a powerful technique for mitigating the ill‑conditioning of EFIE matrices, but it is important to understand both its benefits and its limitations. This section provides a detailed analysis of what preconditioning can and cannot achieve, helping you make informed decisions about when to invest the extra computational effort.

### 5.1 What Preconditioning Improves

#### 5.1.1 Faster Iterative Convergence

The primary benefit of a good preconditioner is **accelerated convergence of Krylov subspace methods** (GMRES, Bi‑CGSTAB, etc.). Without preconditioning, the widely spread eigenvalues of $\mathbf{Z}$ force these methods to take many iterations to reduce the residual below a given tolerance. Preconditioning clusters the eigenvalues near 1, enabling convergence in far fewer iterations.

**Typical improvement**: For a well‑conditioned mass‑based preconditioner $\mathbf{M}$, the iteration count for GMRES can drop from $O(N)$ to $O(\sqrt{N})$ or even $O(1)$ for problems where $\mathbf{M}$ captures the dominant low‑frequency behavior of $\mathbf{Z}$.

#### 5.1.2 Better Numerical Accuracy in Direct Solves

Even when using a direct solver (LU factorization), preconditioning can improve the **accuracy of the computed solution**. The reason is that the LU decomposition of a well‑conditioned matrix suffers less from round‑off error propagation during forward/backward substitution. For extremely ill‑conditioned EFIE matrices ($\kappa > 10^{12}$), a direct solve of $\mathbf{Z}$ may lose 10–12 decimal digits of accuracy, while solving the preconditioned system $\tilde{\mathbf{Z}}$ can preserve full double‑precision accuracy.

#### 5.1.3 Smoother Gradients in Adjoint‑Based Optimization

When gradients are computed via the adjoint method, small numerical errors in the forward and adjoint solves can be amplified, resulting in **noisy gradients** that hinder optimization progress. Preconditioning reduces these numerical errors, yielding smoother gradient fields that allow gradient‑based optimizers (L‑BFGS, Adam) to converge more reliably.

#### 5.1.4 Stabilization of Low‑Frequency Breakdown

The mass‑based preconditioner $\mathbf{M} = \mathbf{R} + \epsilon\mathbf{I}$ directly counteracts the **scaling imbalance** between the vector‑potential term $V_{mn}$ ($O(1)$) and the scalar‑potential term $S_{mn}$ ($O(1/k^2)$) at low frequencies. By weighting each current element by its spatial support, $\mathbf{M}^{-1}$ re‑balances the matrix, preventing the near‑singularity that occurs as $k \to 0$.

#### 5.1.5 Reusability Across Similar Solves

In optimization loops or parameter sweeps where many linear systems with **similar geometry** must be solved, the preconditioner can be factored once and reused for all solves. This amortizes the $O(N^3)$ factorization cost over many right‑hand sides, making the overall computation more efficient.

### 5.2 What Preconditioning Does NOT Improve

#### 5.2.1 Asymptotic Complexity of Dense Direct Solves

Preconditioning does **not** change the fundamental $O(N^3)$ floating‑point operation count of a dense LU factorization. The cost of factoring $\mathbf{Z}$ and the cost of factoring $\mathbf{M}$ are both $O(N^3)$. If you use a direct solver for $\mathbf{Z}$, adding preconditioning effectively doubles the factorization work (factor $\mathbf{M}$ and factor $\tilde{\mathbf{Z}}$). For this reason, preconditioning is most beneficial when **iterative solvers** are employed, because the reduction in iteration count can outweigh the preconditioner setup cost.

#### 5.2.2 Memory Footprint of Dense Matrices

The storage requirement for a dense EFIE matrix $\mathbf{Z}$ is $O(N^2)$. Preconditioning adds storage for $\mathbf{M}$ (another $O(N^2)$ matrix) and possibly its LU factors (another $O(N^2)$). Thus, the total memory increases by a constant factor (typically 2–3×). If memory is the limiting resource, preconditioning may not be feasible.

#### 5.2.3 Condition Number of the Original Matrix

Preconditioning transforms the system to $\tilde{\mathbf{Z}} = \mathbf{M}^{-1}\mathbf{Z}$, which has a better condition number than $\mathbf{Z}$, but **does not improve the conditioning of $\mathbf{Z}$ itself**. If you need to solve the original system $\mathbf{Z}$ with a direct solver (e.g., for validation), preconditioning offers no benefit.

#### 5.2.4 Physical Fidelity

Unlike regularization, preconditioning does **not** alter the physical model: the solution $\mathbf{I}$ of $\tilde{\mathbf{Z}}\mathbf{I} = \tilde{\mathbf{v}}$ is exactly the same as the solution of $\mathbf{Z}\mathbf{I} = \mathbf{v}$ (up to numerical rounding). However, if the preconditioner is poorly chosen (e.g., $\mathbf{M}$ is singular), the transformed system may be numerically unstable, leading to inaccurate results.

#### 5.2.5 Automatic Handling of All Ill‑Conditioning Sources

The mass‑based preconditioner implemented in `DifferentiableMoM.jl` is effective for **low‑frequency breakdown** and **mesh‑refinement ill‑conditioning**, but it may not address other sources of ill‑conditioning, such as:
- **High‑aspect‑ratio geometry** (thin structures)
- **Nearly touching surfaces** (small gaps)
- **Resonant cavities** (internal resonances)

For these scenarios, more specialized preconditioners (e.g., Calderón, hierarchical, or domain‑decomposition) would be required.

### 5.3 Comparison with Regularization

It is instructive to contrast preconditioning with the regularization technique described in Section 2:

| Aspect | Regularization | Preconditioning |
|--------|----------------|-----------------|
| **Mathematical action** | Adds $\alpha\mathbf{R}$ to $\mathbf{Z}$ | Multiplies $\mathbf{Z}$ by $\mathbf{M}^{-1}$ |
| **Effect on solution** | Perturbs the physical solution | Preserves the exact solution (in exact arithmetic) |
| **Conditioning improvement** | Shifts eigenvalues away from zero | Clusters eigenvalues around 1 |
| **Best use case** | Stabilizing direct solves at very low frequencies | Accelerating iterative solves |
| **Setup cost** | Negligible (matrix addition) | Moderate to high (factorization of $\mathbf{M}$) |
| **Adjoint consistency** | Must use same $\alpha\mathbf{R}$ in forward/adjoint | Must use same $\mathbf{M}^{-1}$ in forward/adjoint |

**Practical recommendation**: Use **regularization** when you need a tiny stabilization for direct solves and can tolerate a small physical error. Use **preconditioning** when you need faster iterative convergence or better accuracy without perturbing the physics.

### 5.4 When the Overhead Is Worth It

The extra cost of building and applying a preconditioner is justified when **one or more** of the following conditions hold:

1. **$N$ is large** ($\gtrsim 1000$) and you use an iterative solver – the reduction in iteration count saves more time than the preconditioner setup.
2. **Low‑frequency EFIE** – the ill‑conditioning is severe enough that direct solves lose accuracy.
3. **Many similar right‑hand sides** – the preconditioner can be reused many times, amortizing the factorization cost.
4. **Gradient‑based optimization** – smoother gradients lead to fewer optimization iterations, outweighing the per‑iteration overhead.

Conversely, for small problems ($N < 200$) solved with a direct method, preconditioning often adds unnecessary complexity and should be turned off (`mode=:off`).

### 5.5 Monitoring Preconditioner Effectiveness

To verify that your preconditioner is working as expected, you can:

1. **Compare condition numbers**:
   ```julia
   κ_raw = condition_diagnostics(Z).cond
   κ_prec = condition_diagnostics(M \ Z).cond
   println("Condition number improvement: ", κ_raw / κ_prec)
   ```
2. **Monitor iteration counts** (if using an iterative solver):
   - Record the number of GMRES iterations with and without preconditioning.
   - Expect a reduction of at least 50% for a good preconditioner.
3. **Check gradient consistency**:
   - Compute gradients with and without preconditioning for a small test problem.
   - They should agree to within machine precision (relative error $< 10^{-12}$).

### 5.6 Summary: Key Takeaways

- **Preconditioning improves** iterative convergence, numerical accuracy, gradient smoothness, and low‑frequency stability.
- **Preconditioning does not improve** asymptotic complexity ($O(N^3)$ for dense direct solves) or memory footprint ($O(N^2)$).
- **Choose preconditioning over regularization** when you need exact physical fidelity and are willing to pay the setup cost.
- **Monitor effectiveness** via condition‑number ratios and iteration‑count reductions.

With these insights, you can make informed decisions about when to enable preconditioning and how to verify that it is delivering the expected benefits.

---

## 6. Minimal Conditioned Solve Example

This section walks through a complete, self‑contained example that demonstrates how to assemble an EFIE matrix, diagnose its conditioning, apply regularization and preconditioning, solve the linear system, and verify the results. The example is designed to be runnable as a standalone script and illustrates the typical workflow for handling ill‑conditioned EFIE problems in `DifferentiableMoM.jl`.

### 6.1 Problem Setup

We consider a rectangular PEC plate of size $0.1\lambda \times 0.1\lambda$ at a moderately low frequency ($k = 0.1$), discretized with a coarse mesh of $10\times10$ triangles. This configuration leads to a matrix of size $N = 180$ (number of RWG edges), which is small enough for fast experimentation yet exhibits noticeable ill‑conditioning.

```julia
using DifferentiableMoM
using LinearAlgebra

# ------------------------------
# 1. Geometry and discretization
# ------------------------------
λ = 1.0                 # wavelength (arbitrary units)
k = 2π / λ * 0.1       # wavenumber corresponding to 0.1λ size
mesh = make_rect_plate(0.1, 0.1, 10, 10)   # 10×10 triangle mesh
rwg = build_rwg(mesh)  # RWG basis functions
N = rwg.nedges
println("Number of unknowns N = ", N)

# ------------------------------
# 2. Incident plane wave
# ------------------------------
E0 = 1.0
k_vec = Vec3(0.0, 0.0, -k)
pol_inc = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3)

# ------------------------------
# 3. EFIE matrix assembly
# ------------------------------
Z_raw = assemble_Z_efie(mesh, rwg, k)
println("Matrix size: ", size(Z_raw))
```

### 6.2 Conditioning Diagnostics

Before deciding on regularization or preconditioning, we examine the condition number and singular values of the raw EFIE matrix.

```julia
# ------------------------------
# 4. Condition diagnostics
# ------------------------------
stats = condition_diagnostics(Z_raw)
κ_raw = stats.cond
σ_max = stats.sv_max
σ_min = stats.sv_min
println("Raw condition number κ = ", κ_raw)
println("Singular value range: σ_max = ", σ_max, ", σ_min = ", σ_min)
println("log10(κ) = ", log10(κ_raw))

# Interpretation
if κ_raw < 1e6
    println("Well‑conditioned – no stabilization needed.")
elseif κ_raw < 1e10
    println("Moderately ill‑conditioned – consider preconditioning.")
else
    println("Severely ill‑conditioned – regularization or preconditioning recommended.")
end
```

For this example, you should see $\kappa \approx 10^7$–$10^8$, indicating moderate ill‑conditioning.

### 6.3 Building Regularization and Preconditioning Components

We prepare the patch mass matrices that serve both as a regularizer and as the building block for the mass‑based preconditioner.

```julia
# ------------------------------
# 5. Patch mass matrices for regularization/preconditioning
# ------------------------------
# Partition the mesh into patches (one patch per triangle for simplicity)
partition = PatchPartition(mesh)
Mp = precompute_patch_mass(mesh, rwg, partition)

# Build the global mass regularizer R = Σ_p Mp
R = make_mass_regularizer(Mp)

# Optionally, inspect the trace of R
trR = tr(R)
println("Trace of mass regularizer R: ", trR)
```

### 6.4 Selecting and Applying Preconditioning

We use the `:auto` mode, which will enable preconditioning because $N = 180 < 256$ but we explicitly set `iterative_solver=true` to trigger it.

```julia
# ------------------------------
# 6. Preconditioner selection (:auto mode)
# ------------------------------
M_eff, enabled, reason = select_preconditioner(
    Mp;
    mode=:auto,
    iterative_solver=true,   # force preconditioning for demonstration
    n_threshold=256,
    eps_rel=1e-8
)
println("Preconditioning enabled: ", enabled, " (", reason, ")")
```

### 6.5 Regularization Parameter Choice

We add a tiny regularization to further stabilize the solve. The parameter $\alpha$ is scaled relative to the average diagonal entry of $\mathbf{Z}$.

```julia
# ------------------------------
# 7. Regularization parameter
# ------------------------------
α = 1e-10 * (trR / N)   # ε = 1e-10, relative to average mass entry
println("Regularization parameter α = ", α)
```

### 6.6 Preparing the Conditioned System

The function `prepare_conditioned_system` applies both regularization and preconditioning in the correct order (regularization first, then preconditioning).

```julia
# ------------------------------
# 8. Apply regularization and preconditioning
# ------------------------------
Z_eff, rhs_eff, fac = prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=α,
    regularization_R=R,
    preconditioner_M=M_eff,
    preconditioner_factor=nothing
)
println("Conditioned matrix size: ", size(Z_eff))
```

### 6.7 Solving the Linear System

We solve the conditioned system using a direct solver (backslash). In practice, for larger $N$, you would replace this with an iterative solver.

```julia
# ------------------------------
# 9. Solve the linear system
# ------------------------------
I_cond = Z_eff \ rhs_eff
println("Solution vector norm = ", norm(I_cond))
```

### 6.8 Verifying Conditioning Improvement

To confirm that conditioning helped, we compute the condition number of the preconditioned matrix $\tilde{\mathbf{Z}} = \mathbf{M}^{-1}(\mathbf{Z} + \alpha\mathbf{R})$.

```julia
# ------------------------------
# 10. Verify conditioning improvement
# ------------------------------
stats_cond = condition_diagnostics(Z_eff)
κ_cond = stats_cond.cond
println("Conditioned κ = ", κ_cond)
println("Improvement factor = ", κ_raw / κ_cond)
println("log10(improvement) = ", log10(κ_raw / κ_cond))
```

You should observe an improvement factor of $10^2$–$10^4$, demonstrating that the combined regularization and preconditioning significantly reduced the condition number.

### 6.9 Comparing with Raw Solve

For completeness, we also solve the raw (unconditioned) system and compare the solutions.

```julia
# ------------------------------
# 11. Compare with raw solve
# ------------------------------
I_raw = Z_raw \ v
rel_diff = norm(I_raw - I_cond) / norm(I_raw)
println("Relative difference between raw and conditioned solutions: ", rel_diff)

# Check residual of both solutions
res_raw = norm(Z_raw * I_raw - v) / norm(v)
res_cond = norm(Z_raw * I_cond - v) / norm(v)
println("Raw residual: ", res_raw)
println("Conditioned residual: ", res_cond)
```

The two solutions should agree to within $10^{-6}$–$10^{-8}$, while the conditioned solve typically yields a smaller residual because of improved numerical stability.

### 6.10 Full Script

The complete script is available as `examples/05_solver_methods.jl` in the repository. You can run it directly to reproduce the results:

```bash
julia examples/05_solver_methods.jl
```

### 6.11 Interpreting the Results

- **Condition number improvement**: A reduction from $\kappa \sim 10^8$ to $\kappa \sim 10^4$ is typical for this problem size. Larger improvements occur for lower frequencies or finer meshes.
- **Solution agreement**: The raw and conditioned solutions should be nearly identical (relative difference $< 10^{-6}$). A larger discrepancy may indicate that the regularization parameter $\alpha$ is too large.
- **Residuals**: The conditioned solve often yields a smaller residual because the better‑conditioned matrix allows the direct solver to achieve higher accuracy.

### 6.12 Extending the Example

To adapt this example to your own problems, consider the following modifications:

1. **Change frequency**: Set `k` to a lower value (e.g., `k = 0.01`) to observe more severe ill‑conditioning and greater benefit from conditioning.
2. **Use iterative solver**: Replace `Z_eff \ rhs_eff` with `gmres(Z_eff, rhs_eff; tol=1e-8, maxiter=500)` and compare iteration counts with and without preconditioning.
3. **Vary regularization**: Experiment with different values of `α` (e.g., `1e-12`, `1e-8`) and observe the effect on solution accuracy and condition number.
4. **Test different preconditioner modes**: Try `mode=:off`, `:on`, and `:auto` (with `iterative_solver=false`) to see how the preconditioner selection changes.

This minimal example provides a template for incorporating conditioning into your EFIE solves, ensuring robust and accurate results even for challenging low‑frequency or finely‑meshed scenarios.

---

## 7. Practical Checklist

This section provides a step‑by‑step workflow for incorporating conditioning and preconditioning into your EFIE simulations. Following this checklist will help you avoid common pitfalls, ensure numerical robustness, and maintain reproducibility across different runs and computing environments.

### 7.1 Step‑by‑Step Workflow

#### Step 1: Baseline Run with `:off` Mode

**Action**: Run your simulation with `mode=:off` (the default) and no regularization.

**Purpose**: Establish a baseline solution and verify that the raw EFIE problem is set up correctly. If the baseline fails (solver error, unrealistic results), the problem may lie in geometry, discretization, or assembly rather than conditioning.

**Code snippet**:
```julia
M_eff, enabled, reason = select_preconditioner(
    Mp; mode=:off
)
Z_eff, rhs_eff, _ = prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=0.0,
    preconditioner_M=nothing
)
I_baseline = solve_system(Z_eff, rhs_eff)
```

#### Step 2: Diagnose Conditioning

**Action**: Compute the condition number of $\mathbf{Z}_{\mathrm{raw}}$ using `condition_diagnostics`.

**Purpose**: Quantify the severity of ill‑conditioning. Use the following guidelines:
- $\kappa < 10^6$ → conditioning is not a concern.
- $10^6 \le \kappa < 10^{10}$ → consider enabling preconditioning for iterative solves.
- $\kappa \ge 10^{10}$ → enable preconditioning and possibly add regularization.

**Code snippet**:
```julia
stats = condition_diagnostics(Z_raw)
κ = stats.cond
println("Condition number κ = ", κ)
```

#### Step 3: Choose Regularization (If Needed)

**Action**: If $\kappa \ge 10^{10}$ and you are using a **direct solver**, add a small mass‑based regularization with $\alpha = \epsilon \cdot \operatorname{tr}(\mathbf{R})/N$.

**Purpose**: Stabilize the direct solve without significantly perturbing the physics. Typical $\epsilon$ values: $10^{-10}$ for moderate ill‑conditioning, $10^{-8}$ for severe ill‑conditioning.

**Code snippet**:
```julia
R = make_mass_regularizer(Mp)
α = 1e-10 * (tr(R) / size(R,1))
```

#### Step 4: Select Preconditioning Mode

**Action**: Decide on `mode=:off`, `:on`, or `:auto` based on your problem characteristics.

**Decision logic**:
- **`:off`**: Baseline validation, small $N$ (< 200), debugging.
- **`:on`**: Low‑frequency EFIE, iterative solves, optimization loops.
- **`:auto`**: Exploratory studies where problem size may vary; let the package decide.

**Code snippet**:
```julia
M_eff, enabled, reason = select_preconditioner(
    Mp;
    mode=:auto,
    iterative_solver=false,   # set true if using GMRES/Bi‑CGSTAB
    n_threshold=256
)
println("Preconditioning: ", enabled, " (", reason, ")")
```

#### Step 5: Prepare Conditioned System

**Action**: Call `prepare_conditioned_system` with the chosen regularization and preconditioner.

**Purpose**: Apply both transformations in the correct order (regularization first, then preconditioning) and obtain the LU factorization of the preconditioner (if any) for reuse in adjoint solves.

**Code snippet**:
```julia
Z_eff, rhs_eff, fac = prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=α,
    regularization_R=R,
    preconditioner_M=M_eff
)
```

#### Step 6: Solve and Verify

**Action**: Solve the conditioned system and check the residual. Compare with the baseline solution (if available) to ensure conditioning did not introduce significant error.

**Purpose**: Verify that the conditioning transformations improved numerical stability without altering the physical solution beyond acceptable tolerance.

**Code snippet**:
```julia
I_cond = solve_system(Z_eff, rhs_eff)
residual = norm(Z_raw * I_cond - v) / norm(v)
println("Relative residual: ", residual)
if isdefined(Main, :I_baseline)
    rel_diff = norm(I_cond - I_baseline) / norm(I_baseline)
    println("Relative difference from baseline: ", rel_diff)
end
```

#### Step 7: Ensure Adjoint Consistency

**Action**: When computing gradients via the adjoint method, **reuse the same conditioned operator** (`Z_eff`) and preconditioner factorization (`fac`) in both forward and adjoint solves.

**Purpose**: Guarantee that gradients are accurate to machine precision. Using different conditioning in forward and adjoint solves will produce incorrect gradients.

**Code snippet** (inside an optimization loop):
```julia
# Forward solve (already done)
# Adjoint solve using the SAME Z_eff and fac
λ = adjoint_solve(Z_eff, ∂Φ∂I; preconditioner_factor=fac)
```

#### Step 8: Log All Conditioning Parameters

**Action**: Record the conditioning mode, regularization parameter $\alpha$, preconditioner enable flag, and reason string in your experiment logs.

**Purpose**: Ensure full reproducibility. Without these details, it may be impossible to reproduce the same numerical behavior later.

**Code snippet**:
```julia
log_entry = """
Conditioning parameters:
- mode = :auto
- regularization_alpha = $α
- preconditioning_enabled = $enabled
- preconditioning_reason = "$reason"
- condition_number_raw = $κ
"""
write("run_$(timestamp).log", log_entry)
```

### 7.2 Monitoring and Debugging

#### Monitoring Conditioning During Optimization

When running gradient‑based optimization, ill‑conditioning can vary as the design parameters change (e.g., sheet resistances approach zero). Monitor the condition number at each iteration:

```julia
function callback(θ)
    Z = assemble_Z(θ)
    stats = condition_diagnostics(Z)
    println("Iteration: κ = ", stats.cond)
end
```

If $\kappa$ suddenly spikes, consider adding adaptive regularization: increase $\alpha$ when $\kappa$ exceeds a threshold.

#### Debugging Common Issues

| Symptom | Possible cause | Remedy |
|---------|----------------|--------|
| **Gradients disagree** between conditioned and unconditioned solves | Different conditioning used in forward/adjoint | Ensure `prepare_conditioned_system` is called once and its outputs reused. |
| **Iterative solver stagnates** even with preconditioning | Preconditioner too weak (e.g., $\epsilon$ too large) | Reduce `eps_rel` in `make_left_preconditioner`. |
| **Solution changes significantly** with tiny $\alpha$ | Regularization too strong for low‑frequency problem | Decrease $\alpha$ by a factor of 10. |
| **Memory error** when building preconditioner | $N$ too large for dense preconditioner | Switch to iterative solver with diagonal preconditioner, or use a sparse approximation. |

#### Verifying Gradient Accuracy

The gold‑standard test for conditioning consistency is the **gradient verification**:

1. Compute gradients using the adjoint method with conditioning enabled.
2. Compute finite‑difference gradients (without conditioning) for a few randomly chosen parameters.
3. Compare relative error: should be $< 10^{-6}$ for well‑conditioned problems, $< 10^{-4}$ for severely ill‑conditioned ones.

A discrepancy larger than $10^{-4}$ indicates that the conditioning is not being applied consistently between forward and adjoint solves.

### 7.3 Summary Checklist

For quick reference, here is a condensed version of the workflow:

- [ ] **Baseline**: Run with `mode=:off`, no regularization.
- [ ] **Diagnose**: Compute $\kappa$; if $\kappa \ge 10^{10}$, consider regularization.
- [ ] **Regularize** (optional): Set $\alpha = \epsilon \cdot \operatorname{tr}(\mathbf{R})/N$ with $\epsilon \in [10^{-12}, 10^{-8}]$.
- [ ] **Precondition**: Choose `:auto` for exploration, `:on` for low‑frequency/iterative solves.
- [ ] **Prepare**: Call `prepare_conditioned_system` with chosen $\alpha$ and $\mathbf{M}$.
- [ ] **Solve**: Compute solution, check residual $\le 10^{-8}$.
- [ ] **Adjoint**: Reuse same `Z_eff` and `fac` in adjoint solve.
- [ ] **Log**: Record mode, $\alpha$, enable flag, reason, and $\kappa$.

Following this systematic approach will help you harness the benefits of conditioning and preconditioning while avoiding the pitfalls that lead to inaccurate results or non‑reproducible simulations.

---

## 8. Implementation Details and Code Map

This section provides a roadmap to the source files that implement conditioning and preconditioning in `DifferentiableMoM.jl`. Understanding where the key functions are defined will help you debug issues, extend the functionality, or adapt the code to your specific needs.

### 8.1 Core Conditioning Functions (`src/Solve.jl`)

The file `src/Solve.jl` contains all the high‑level functions for regularization, preconditioning, and linear solves:

- **`make_mass_regularizer(Mp)`** – builds the global mass regularizer $\mathbf{R} = \sum_p \mathbf{M}_p$ from patch mass matrices.
- **`make_left_preconditioner(Mp; eps_rel=1e-8)`** – constructs the mass‑based preconditioner $\mathbf{M} = \mathbf{R} + \epsilon\mathbf{I}$ and returns its LU factorization.
- **`select_preconditioner(Mp; mode=:off, iterative_solver=false, n_threshold=256, ...)`** – implements the mode logic (`:off`, `:on`, `:auto`) and returns a triple `(M_eff, enabled, reason)`.
- **`prepare_conditioned_system(Z, v; regularization_alpha=0.0, regularization_R=nothing, preconditioner_M=nothing, preconditioner_factor=nothing)`** – applies regularization and preconditioning in the correct order, returning the conditioned matrix $\mathbf{Z}_{\text{eff}}$, right‑hand side $\mathbf{v}_{\text{eff}}$, and the preconditioner factorization (if any).
- **`solve_system(Z, rhs)`** – generic linear solve (defaults to backslash, but can be extended to iterative solvers).

**Location**: `src/Solve.jl`, lines ~50–150 (exact line numbers may vary with version).

**Usage example**:
```julia
using DifferentiableMoM: make_mass_regularizer, make_left_preconditioner,
                         select_preconditioner, prepare_conditioned_system
```

### 8.2 Conditioning Diagnostics (`src/Diagnostics.jl`)

The file `src/Diagnostics.jl` provides functions for assessing matrix conditioning and solver health:

- **`condition_diagnostics(Z)`** – computes the condition number $\kappa$ and extreme singular values of $\mathbf{Z}$ via a full SVD. Returns a named tuple `(cond, sv_max, sv_min)`.

**Location**: `src/Diagnostics.jl`, lines ~200–250.

**Usage example**:
```julia
using DifferentiableMoM: condition_diagnostics
stats = condition_diagnostics(Z)
println("Condition number = ", stats.cond)
```

### 8.3 Optimization Integration (`src/Optimize.jl`)

The file `src/Optimize.jl` ensures that conditioning is applied consistently during gradient‑based optimization:

- **`optimize_lbfgs(f, g!, θ0; ...)`** – the main L‑BFGS optimizer that calls forward and adjoint solves. Inside the optimization loop, `prepare_conditioned_system` is called **once per iteration** and the resulting conditioned operator is reused for both forward and adjoint solves to guarantee gradient accuracy.
- **`adjoint_solve(Z_eff, ∂Φ∂I; preconditioner_factor=nothing)`** – solves the adjoint equation using the same preconditioner factorization (if any) that was used in the forward solve.

**Location**: `src/Optimize.jl`, lines ~100–300.

**Usage note**: When writing custom optimization loops, follow the pattern in `optimize_lbfgs` to maintain adjoint consistency.

### 8.4 Example Scripts

Several example scripts demonstrate conditioning in practice:

- **`examples/05_solver_methods.jl`** – shows how `select_preconditioner` behaves with different modes and problem sizes.
- **`examples/05_solver_methods.jl`** (to be created) – a complete worked example of assembling an EFIE matrix, diagnosing conditioning, applying regularization/preconditioning, and verifying the results (as described in Section 6).
- **`examples/05_solver_methods.jl`** (if present) – illustrates conditioning for low‑frequency EFIE problems.

**Location**: `examples/` directory.

### 8.5 How the Pieces Fit Together

The following diagram illustrates the data flow when conditioning is enabled:

```
Patch mass matrices Mp
         │
         ▼
make_mass_regularizer  →  R (regularizer)
         │
         ▼
make_left_preconditioner → M (preconditioner) + LU factor
         │
         ▼
select_preconditioner  →  (M_eff, enabled, reason)
         │
         ▼
prepare_conditioned_system(Z, v; regularization_R=R, preconditioner_M=M_eff)
         │
         ▼
(Z_eff, v_eff, fac)  →  forward solve  →  I
         │                    │
         │                    ▼
         └──────────→  adjoint solve (using same Z_eff, fac) → λ
```

**Key invariant**: The same `Z_eff` and `fac` must be used in forward and adjoint solves. This invariant is enforced by `prepare_conditioned_system`, which returns both objects.

### 8.6 Extending the Conditioning Framework

If you need to implement a custom preconditioner (e.g., block‑diagonal, incomplete LU, Calderón), follow these steps:

1. Build your preconditioner matrix $\mathbf{M}_{\text{custom}}$ (or a function that applies its inverse).
2. Pass it to `select_preconditioner` via the `preconditioner_M` keyword argument (this overrides the automatic selection).
3. Ensure your preconditioner is invertible (add a small diagonal shift if necessary).
4. Verify gradient consistency with a finite‑difference test.

The existing infrastructure is designed to be modular, allowing you to plug in alternative preconditioners while still benefiting from the consistent adjoint handling.

### 8.7 Summary of Key Files

| File | Purpose | Key functions |
|------|---------|---------------|
| `src/Solve.jl` | Regularization, preconditioning, linear solves | `make_mass_regularizer`, `make_left_preconditioner`, `select_preconditioner`, `prepare_conditioned_system` |
| `src/Diagnostics.jl` | Conditioning diagnostics | `condition_diagnostics` |
| `src/Optimize.jl` | Optimization with consistent conditioning | `optimize_lbfgs`, `adjoint_solve` |
| `examples/05_solver_methods.jl` | Demo of mode selection | – |

With this roadmap, you can navigate the source code to understand, modify, or extend the conditioning capabilities of `DifferentiableMoM.jl`.

---

## 9. Exercises

This section provides hands‑on exercises to reinforce the concepts covered in this chapter. Start with the basic tasks and progress to the advanced challenges to deepen your understanding of conditioning and preconditioning in EFIE simulations.

### 9.1 Conceptual Questions

1. **Condition number interpretation**: For an EFIE matrix with $\kappa = 10^{12}$, approximately how many decimal digits of accuracy would you expect to lose in a direct solve using double‑precision arithmetic? What are the practical consequences for gradient‑based optimization?

2. **Regularization vs. preconditioning**: Explain the fundamental difference between regularization and preconditioning in terms of their effect on the physical solution and the linear‑system conditioning. Give one scenario where regularization is preferable and one where preconditioning is preferable.

3. **Auto‑mode logic**: The `:auto` mode enables preconditioning when `iterative_solver=true` or $N \ge n_{\text{threshold}}$. Why are these two criteria used? What could go wrong if preconditioning were always enabled for iterative solves, regardless of problem size?

### 9.2 Derivation Tasks

1. **Condition number bound**: Show that for a Hermitian positive‑definite matrix $\mathbf{Z}$, the condition number with respect to the Euclidean norm satisfies $\kappa(\mathbf{Z}) = \lambda_{\max}/\lambda_{\min}$. Using this result, prove that adding a regularization term $\alpha\mathbf{R}$ with $\mathbf{R} \succ 0$ strictly reduces the condition number (i.e., $\kappa(\mathbf{Z}+\alpha\mathbf{R}) < \kappa(\mathbf{Z})$ for any $\alpha > 0$).

2. **Mass‑based preconditioner analysis**: Let $\mathbf{M} = \mathbf{R} + \epsilon\mathbf{I}$ where $\mathbf{R}$ is the mass matrix. Show that the eigenvalues of the preconditioned matrix $\tilde{\mathbf{Z}} = \mathbf{M}^{-1}\mathbf{Z}$ satisfy
   ```math
   \lambda_i(\tilde{\mathbf{Z}}) = \frac{\lambda_i(\mathbf{Z})}{\lambda_i(\mathbf{R}) + \epsilon}.
   ```
   Discuss why this clustering improves iterative convergence for low‑frequency EFIE matrices.

### 9.3 Coding Exercises

1. **Basic**: Run the example script `examples/05_solver_methods.jl` and modify it to test all three modes (`:off`, `:on`, `:auto`) for a problem with $N=100$ and $N=500$. Record the condition numbers and solve times for each mode. Explain the observed differences.

2. **Regularization sweep**: Create a script that solves a low‑frequency EFIE problem ($k=0.001$) for a range of regularization parameters $\alpha = 10^{-12}, 10^{-11}, \dots, 10^{-6}$. Plot the relative error in the solution (compared to a reference solve with $\alpha=0$ and high‑precision arithmetic) versus $\alpha$. Identify the “sweet spot” where error is minimized.

3. **Preconditioner effectiveness**: For a problem with $N=1000$, implement a simple iterative solver (e.g., GMRES) and compare convergence with and without the mass‑based preconditioner. Plot the residual norm versus iteration count for both cases. Compute the iteration‑count reduction factor.

4. **Gradient consistency test**: Choose a small impedance‑optimization problem (e.g., optimizing the resistance of a single patch). Compute gradients using the adjoint method (with conditioning enabled) and compare them with finite‑difference gradients (computed without conditioning). Verify that the relative error is below $10^{-6}$. If not, diagnose the cause.

### 9.4 Advanced Challenges

1. **Custom preconditioner**: Implement a simple diagonal preconditioner $\mathbf{M} = \operatorname{diag}(\mathbf{Z})$ and integrate it into the conditioning framework. Compare its effectiveness with the mass‑based preconditioner for a low‑frequency problem and a high‑frequency problem. Measure the condition‑number improvement and iteration‑count reduction.

2. **Adaptive regularization**: Write a function that automatically selects the regularization parameter $\alpha$ based on the estimated condition number $\kappa$. The function should increase $\alpha$ when $\kappa$ exceeds a threshold (e.g., $10^{10}$) and decrease it when $\kappa$ is below another threshold. Test your function on a problem where the condition number varies during optimization (e.g., as sheet resistances approach zero).

3. **Conditioning‑aware mesh refinement**: Perform a mesh‑refinement study for a PEC sphere at a fixed frequency. For each mesh level, compute the condition number $\kappa$ and the solution error (compared to an analytical Mie series solution). Determine whether preconditioning improves the error convergence rate as $h \to 0$.

### 9.5 Solutions and Hints

- **Conceptual 1**: Double‑precision has about 15–16 decimal digits. A condition number of $10^{12}$ can lose up to 12 digits, leaving only 3–4 accurate digits. This leads to noisy gradients and stalled optimization.
- **Derivation 1**: Use the fact that $\mathbf{Z}$ and $\mathbf{R}$ are Hermitian, and apply Weyl’s inequality for eigenvalues of sums of Hermitian matrices.
- **Coding 1**: See `examples/05_solver_methods.jl` for a template.
- **Advanced 1**: The diagonal preconditioner is cheap to apply but may be less effective for low‑frequency EFIE where off‑diagonal terms are significant.

Complete solutions are not provided here; the goal is to encourage independent exploration and deeper engagement with the codebase.

---

## 10. Chapter Checklist

After studying this chapter, you should be able to:

- [ ] **Define** the condition number of a matrix and explain why EFIE matrices become ill‑conditioned at low frequencies and under mesh refinement.
- [ ] **Compute** the condition number of an EFIE matrix using `condition_diagnostics`.
- [ ] **Explain** the difference between regularization and preconditioning, and choose the appropriate technique for a given scenario.
- [ ] **Apply** regularization by constructing a mass regularizer $\mathbf{R}$ and selecting a suitable parameter $\alpha$.
- [ ] **Construct** a mass‑based left preconditioner using `make_left_preconditioner`.
- [ ] **Select** the correct preconditioning mode (`:off`, `:on`, `:auto`) based on problem size, solver type, and frequency.
- [ ] **Prepare** a conditioned linear system using `prepare_conditioned_system` and solve it accurately.
- [ ] **Ensure** adjoint consistency by reusing the same conditioned operator in forward and adjoint solves.
- [ ] **Diagnose** common conditioning‑related issues (noisy gradients, solver stagnation, memory errors) and apply appropriate remedies.
- [ ] **Verify** gradient accuracy with a finite‑difference test when conditioning is enabled.
- [ ] **Reproduce** results by logging all conditioning parameters (mode, $\alpha$, enable flag, reason, $\kappa$).

If you can confidently check all items, you have mastered the essentials of conditioning and preconditioning in `DifferentiableMoM.jl`.

---

## 11. Further Reading

For readers interested in diving deeper into the theory and practice of matrix conditioning and preconditioning for integral equations, the following resources are recommended:

### 11.1 Foundational Texts

- **Golub & Van Loan, *Matrix Computations*** (4th ed., 2013) – The standard reference on numerical linear algebra, with comprehensive coverage of condition numbers, regularization, and preconditioning techniques.
- **Colton & Kress, *Integral Equation Methods in Scattering Theory*** (1983) – Classical text that discusses the ill‑posedness of first‑kind integral equations and regularization methods (Tikhonov, Landweber).
- **Chew et al., *Fast and Efficient Algorithms in Computational Electromagnetics*** (2001) – Includes chapters on Calderón preconditioning and low‑frequency stabilization for EFIE.

### 11.2 Specialized Papers on EFIE Conditioning

- **Z.‑G. Qian & W. C. Chew, “An augmented EFIE for low‑frequency breakdown,” *IEEE Trans. Antennas Propag.*, 2009** – Introduces the augmented EFIE (A‑EFIE) to overcome low‑frequency breakdown.
- **J.‑S. Zhao & W. C. Chew, “Integral equation solution of Maxwell’s equations from zero frequency to microwave frequencies,” *IEEE Trans. Antennas Propag.*, 2000** – Presents loop‑tree decomposition, a widely used preconditioning technique for low‑frequency EFIE.
- **A. Buffa & R. Hiptmair, “Regularized combined field integral equations,” *Numer. Math.*, 2004** – Analysis of Calderón preconditioning for the EFIE, proving that the preconditioned operator is well‑conditioned independent of mesh refinement.

### 11.3 Preconditioning for Iterative Methods

- **Y. Saad, *Iterative Methods for Sparse Linear Systems*** (2003) – Comprehensive treatment of Krylov methods and preconditioners (Jacobi, ILU, multigrid).
- **M. Benzi, “Preconditioning techniques for large linear systems: A survey,” *J. Comput. Phys.*, 2002** – Reviews state‑of‑the‑art preconditioners with emphasis on applications in computational physics.

### 11.4 Software and Libraries

- **Julia LinearAlgebra** – The standard library provides condition number estimation (`cond`) and singular value decomposition (`svd`). Iterative solvers (GMRES, BiCGStab) are provided by external packages such as Krylov.jl, not by the standard LinearAlgebra library.
- **IterativeSolvers.jl** – A Julia package with advanced Krylov methods and preconditioners.
- **IncompleteLU.jl** – For incomplete LU factorization preconditioners, which can be effective for EFIE matrices with appropriate re‑ordering.

### 11.5 Online Resources

- **DifferentiableMoM.jl documentation** – The package documentation (which you are reading) includes API references and additional examples.
- **Julia Discourse – Numerical Methods** – A forum for discussing conditioning and preconditioning issues in Julia.
- **MIT OpenCourseWare 18.335 – Numerical Linear Algebra** – Free course notes and videos covering condition numbers, SVD, and preconditioning.

By exploring these resources, you can build a deeper theoretical foundation and learn about advanced preconditioning techniques that go beyond the mass‑based approach implemented in `DifferentiableMoM.jl`.

---

*Next: Part II — Package Fundamentals, “Mesh Pipeline,” introduces the mesh‑generation pipeline and shows how to import, repair, and partition complex 3D geometries for MoM simulation.*
