# Multi-Level Fast Multipole Algorithm (MLFMA)

## Purpose

For very large problems ($N > 50{,}000$ unknowns), even H-matrix compression with ACA becomes limited by the $O(N \log^2 N)$ near-field storage cost. The Multi-Level Fast Multipole Algorithm (MLFMA) reduces both storage and matrix-vector product cost to $O(N \log N)$ by hierarchically decomposing electromagnetic interactions into near-field (directly computed) and far-field (computed via radiation patterns and translation operators) contributions.

This chapter explains the MLFMA implementation in `DifferentiableMoM.jl`: octree construction, radiation pattern computation, multi-level aggregation and disaggregation with per-$m$ spectral filters, translation operators, near-field preconditioning, and practical usage guidelines including accuracy considerations.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why MLFMA achieves $O(N \log N)$ complexity for EFIE matrix-vector products.
2. Describe the octree spatial decomposition and its role in the MLFMA hierarchy.
3. Understand radiation pattern computation and the 4-component vector formulation.
4. Explain aggregation (upward pass), translation (interaction), and disaggregation (downward pass).
5. Describe the per-$m$ spectral filter implementation and why separated filters fail for $m \neq 0$ modes.
6. Use `build_mlfma_operator` and `solve_scattering` with MLFMA in practice.
7. Apply accuracy guidelines: `leaf_lambda >= 1.0` for production (<1% error).
8. Configure reordered-ILU preconditioning for fast convergence.

---

## 1. The Scalability Problem

### 1.1 Comparison of Methods

| Method | Storage | Matvec | Assembly | Best for |
|--------|---------|--------|----------|----------|
| Dense | $O(N^2)$ | $O(N^2)$ | $O(N^2)$ | $N < 2{,}000$ |
| ACA H-matrix | $O(N \log^2 N)$ | $O(N \log^2 N)$ | $O(N \log^2 N)$ | $2{,}000 < N < 50{,}000$ |
| **MLFMA** | **$O(N \log N)$** | **$O(N \log N)$** | **$O(N \log N)$** | **$N > 50{,}000$** |

For a 1λ-resolution aircraft ($\sim$14λ wingspan), $N \approx 30{,}000$ unknowns. Dense assembly would require 14.7 GB of storage; MLFMA reduces this to ~7 GB (47% near-field fill at `leaf_lambda=1.0`).

### 1.2 The MLFMA Idea

The key insight is that far-field electromagnetic interactions can be computed via:
1. **Aggregation**: Compute radiation patterns for groups of sources
2. **Translation**: Shift these patterns between distant groups (cheap in the far field)
3. **Disaggregation**: Convert received patterns to fields at individual test points

By organizing basis functions into a spatial octree hierarchy, MLFMA performs these operations at multiple scales, achieving logarithmic complexity.

---

## 2. Octree Spatial Decomposition

### 2.1 Octree Construction

MLFMA builds an octree by recursively subdividing a bounding cube into $8$ children until leaf boxes contain a small number of basis functions. The tree depth is determined by the `leaf_lambda` parameter:

```math
\text{leaf edge} = \text{leaf\_lambda} \times \lambda, \quad \text{nLevels} = \lceil \log_2(\text{domain size} / \text{leaf edge}) \rceil + 1.
```

**Example**: For a 14λ aircraft with `leaf_lambda=1.0`:
- Leaf edge = 1.0λ
- Root edge = 16λ (smallest power of 2 containing the geometry)
- Tree depth = 5 levels (root → level 2 → 3 → 4 → leaf level 5)

### 2.2 Neighbor and Interaction Lists

At each level, boxes are classified into:
- **Neighbors**: boxes within ±1 in each $(i,j,k)$ index (including self) → near-field
- **Interaction list**: children of parent's neighbors, minus own neighbors → far-field translations

The near-field matrix $\mathbf{Z}_{\text{near}}$ stores interactions between neighboring boxes, while far-field interactions are computed via the MLFMA operators.

---

## 3. Radiation Patterns and Sampling

### 3.1 Far-Field Radiation from RWG Basis Functions

Each RWG basis function $\mathbf{f}_n$ radiates a far-field pattern. MLFMA samples this pattern on a spherical grid (Gauss-Legendre in $\theta$, uniform in $\phi$) with truncation order:

```math
L = \lfloor k d + 2.16 p^{2/3} (ke)^{1/3} \rfloor,
```

where $d = \sqrt{3} \cdot \text{edge}$ (box diagonal), $e = \text{edge}$, and $p$ is the precision parameter (default 3).

### 3.2 Four-Component Representation

The radiation pattern is stored as a 4-vector $(\mathbf{F}_{vec}, F_{scl})$:

```math
\mathbf{F}(\hat{\mathbf{k}}) = k^2 \int_S \mathbf{f}(\mathbf{r}) \, e^{-i k \hat{\mathbf{k}} \cdot \mathbf{r}} \, dS, \quad F_{scl}(\hat{\mathbf{k}}) = \frac{1}{k} \int_S (\nabla_s \cdot \mathbf{f}) \, e^{-i k \hat{\mathbf{k}} \cdot \mathbf{r}} \, dS.
```

This formulation separates the vector and scalar potentials, simplifying the EFIE computation:

```math
Z \mathbf{I} \approx -\frac{k^2 \eta_0}{16\pi^2} \sum_q w_q \left[ \mathbf{F}_{\text{rx}}^*(\hat{\mathbf{k}}_q) \cdot \mathbf{F}_{\text{tx}}(\hat{\mathbf{k}}_q) - F_{\text{rx,scl}}^*(\hat{\mathbf{k}}_q) F_{\text{tx,scl}}(\hat{\mathbf{k}}_q) \right].
```

---

## 4. Multi-Level Algorithm

### 4.1 Aggregation (Upward Pass)

Starting from the leaf level, aggregate source radiation patterns upward:
1. **Phase shift**: $e^{i k \hat{\mathbf{k}} \cdot \Delta \mathbf{r}}$ to move pattern center from child to parent
2. **Spectral interpolation**: Resample from child sampling ($L_{\text{child}}$) to parent sampling ($L_{\text{parent}}$)

**Key**: Aggregation uses **per-$m$ spectral filters** based on associated Legendre functions $P_l^m(\cos\theta)$ to correctly handle all azimuthal modes.

### 4.2 Translation (Interaction)

At each level, translate aggregated patterns between boxes in the interaction list using the translation operator:

```math
T_L(\hat{\mathbf{k}}; \mathbf{d}) = \sum_{l=0}^L (-i)^l (2l+1) \, h_l^{(2)}(k|\mathbf{d}|) \, P_l(\hat{\mathbf{k}} \cdot \hat{\mathbf{d}}),
```

where $h_l^{(2)}$ is the spherical Hankel function of the second kind and $P_l$ is the Legendre polynomial.

### 4.3 Disaggregation (Downward Pass)

Translate incoming patterns from parents to children:
1. **Phase shift**: $e^{-i k \hat{\mathbf{k}} \cdot \Delta \mathbf{r}}$ to move from parent to child center
2. **Spectral filter**: Band-limit from parent sampling to child sampling (prevents aliasing)

Again, **per-$m$ spectral filters** are essential for accuracy.

---

## 5. Per-$m$ Spectral Filters: The Correct Implementation

### 5.1 Why Separated Filters Fail

Early implementations used separated θ/φ filters: Lagrange interpolation in $\theta$ (on Gauss-Legendre nodes) and Fourier interpolation in $\phi$. This works for $m=0$ (axisymmetric modes) but **fails for $m \neq 0$** because:

```math
P_l^m(\cos\theta) \text{ for odd } m \text{ contains } \sin^m\theta \text{ factors} \Rightarrow \text{not a polynomial of } \cos\theta.
```

Lagrange interpolation on GL nodes assumes polynomial structure, causing errors when azimuthal modes couple θ and φ dependence.

### 5.2 The Correct Approach: DFT → Per-$m$ Filter → IDFT

The `_apply_disagg_filter` function implements the rigorous approach:

1. **DFT analysis in $\phi$**: Decompose data into Fourier modes $a_m, b_m$
2. **Per-$m$ θ filter**: Apply associated Legendre $P_l^m$ filter for each $m$:
   ```math
   F_{m}[i,j] = w_{\theta,j} \sum_{l=m}^{L} \frac{(2l+1)}{2} \frac{(l-m)!}{(l+m)!} P_l^m(x_i) P_l^m(x_j)
   ```
3. **IDFT synthesis**: Reconstruct at target $\phi$ points

This correctly handles all $m$ modes and improves accuracy by 5-12× over separated filters at 4-5 octree levels.

---

## 6. Accuracy and Convergence

### 6.1 Tested Accuracy (February 2026)

| `leaf_lambda` | Octree Levels | Matvec Error | Status |
|---------------|---------------|--------------|--------|
| 3.0 | 4 | 0.0007% | Excellent ✓ |
| 2.0 | 4 | 0.0036% | Good ✓ |
| **1.0** | **5** | **0.15%** | **Acceptable ✓ RECOMMENDED** |
| 0.75 | 6 | 11% | Unstable ✗ |

**Production Guideline**: **Use `leaf_lambda >= 1.0` for <1% error.**

### 6.2 Why 6+ Levels Fail

At 6 levels (`leaf_lambda=0.75`), the truncation order $L$ exceeds $kr_{\min}$ (minimum box separation):
- Leaf level: $L = 15$, $kr_{\min} = 9.42$ → $L/kr = 1.59$
- Spherical Hankel $h_l^{(2)}(kr)$ grows exponentially for $l > kr$ (evanescent region)
- Translation sum requires exact cancellation of large terms → numerical instability

This is **fundamental physics**, not a bug. Solution: keep $L \leq kr$ at all levels (i.e., `leaf_lambda >= 1.0`).

### 6.3 Octree Completeness

The octree decomposition has been verified to be **complete**: every pair of basis functions is accounted for either in the near-field matrix or in an interaction list at some level (zero missing pairs).

---

## 7. Near-Field Preconditioning

### 7.1 Reordered-ILU Preconditioner

For $N = 30{,}000$ with `leaf_lambda=1.0`:
- Near-field: 47% fill = 6.9 GB sparse matrix
- Direct ILU($\tau$) on natural ordering: **slow** (matrix is not block-banded)
- **Reordered-ILU**: Permute $\mathbf{Z}_{\text{near}}$ to MLFMA ordering (basis functions sorted by leaf box) → block-banded structure → 10-20× faster ILU factorization

```julia
P_nf = build_mlfma_preconditioner(A; factorization=:ilu, ilu_tau=1e-3)
```

Build time: ~4 minutes for $N=30k$ (vs. hours with natural ordering).

### 7.2 Iteration Counts

With reordered-ILU:
- $N = 7{,}584$: 20 GMRES iterations to $10^{-6}$ tolerance
- $N = 30{,}000$: ~20-30 iterations (iteration count grows slowly with $N$)

---

## 8. Practical Usage

### 8.1 Basic Usage

```julia
using DifferentiableMoM

mesh = read_obj_mesh("aircraft.obj")
freq = 3e8  # 0.3 GHz
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

# Auto-selects MLFMA for N > 50,000
result = solve_scattering(mesh, freq, pw)
```

### 8.2 Manual MLFMA with Preconditioner

```julia
rwg = build_rwg(mesh)
k = 2π * freq / 299792458.0

# Build MLFMA operator
A = build_mlfma_operator(mesh, rwg, k;
                          leaf_lambda=1.0,  # IMPORTANT: >= 1.0 for accuracy
                          verbose=true)

# Build reordered-ILU preconditioner
P_nf = build_mlfma_preconditioner(A; factorization=:ilu, ilu_tau=1e-3)

# Assemble excitation and solve
v = assemble_excitation(mesh, rwg, pw)
I_coeffs, stats = solve_gmres(A, v; preconditioner=P_nf, tol=1e-6)

println("GMRES converged in $(stats.niter) iterations")
```

### 8.3 Forcing MLFMA in `solve_scattering`

```julia
# Force MLFMA even for smaller problems
result = solve_scattering(mesh, freq, pw; method=:mlfma)
```

### 8.4 Adjusting Thresholds

```julia
# Change auto-selection threshold
result = solve_scattering(mesh, freq, pw;
                          mlfma_threshold=20000)  # Use MLFMA for N > 20k
```

---

## 9. Parameter Reference

### 9.1 `build_mlfma_operator` Keywords

| Parameter | Default | Description |
|-----------|---------|-------------|
| `leaf_lambda` | 0.25 | Leaf box edge in wavelengths (**use ≥1.0 for production**) |
| `precision` | 3 | Precision parameter $p$ in truncation formula |
| `quad_order` | 3 | Quadrature order for near-field and radiation patterns |
| `verbose` | false | Print build progress information |

### 9.2 `build_mlfma_preconditioner` Keywords

| Parameter | Default | Description |
|-----------|---------|-------------|
| `factorization` | `:lu` | One of `:lu`, `:ilu`, `:diag` |
| `ilu_tau` | 1e-3 | ILU drop tolerance (smaller = more fill, better preconditioner) |

---

## 10. Performance Data

### 10.1 Scalability (0.3 GHz aircraft, `leaf_lambda=1.0`)

| $N$ | Octree Build | NF Assembly | ILU Precond | GMRES Iters | Total Time |
|-----|--------------|-------------|-------------|-------------|------------|
| 7,584 (4λ) | 0.0s | 29s | 197s | 20 | ~7 min |
| 30,336 (1λ) | 0.03s | 717s | 238s | ~20-30 | ~20 min |

### 10.2 Memory Footprint ($N = 30{,}336$)

| Component | Memory |
|-----------|--------|
| Near-field $\mathbf{Z}_{\text{near}}$ (47% fill) | 6.9 GB |
| Radiation patterns (4 × $n_{\text{pts}}$ × $N$) | ~0.6 GB |
| ILU factors (~85% fill) | ~6.2 GB |
| **Total peak** | **~14 GB** |

Fits comfortably on 24 GB RAM (tested on Mac M3).

---

## 11. Code Mapping

| Concept | Source file | Key function / type |
|---------|------------|---------------------|
| Octree construction | `src/fast/Octree.jl` | `build_octree`, `OctreeBox`, `OctreeLevel` |
| MLFMA operator | `src/fast/MLFMA.jl` | `build_mlfma_operator`, `MLFMAOperator` |
| Radiation patterns | `src/fast/MLFMA.jl` | `compute_bf_radiation_patterns` |
| Translation factors | `src/fast/MLFMA.jl` | `compute_translation_factor` |
| Per-$m$ spectral filters | `src/fast/MLFMA.jl` | `_apply_disagg_filter`, `_build_theta_filter_m` |
| Spherical sampling | `src/fast/MLFMA.jl` | `make_sphere_sampling`, `SphereSampling` |
| Matvec | `src/fast/MLFMA.jl` | `mul!(y, A::MLFMAOperator, x)` |
| Reordered preconditioner | `src/solver/NearFieldPreconditioner.jl` | `build_mlfma_preconditioner` |
| Workflow integration | `src/Workflow.jl` | `solve_scattering(...; method=:mlfma)` |

---

## 12. Common Pitfalls

### 12.1 Using `leaf_lambda < 1.0`

**Problem**: Accuracy degrades rapidly at 6+ octree levels due to translation operator instability.

**Solution**: Always use `leaf_lambda >= 1.0` for production. Test with `leaf_lambda=1.5` or `2.0` for safety.

### 12.2 Natural-Order ILU

**Problem**: ILU factorization on $\mathbf{Z}_{\text{near}}$ in natural RWG ordering is extremely slow (hours for $N=30k$).

**Solution**: Use `build_mlfma_preconditioner` which automatically reorders to MLFMA (box-ordered) structure before ILU.

### 12.3 Under-Resolved Meshes

**Problem**: MLFMA assumes smooth basis function distributions. Under-resolved meshes (edge > $\lambda/10$) violate sampling assumptions.

**Solution**: Use `solve_scattering` with `check_resolution=true` (default) to validate mesh density.

---

## 13. Adjoint and Differentiation

The MLFMA operator supports adjoint matvec via `mul!(y, A', x)`:
- Near-field: `adjoint(A.Z_near) * x`
- Far-field: Conjugated translation factors + swapped aggregation/disaggregation roles

This enables impedance optimization with MLFMA-accelerated adjoint gradient computation (see "Differentiable Design" documentation).

---

## 14. Exercises

### 14.1 Conceptual Questions

1. **Complexity**: Derive why MLFMA achieves $O(N \log N)$ complexity. Hint: count operations at each level and sum over $O(\log N)$ levels.

2. **Translation instability**: Explain why $h_l^{(2)}(kr)$ for $l > kr$ causes numerical issues. What is the physical meaning of this regime?

3. **Per-$m$ filters**: Why does separated Lagrange interpolation fail for $m \neq 0$ modes? Sketch $P_3^1(\cos\theta)$ and explain why it's not polynomial in $\cos\theta$.

### 14.2 Coding Exercises

1. Build an MLFMA operator for a 1λ-resolution sphere at 1 GHz. Print octree levels, near-field fill %, and leaf truncation order $L$.

2. Compare dense vs. MLFMA matvec accuracy for $N \approx 5000$. Use `norm(A*x - Z*x) / norm(Z*x)`.

3. Test `leaf_lambda` in `[0.75, 1.0, 1.5, 2.0]` and plot matvec error vs. number of octree levels.

4. Benchmark ILU build time with and without reordering for $N \approx 10{,}000$.

### 14.3 Advanced Challenge

Implement a frequency sweep (0.1--1.0 GHz) with MLFMA. At each frequency, rebuild the operator (geometry is constant but wavelength changes). Compare total time vs. dense assembly.

---

## 15. Chapter Checklist

- [ ] Explain MLFMA's $O(N \log N)$ complexity and why it outscales ACA H-matrices.
- [ ] Build an octree and inspect neighbor/interaction lists at each level.
- [ ] Describe aggregation, translation, and disaggregation in the multi-level algorithm.
- [ ] Understand per-$m$ spectral filters and why separated filters fail.
- [ ] Apply the production guideline: `leaf_lambda >= 1.0` for <1% accuracy.
- [ ] Use `build_mlfma_operator` and reordered-ILU preconditioning.
- [ ] Integrate MLFMA into `solve_scattering` with auto-selection or forced method.
- [ ] Recognize translation operator instability at deep octree levels.

---

## 16. Further Reading

- **Chew, W. C. et al.** *Fast and Efficient Algorithms in Computational Electromagnetics* (2001) -- Ch. 5-7: MLFMA theory, aggregation/disaggregation, and translation operators.
- **Ergül, Ö. & Gürel, L.** *The Multilevel Fast Multipole Algorithm (MLFMA) for Solving Large-Scale Computational Electromagnetics Problems* (2014) -- Comprehensive MLFMA reference with implementation details.
- **DifferentiableMoM.jl source**: `src/fast/MLFMA.jl` for per-$m$ filter implementation; `src/fast/Octree.jl` for tree construction algorithms.
- **Memory documentation**: `MEMORY.md` in the project root for latest accuracy benchmarks and troubleshooting notes.

---

*Next: "Automatic Differentiation and Adjoint Methods" -- extends MLFMA to differentiable optimization workflows with impedance sensitivities.*
