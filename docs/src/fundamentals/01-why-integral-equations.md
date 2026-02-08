# Why Integral Equations

## Purpose

If you know basic electromagnetics but are new to computational methods, the
main shift is this:

- **Volume methods** (FDTD/FEM) discretize the full 3D region.
- **Integral-equation methods** (MoM/BEM) discretize only the radiating or
  scattering boundary.

For open-region PEC/impedance scattering, that boundary-only unknown is often
more natural and physically transparent.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why EFIE-MoM is a strong choice for open-region conductor problems.
2. Identify what this package solves for (surface-current coefficients).
3. Estimate memory/compute implications before running a solve.

---

## From Radiation Condition to Surface Unknowns

In free space, fields satisfy an outgoing-wave condition at infinity. Integral
formulations bake this into the Green's function instead of truncating space
with artificial boundaries.

For a scalar Helmholtz field `u` (intuition model), Green's second identity
leads to a boundary representation:

```math
u(\mathbf r)
=
\int_{\Gamma}
\left[
\frac{\partial G(\mathbf r,\mathbf r')}{\partial n'}u(\mathbf r')
-
G(\mathbf r,\mathbf r')\frac{\partial u(\mathbf r')}{\partial n'}
\right]dS'.
```

The key point is not the exact scalar formula; it is the **structure**:
unknowns move from a 3D volume to boundary traces on `Γ`.

EFIE is the vector-electromagnetic analog used in this package.

### Concrete counting example (same electrical resolution)

Suppose a metallic object is enclosed by a cube of side `L=1 m` and you target
`h=1 cm` geometric resolution.

- Volume grid points are on the order of `(L/h)^3 = 100^3 = 10^6`.
- Surface-triangle scale is on the order of `(L/h)^2 = 100^2 = 10^4`.

Even if constants differ by meshing style, the **dimension trend** is robust:
surface unknown count grows one power of `L/h` slower than volume unknown count.
This is the core reason EFIE-MoM is attractive for open-region scattering.

### Why this does not automatically make solves cheap

Surface reduction lowers unknown dimension, but EFIE couples all basis
functions through the Green kernel, so matrices are dense:

```math
Z_{mn}=\langle f_m,\mathcal T[f_n]\rangle.
```

Dense coupling is the tradeoff: fewer unknowns than volume methods, but
`O(N^2)` storage and `O(N^3)` direct solves in the baseline implementation.

---

## What the Package Solves

The code solves for induced surface current density expanded in RWG basis:

```math
\mathbf J(\mathbf r)
\approx
\sum_{n=1}^{N} I_n\,\mathbf f_n(\mathbf r).
```

The linear system is:

```math
\mathbf Z\mathbf I=\mathbf v,
\qquad
\mathbf I=[I_1,\dots,I_N]^T.
```

So the computational unknown is the coefficient vector `I`, not a volumetric
field grid.

### What you reconstruct after solving `I`

Once `I` is known, every downstream quantity is a linear or quadratic function
of the same coefficients:

- Near/far fields are linear in `I`.
- Power-like objectives are quadratic in `I` (for example `I†QI`).
- Adjoint sensitivities reuse the same operator with transposed/conjugated
  solves.

This is why one accurate forward solve is the anchor for both analysis and
optimization.

---

## Concrete Size Comparison (Why This Matters)

Suppose a target has characteristic size `L` and spatial resolution `h`.

- Volume unknown count scales roughly like `(L/h)^3`.
- Surface unknown count scales roughly like `(L/h)^2`.

That order reduction is exactly why surface methods are compelling for many
open-region scattering problems.

However, EFIE matrices are dense, so cost is not automatically low.

### Back-of-envelope runtime scaling

If unknown count doubles (`N -> 2N`):

- Dense memory grows by `4×`.
- Dense direct solve time grows by roughly `8×`.

This explains why careful mesh planning is essential even for boundary-only
discretizations.

---

## Dense-Operator Cost in This Implementation

For `N` RWG unknowns:

```math
\text{memory} \sim O(N^2),
\qquad
\text{direct solve} \sim O(N^3).
```

This repository is a **reference dense implementation** focused on:

- operator transparency,
- differentiable design correctness,
- verification and reproducibility.

It is ideal for benchmark-scale studies and method development.

---

## Practical Rule of Thumb Before You Run

Estimate dense complex-matrix storage (16 bytes per entry):

```math
\text{GiB} \approx \frac{16N^2}{1024^3}.
```

Use this first sanity check:

```julia
using DifferentiableMoM

mesh = make_rect_plate(0.1, 0.1, 6, 6)
rwg  = build_rwg(mesh)
N    = rwg.nedges

println("RWG unknown count N = ", N)
println("Estimated dense matrix memory (GiB) = ", estimate_dense_matrix_gib(N))
```

---

## How This Chapter Maps to Code

- Mesh generation and preprocessing: `src/Mesh.jl`
- RWG basis construction: `src/RWG.jl`
- EFIE assembly core: `src/EFIE.jl`
- Dense-solve path: `src/Solve.jl`

---

## Common Misunderstandings

- “Surface methods are always faster.”
  They reduce unknown dimension but produce dense operators.

- “If my mesh is visualizable, it is good enough.”
  Not necessarily. Non-manifold, degenerate, or badly oriented meshes can break
  RWG and EFIE assembly.

- “MoM gives fields directly without postprocessing.”
  The primary unknown is current coefficients; fields and RCS are derived
  quantities computed from those coefficients.

---

## Exercises

- Basic: compute `N` and memory estimate for `Nx=Ny=6,8,10` plate meshes.
- Derivation check: show that doubling geometric resolution (halving `h`)
  multiplies surface unknowns by ~4 but dense memory by ~16.
