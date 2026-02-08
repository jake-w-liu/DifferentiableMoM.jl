# Impedance Sensitivities

## Purpose

Show why impedance-parameter derivatives are especially clean in this package:
they are assembled from precomputed patch mass matrices and do not require
differentiating the singular EFIE kernel.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive ``\partial\mathbf Z/\partial\theta_p`` for patch-wise impedance.
2. Distinguish resistive and reactive parameterizations.
3. Map derivative expressions to `gradient_impedance` behavior.

---

## 1) Patch Parameterization

Let

```math
Z_s(\mathbf r;\theta)=\sum_{p=1}^{P}\theta_p\,\chi_{\Gamma_p}(\mathbf r).
```

Impedance matrix contribution:

```math
\mathbf Z_{\mathrm{imp}}(\theta)
=
-\sum_{p=1}^{P}\theta_p \mathbf M_p
\quad\text{(resistive)}.
```

For reactive design in this package:

```math
\mathbf Z_{\mathrm{imp}}(\theta)
=
-\sum_{p=1}^{P}(i\theta_p)\mathbf M_p.
```

---

## 2) Derivative Blocks

Resistive case:

```math
\frac{\partial \mathbf Z}{\partial\theta_p} = -\mathbf M_p.
```

Reactive case:

```math
\frac{\partial \mathbf Z}{\partial\theta_p} = -i\mathbf M_p.
```

These are exact for the chosen discretization and patch map.

---

## 3) Gradient Forms Used in Code

With ``l_p=\boldsymbol\lambda^\dagger \mathbf M_p \mathbf I``:

- resistive: ``\partial J/\partial\theta_p = 2\Re\{l_p\}``,
- reactive: ``\partial J/\partial\theta_p = -2\Im\{l_p\}``.

This is exactly what `gradient_impedance(...; reactive=false/true)` computes.

---

## 4) Why This Is Numerically Attractive

1. No numerical differentiation of EFIE singular kernels.
2. Expensive geometry/kernel assembly is decoupled from parameter derivatives.
3. Derivative blocks can be reused across iterations when patch map is fixed.

---

## 5) Minimal Example

```julia
part = PatchPartition(fill(1, ntriangles(mesh)), 1)
Mp = precompute_patch_mass(mesh, rwg, part; quad_order=3)

Z = assemble_full_Z(Zefie, Mp, theta; reactive=true)
I = solve_forward(Z, v)
λ = solve_adjoint(Z, Q, I)
g = gradient_impedance(Mp, I, λ; reactive=true)
```

---

## Code Mapping

- Patch mass and impedance assembly: `src/Impedance.jl`, `src/Solve.jl`
- Adjoint gradient contractions: `src/Adjoint.jl`
- Optimization integration: `src/Optimize.jl`

---

## Exercises

- Basic: verify sign change between `reactive=false` and `reactive=true` on the
  same test case.
- Challenge: perform one finite-difference check on a selected patch and
  compare to the adjoint component.
