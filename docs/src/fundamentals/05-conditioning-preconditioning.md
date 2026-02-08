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

## 1) What “Conditioning” Means Here

For the linear system

```math
\mathbf Z \mathbf I = \mathbf v,
```

small perturbations in data or arithmetic can be amplified roughly in
proportion to $\kappa(\mathbf Z)$.
Large $\kappa(\mathbf Z)$ can lead to:

- slow/unstable iterative behavior (if you use iterative methods),
- loss of significant digits in solves,
- noisy adjoint gradients.

In this package, conditioning diagnostics are available through
`condition_diagnostics`.

---

## 2) Regularization Pathway

The package supports optional mass-based regularization:

```math
\mathbf Z_\alpha = \mathbf Z + \alpha \mathbf R,
```

with $\mathbf R \succeq 0$ built from patch mass matrices when not explicitly
provided.

Implementation helpers:

- `make_mass_regularizer(Mp)`
- `prepare_conditioned_system(...; regularization_alpha, regularization_R)`

Important rule: if you regularize forward solves, use the same effective
operator in adjoint solves.

---

## 3) Left Preconditioning Pathway

With left preconditioner $\mathbf M$:

```math
\tilde{\mathbf Z} = \mathbf M^{-1}\mathbf Z,\qquad
\tilde{\mathbf v} = \mathbf M^{-1}\mathbf v.
```

Then solve

```math
\tilde{\mathbf Z}\mathbf I = \tilde{\mathbf v}.
```

Adjoint consistency requires:

```math
\tilde{\mathbf Z}^{\dagger}\boldsymbol\lambda
=
\frac{\partial \Phi}{\partial \mathbf I^*}.
```

The package implements this through:

- `make_left_preconditioner`
- `select_preconditioner`
- `transform_patch_matrices`
- `prepare_conditioned_system`

---

## 4) User Modes: `:off`, `:on`, `:auto`

`select_preconditioner` supports:

- `:off`: never build a default preconditioner.
- `:on`: always build and apply mass-based preconditioner.
- `:auto`: conservative activation for larger/iterative settings.

`preconditioner_M` (user-provided matrix) takes precedence over mode selection.

This design keeps baseline validation runs reproducible while still enabling
conditioned workflows for harder problems.

---

## 5) What Preconditioning Improves (and What It Does Not)

Preconditioning can improve numerical behavior and robustness of linear solves.
It does **not** change dense EFIE memory scaling:

- storage remains $O(N^2)$,
- dense direct solve cost remains roughly $O(N^3)$.

So preconditioning helps solver quality, not asymptotic dense complexity.

---

## 6) Minimal Conditioned Solve Example

```julia
using DifferentiableMoM

# Assume Z_raw, v, Mp already built
M_eff, enabled, reason = select_preconditioner(
    Mp; mode=:auto, n_threshold=256, iterative_solver=true
)

Z_eff, rhs_eff, fac = prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=1e-6,
    regularization_R=nothing,
    preconditioner_M=M_eff,
)

I = solve_system(Z_eff, rhs_eff)
```

---

## 7) Practical Checklist

1. Start with `:off` for baseline reproduction.
2. Turn on `:auto` for larger exploratory studies.
3. Keep forward and adjoint operators conditioned identically.
4. Record mode/reason in logs so runs are reproducible.

---

## Code Mapping

- Conditioning helpers: `src/Solve.jl`
- Optimizer integration: `src/Optimize.jl`
- Diagnostics: `src/Diagnostics.jl`
- Auto-mode example: `examples/ex_auto_preconditioning.jl`

---

## Exercises

- Basic: run `examples/ex_auto_preconditioning.jl` and compare selected modes.
- Challenge: verify gradient agreement with and without conditioning for one
  small impedance-optimization case.
