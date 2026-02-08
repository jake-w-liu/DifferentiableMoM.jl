# Adjoint Method

## Purpose

Explain how the package computes gradients with cost largely independent of the
number of design variables, which is the key enabler for high-dimensional
impedance design.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the adjoint system from the forward MoM equation.
2. Interpret the implemented gradient formula in code.
3. Understand when one vs two adjoint solves are required.

---

## 1) Forward Constraint

```math
\mathbf Z(\theta)\mathbf I(\theta)=\mathbf v.
```

Objective (quadratic form):

```math
J(\theta)=\mathbf I^\dagger \mathbf Q \mathbf I.
```

---

## 2) Adjoint System

Differentiate the constraint and eliminate ``\partial \mathbf I/\partial\theta_p``
with adjoint variable ``\boldsymbol\lambda``:

```math
\mathbf Z^\dagger \boldsymbol\lambda
=
\frac{\partial J}{\partial \mathbf I^*}
=
\mathbf Q\mathbf I.
```

Then gradient component:

```math
\frac{\partial J}{\partial\theta_p}
=
-2\,\Re\!\left\{
\boldsymbol\lambda^\dagger
\left(\frac{\partial\mathbf Z}{\partial\theta_p}\right)
\mathbf I
\right\}.
```

---

## 3) Why It Is Efficient

Cost per iteration for one quadratic objective:

1. one forward solve,
2. one adjoint solve,
3. cheap per-parameter contractions with precomputed derivative blocks.

So cost does not scale linearly with parameter count the way finite
differences do.

---

## 4) Package Implementation

- objective: `compute_objective(I,Q)`
- adjoint solve: `solve_adjoint(Z,Q,I)` (solves `Z' \\ (Q*I)`)
- gradient accumulation: `gradient_impedance(Mp,I,lambda; reactive=...)`

For impedance design, derivative blocks come from patch mass matrices.

---

## 5) Minimal Code Walkthrough

```julia
I = solve_forward(Z, v)
J = compute_objective(I, Q)
λ = solve_adjoint(Z, Q, I)
g = gradient_impedance(Mp, I, λ; reactive=true)
```

This is the core loop used inside optimization routines.

---

## Code Mapping

- Core adjoint functions: `src/Adjoint.jl`
- Operator assembly: `src/EFIE.jl`, `src/Impedance.jl`, `src/Solve.jl`
- Optimization usage: `src/Optimize.jl`

---

## Exercises

- Basic: verify numerically that `length(g)` equals number of design patches.
- Challenge: compare runtime of adjoint gradient vs central finite differences
  for increasing parameter count.
