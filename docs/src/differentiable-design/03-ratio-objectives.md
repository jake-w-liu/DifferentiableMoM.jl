# Ratio Objectives

## Purpose

Explain why directivity-style ratio objectives are preferred for beam steering,
and how the package computes stable gradients using two adjoint solves.

---

## Learning Goals

After this chapter, you should be able to:

1. Write the ratio objective used in steering optimization.
2. Derive the quotient-rule gradient structure.
3. Understand why two separate adjoint solves improve robustness.

---

## 1) Ratio Objective

Define:

```math
f(\theta)=\mathbf I^\dagger\mathbf Q_t\mathbf I,\qquad
g(\theta)=\mathbf I^\dagger\mathbf Q_{\mathrm{tot}}\mathbf I.
```

Then optimize:

```math
J(\theta)=\frac{f(\theta)}{g(\theta)}.
```

This rewards concentration of radiation in a target region relative to total
radiation in the selected polarization channel.

---

## 2) Quotient-Rule Gradient

```math
\frac{\partial J}{\partial\theta_p}
=
\frac{
g\,\frac{\partial f}{\partial\theta_p}
-f\,\frac{\partial g}{\partial\theta_p}
}{g^2}.
```

### ASCII Diagram: Ratio Objective Gradient Computation

```
    Ratio objective: J(θ) = f(θ)/g(θ)
    
    where:
    f(θ) = I† Q_t I      (numerator - target region)
    g(θ) = I† Q_tot I    (denominator - total region)
    
    Gradient via quotient rule:
    
    ∂J/∂θ_p = [g·∂f/∂θ_p - f·∂g/∂θ_p] / g²
    
    ┌─────────────────────────────────────────────────────────┐
    │         Two adjoint solves required                    │
    │                                                         │
    │   Solve 1: Z† λ_f = Q_t I      → gives ∂f/∂θ_p         │
    │                                                         │
    │   Solve 2: Z† λ_g = Q_tot I    → gives ∂g/∂θ_p         │
    │                                                         │
    │   Then combine using quotient rule formula              │
    └─────────────────────────────────────────────────────────┘

    Why not combine into one adjoint solve?
    
    Single solve approach: Z† λ = (Q_t - J Q_tot) I
    
    Problem: When J ≈ f/g, cancellation makes RHS numerically unstable.
    Two-solve approach avoids this cancellation issue.
```

Each derivative term is computed adjointly:

```math
\mathbf Z^\dagger\lambda_f = \mathbf Q_t\mathbf I,\qquad
\mathbf Z^\dagger\lambda_g = \mathbf Q_{\mathrm{tot}}\mathbf I.
```

---

## 3) Why Two Adjoint Solves

Using one “effective” matrix
``\mathbf Q_t - J\mathbf Q_{\mathrm{tot}}`` can become numerically delicate near
convergence due to cancellation. The package avoids that by solving separate
adjoint systems for numerator and denominator.

This is more stable for practical L-BFGS optimization.

---

## 4) Implementation Path

`optimize_directivity` in `src/Optimize.jl` does:

1. forward solve for ``\mathbf I``,
2. compute `f_val`, `g_val`, `J_ratio`,
3. solve two adjoints (`lam_t`, `lam_a`),
4. assemble ratio gradient from `g_f`, `g_g`,
5. update by projected L-BFGS.

---

## 5) Minimal Usage Pattern

```julia
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta0;
    reactive=true, maxiter=300, lb=fill(-500.0,P), ub=fill(500.0,P)
)
```

---

## Code Mapping

- Ratio optimizer: `src/Optimize.jl`
- Adjoint primitives: `src/Adjoint.jl`
- Q-matrix construction: `src/QMatrix.jl`

---

## Exercises

- Basic: compare early-iteration traces of absolute-power vs ratio objectives.
- Challenge: modify target cone width and observe resulting tradeoff between
  peak steering and sidelobe suppression.
