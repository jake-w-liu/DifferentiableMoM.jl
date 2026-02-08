# API: Adjoint and Optimization

## Purpose

Reference for adjoint objective/gradient functions and optimization entry
points.

---

## Adjoint Primitives

### `compute_objective(I, Q)`

Computes real quadratic objective `real(dot(I, Q*I))`.

### `solve_adjoint(Z, Q, I)`

Solves adjoint system:

```math
\mathbf Z^\dagger\lambda=\mathbf Q\mathbf I.
```

### `gradient_impedance(Mp, I, lambda; reactive=false)`

Computes impedance-parameter gradient vector for resistive or reactive modes.

---

## Optimizers

### `optimize_lbfgs(...)`

Projected L-BFGS for single quadratic objective.

Key options:

- `reactive`, `maximize`,
- `lb`, `ub`,
- optional regularization/preconditioning parameters.

### `optimize_directivity(...)`

Projected L-BFGS for directivity ratio objective using two adjoint solves.

---

## Conditioning-Related Options

Both optimizers support:

- `regularization_alpha`, `regularization_R`
- `preconditioning=:off|:on|:auto`
- `preconditioner_M`
- auto thresholds (`auto_precondition_n_threshold`, etc.)

---

## Minimal Usage

```julia
theta_opt, trace = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta0;
    reactive=true, maximize=true, maxiter=100
)
```

or

```julia
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta0;
    reactive=true, maxiter=300
)
```

---

## Code Mapping

- Adjoint kernels: `src/Adjoint.jl`
- Optimizers: `src/Optimize.jl`
- Conditioning helpers used by optimizers: `src/Solve.jl`

---

## Exercises

- Basic: run one `optimize_lbfgs` case and inspect trace length.
- Challenge: compare `:off` vs `:auto` preconditioning traces for same problem.
