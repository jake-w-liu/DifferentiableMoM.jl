# API: Verification

## Purpose

Reference for derivative-check and consistency helper functions.

---

## Gradient Check Helpers

### `complex_step_grad(f, theta, p; eps=1e-30)`

Computes complex-step derivative for parameter index `p` when applicable.

### `fd_grad(f, theta, p; h=1e-6, scheme=:central)`

Finite-difference derivative with central/forward options.

### `verify_gradient(f_objective, adjoint_grad, theta; indices, eps_cs, h_fd)`

Returns per-parameter tuples containing:

- adjoint value,
- complex-step value,
- finite-difference value,
- relative errors.

---

## Typical Pattern

```julia
res = verify_gradient(f_obj, g_adj, theta0; indices=1:10, h_fd=1e-5)
max_err = maximum(r.rel_err_fd for r in res)
println("max rel err = ", max_err)
```

---

## Notes

- Complex-step is powerful but only valid for holomorphic parameter
  dependencies.
- For full real-valued objectives with conjugation, central finite difference
  is the primary reference.

---

## Code Mapping

- Implementation: `src/Verification.jl`
- Related adjoint code: `src/Adjoint.jl`

---

## Exercises

- Basic: run `fd_grad` on one scalar parameter.
- Challenge: compare central and forward FD errors for the same step sizes.
