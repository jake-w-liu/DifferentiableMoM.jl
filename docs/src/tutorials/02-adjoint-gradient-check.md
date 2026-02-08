# Tutorial: Adjoint Gradient Check

## Purpose

Run a compact end-to-end check that compares adjoint gradients against finite
differences for impedance parameters.

---

## Learning Goals

After this tutorial, you should be able to:

1. Compute adjoint gradients for a small test case.
2. Validate selected components by central finite differences.
3. Interpret relative-error magnitudes.

---

## 1) Build a Small Test Problem

Use a modest plate mesh so repeated solves are fast:

```julia
mesh = make_rect_plate(0.1, 0.1, 3, 3)
rwg = build_rwg(mesh)
```

Assemble `Z_efie`, `Mp`, excitation `v`, and objective `Q` as in forward
tutorials.

---

## 2) Compute Adjoint Gradient

```julia
Z = assemble_full_Z(Z_efie, Mp, theta0; reactive=true)
I = solve_forward(Z, v)
λ = solve_adjoint(Z, Q, I)
g_adj = gradient_impedance(Mp, I, λ; reactive=true)
```

---

## 3) Verify Against FD

```julia
results = verify_gradient(f_obj, g_adj, theta0; indices=1:10, h_fd=1e-5)
for r in results
    println((r.p, r.rel_err_fd))
end
```

Where `f_obj(θ)` rebuilds `Z(θ)`, solves forward, and returns objective value.

---

## 4) Expected Output Pattern

- Relative errors should be small and stable for reasonable `h`.
- If errors are large for all parameters, check sign/parameterization
  consistency (`reactive=true/false`).

---

## Code Mapping

- Gradient checks: `src/Verification.jl`
- Adjoint kernels: `src/Adjoint.jl`
- System assembly/solve: `src/Solve.jl`, `src/Impedance.jl`

---

## Exercises

- Basic: verify first 10 parameters.
- Challenge: sweep `h` values and plot max relative error vs `h`.
