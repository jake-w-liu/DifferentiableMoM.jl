# Gradient Verification

## Purpose

Validate that adjoint gradients are correctly implemented for the discretized
problem solved by the package.

---

## Learning Goals

After this chapter, you should be able to:

1. Run finite-difference gradient checks for selected parameters.
2. Understand when complex-step checks are valid.
3. Interpret relative-error trends versus step size.

---

## 1) Main Verification Formula

Adjoint component:

```math
g_p^{\mathrm{adj}}
=
-2\,\Re\!\left\{
\boldsymbol\lambda^\dagger
\left(\frac{\partial\mathbf Z}{\partial\theta_p}\right)\mathbf I
\right\}.
```

Reference finite difference (central):

```math
g_p^{\mathrm{FD}}
\approx
\frac{J(\theta_p+h)-J(\theta_p-h)}{2h}.
```

Compare using relative error.

---

## 2) Package Utilities

- `fd_grad(f, theta, p; h=...)`
- `complex_step_grad(f, theta, p; eps=...)`
- `verify_gradient(f_objective, adjoint_grad, theta; ...)`

Important: complex-step requires holomorphic dependence in the perturbed
parameter; full real-valued quadratic forms with conjugation are generally not
holomorphic end-to-end.

---

## 3) Recommended Workflow

1. choose a small case (fast recomputation),
2. evaluate adjoint gradient,
3. verify a subset of parameters with central FD (`h≈1e-5` typical),
4. sweep `h` to observe truncation/roundoff tradeoff.

---

## 4) Minimal Example

```julia
results = verify_gradient(
    f_obj,           # θ -> J(θ)
    g_adj,           # adjoint gradient vector
    theta0;
    indices=1:10,
    h_fd=1e-5,
)

for r in results
    println((r.p, r.adj, r.fd, r.rel_err_fd))
end
```

---

## 5) Interpreting Outcomes

- Small stable relative errors: adjoint likely correct.
- Error decreases then increases as `h` shrinks: expected FD behavior.
- Large bias insensitive to `h`: likely implementation mismatch.

---

## Code Mapping

- Gradient utilities: `src/Verification.jl`
- Adjoint implementation: `src/Adjoint.jl`
- Optimization use path: `src/Optimize.jl`

---

## Exercises

- Basic: run a 10-parameter check and report max relative error.
- Challenge: perform an `h`-sweep (`1e-3` to `1e-8`) and identify the optimal
  FD step region.
