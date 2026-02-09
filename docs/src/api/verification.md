# API: Verification

## Purpose

Reference for derivative‑check and consistency helper functions.

---

## Gradient Check Helpers

### `complex_step_grad(f, theta, p; eps=1e-30)`

Complex‑step derivative of scalar function `f(θ)` with respect to `θ_p`.

**Parameters:**
- `f::Function`: scalar‑valued function `f(θ)` that can handle `ComplexF64` input
- `theta::Vector{Float64}`: parameter vector
- `p::Int`: index of parameter to differentiate (1‑based)
- `eps::Float64=1e-30`: complex‑step perturbation magnitude

**Returns:** `Float64` approximation of `∂f/∂θ_p`.

**Formula:** `∂f/∂θ_p ≈ Im[f(θ + i ε e_p)] / ε`.

**Note:** Requires `f` to be holomorphic in the perturbed parameter. For real‑valued objectives involving complex conjugation (e.g., `I' * Q * I`), complex‑step is not directly applicable.

---

### `fd_grad(f, theta, p; h=1e-6, scheme=:central)`

Finite‑difference derivative of `f(θ)` with respect to `θ_p`.

**Parameters:**
- `f::Function`: scalar‑valued function `f(θ)` (real‑valued)
- `theta::Vector{Float64}`: parameter vector
- `p::Int`: index of parameter to differentiate
- `h::Float64=1e-6`: finite‑difference step size
- `scheme::Symbol=:central`: `:central` (central difference) or `:forward` (forward difference)

**Returns:** `Float64` FD approximation of `∂f/∂θ_p`.

**Formulas:**
- `:central`: `(f(θ + h e_p) - f(θ - h e_p)) / (2h)`
- `:forward`: `(f(θ + h e_p) - f(θ)) / h`

---

### `verify_gradient(f_objective, adjoint_grad, theta; indices=nothing, eps_cs=1e-30, h_fd=1e-6)`

Verify adjoint gradient against complex‑step and finite‑difference.

**Parameters:**
- `f_objective::Function`: objective function `θ → J(θ)` (must support `ComplexF64` input for complex‑step)
- `adjoint_grad::Vector{Float64}`: adjoint gradient vector `g ∈ R^P`
- `theta::Vector{Float64}`: parameter vector `∈ R^P`
- `indices=nothing`: which parameters to check (default: all)
- `eps_cs::Float64=1e-30`: complex‑step perturbation magnitude
- `h_fd::Float64=1e-6`: finite‑difference step size

**Returns:** `Vector{NamedTuple}` with elements `(p, adj, cs, fd, rel_err_cs, rel_err_fd)` where:
- `p::Int`: parameter index
- `adj::Float64`: adjoint gradient value
- `cs::Float64`: complex‑step derivative
- `fd::Float64`: finite‑difference derivative
- `rel_err_cs::Float64`: relative error `|adj - cs| / max(|cs|, 1e-30)`
- `rel_err_fd::Float64`: relative error `|adj - fd| / max(|fd|, 1e-30)`

---

## Typical Pattern

```julia
res = verify_gradient(f_obj, g_adj, theta0; indices=1:10, h_fd=1e-5)
max_err = maximum(r.rel_err_fd for r in res)
println("max rel err = ", max_err)
```

---

## Notes

- Complex‑step is powerful but only valid for holomorphic parameter dependencies.
- For full real‑valued objectives with conjugation, central finite difference is the primary reference.

---

## Code Mapping

- Implementation: `src/Verification.jl`
- Related adjoint code: `src/Adjoint.jl`

---

## Exercises

- Basic: run `fd_grad` on one scalar parameter.
- Challenge: compare central and forward FD errors for the same step sizes.
- `p::Int`: parameter index
- `adj::Float64`: adjoint gradient value
- `cs::Float64`: complex‑step derivative
- `fd::Float64`: finite‑difference derivative
- `rel_err_cs::Float64`: relative error `|adj - cs| / max(|cs|, 1e-30)`
- `rel_err_fd::Float64`: relative error `|adj - fd| / max(|fd|, 1e-30)`

---

## Typical Pattern

```julia
res = verify_gradient(f_obj, g_adj, theta0; indices=1:10, h_fd=1e-5)
max_err = maximum(r.rel_err_fd for r in res)
println("max rel err = ", max_err)
```

---

## Notes

- Complex‑step is powerful but only valid for holomorphic parameter dependencies.
- For full real‑valued objectives with conjugation, central finite difference is the primary reference.

---

## Code Mapping

- Implementation: `src/Verification.jl`
- Related adjoint code: `src/Adjoint.jl`

---

## Exercises

- Basic: run `fd_grad` on one scalar parameter.
- Challenge: compare central and forward FD errors for the same step sizes.

