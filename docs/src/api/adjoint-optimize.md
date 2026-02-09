# API: Adjoint and Optimization

## Purpose

Reference for adjoint objective/gradient functions and optimization entry points.

---

## Adjoint Primitives

### `compute_objective(I, Q)`

Compute the quadratic objective `J = Re(I† Q I)`.

**Parameters:**
- `I::Vector{<:Number}`: MoM current coefficients
- `Q::Matrix{<:Number}`: Hermitian positive‑semidefinite objective matrix

**Returns:** `Float64` objective value `J`.

---

### `solve_adjoint(Z, Q, I)`

Solve the adjoint system:
```
Z† λ = Q I.
```

**Parameters:**
- `Z::Matrix{<:Number}`: system matrix
- `Q::Matrix{<:Number}`: objective matrix
- `I::Vector{<:Number}`: current coefficients

**Returns:** `Vector{ComplexF64}` adjoint variable `λ`.

---

### `gradient_impedance(Mp, I, lambda; reactive=false)`

Compute the adjoint gradient for impedance parameters:
```
g[p] = -2 Re{ λ† (∂Z/∂θ_p) I }
```

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: patch mass matrices from `precompute_patch_mass`
- `I::Vector{<:Number}`: current coefficients
- `lambda::Vector{<:Number}`: adjoint variable
- `reactive::Bool=false`: if `true`, treat parameters as reactive (`Z_s = iθ`); otherwise resistive (`Z_s = θ`)

**Returns:** `Vector{Float64}` gradient vector `g` of length `P`.

**Formulas:**
- Resistive (`reactive=false`): `∂Z/∂θ_p = -M_p` → `g[p] = +2 Re{ λ† M_p I }`
- Reactive (`reactive=true`): `∂Z/∂θ_p = -iM_p` → `g[p] = -2 Im{ λ† M_p I }`

---

## Optimizers

### `optimize_lbfgs(Z_efie, Mp, v, Q, theta0; kwargs...)`

Projected L‑BFGS for a single quadratic objective `J = I† Q I`.

**Required parameters:**
- `Z_efie::Matrix{ComplexF64}`: EFIE matrix
- `Mp::Vector{<:AbstractMatrix}`: patch mass matrices
- `v::Vector{ComplexF64}`: excitation vector
- `Q::Matrix{ComplexF64}`: objective matrix (Hermitian PSD)
- `theta0::Vector{Float64}`: initial parameter vector

**Key options:**

- `reactive::Bool=false`: if `true`, treat parameters as reactive (`Z_s = iθ`)
- `maximize::Bool=false`: if `true`, maximize `J` instead of minimizing
- `lb`, `ub`: box constraints on `θ` (vectors of same length as `theta0`); `nothing` for no bound
- `maxiter::Int=100`: maximum L‑BFGS iterations
- `tol::Float64=1e-6`: gradient‑norm convergence tolerance
- `m_lbfgs::Int=10`: L‑BFGS memory length
- `alpha0::Float64=0.01`: initial step‑size scaling
- `verbose::Bool=true`: print iteration progress

**Conditioning options:**

- `regularization_alpha::Float64=0.0`: regularization coefficient α
- `regularization_R=nothing`: regularization matrix (default: `Σ_p M_p` if α ≠ 0)
- `preconditioner_M=nothing`: user‑provided left preconditioner matrix
- `preconditioning::Symbol=:off`: `:off`, `:on`, or `:auto`
- `auto_precondition_n_threshold::Int=256`: threshold for `:auto` activation (N ≥ threshold)
- `iterative_solver::Bool=false`: if `true`, enables preconditioning in `:auto` mode
- `auto_precondition_eps_rel::Float64=1e-6`: relative diagonal shift for auto‑built preconditioner

**Returns:** tuple `(theta_opt, trace)` where:
- `theta_opt::Vector{Float64}`: optimized parameter vector
- `trace::Vector{NamedTuple{(:iter, :J, :gnorm), ...}}`: iteration records containing iteration number, objective value, and gradient norm

---

### `optimize_directivity(Z_efie, Mp, v, Q_target, Q_total, theta0; kwargs...)`

Maximize the directivity ratio `J = (I†Q_target I) / (I†Q_total I)` using projected L‑BFGS.

**Required parameters:**
- `Z_efie::Matrix{ComplexF64}`: EFIE matrix
- `Mp::Vector{<:AbstractMatrix}`: patch mass matrices
- `v::Vector{ComplexF64}`: excitation vector
- `Q_target::Matrix{ComplexF64}`: objective matrix for numerator (target region)
- `Q_total::Matrix{ComplexF64}`: objective matrix for denominator (total radiated power)
- `theta0::Vector{Float64}`: initial parameter vector

**Key options:** same as `optimize_lbfgs`, except `reactive` and `maximize` are fixed (`reactive` is allowed, `maximize` is implicitly true for the ratio). The routine internally minimizes `-J`.

**Returns:** same structure as `optimize_lbfgs`.

---

## Conditioning-Related Options

Both optimizers support the following conditioning helpers (see `assembly‑solve.md` for details):

- `regularization_alpha`, `regularization_R`
- `preconditioning=:off|:on|:auto`
- `preconditioner_M`
- auto‑thresholds (`auto_precondition_n_threshold`, etc.)

Internally, they call `select_preconditioner`, `transform_patch_matrices`, and `prepare_conditioned_system`.

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
