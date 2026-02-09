# API: Assembly and Solve

## Purpose

Reference for forward-system assembly and linear-solve helpers.

---

## Kernel Functions

### `greens(r, rp, k)`

Scalar free‑space Green's function `G(r, r') = exp(-ikR) / (4πR)` where `R = |r - r'|`. Compatible with complex `k` for complex‑step differentiation.

**Parameters:**
- `r::Vec3`: observation point (meters)
- `rp::Vec3`: source point (meters)
- `k`: wavenumber (real or complex)

**Returns:** `ComplexF64` value of G.

**Convention:** Uses `exp(+iωt)` time convention, hence `exp(-ikR)` in the numerator.

---

### `greens_smooth(r, rp, k)`
Smooth part of the Green's function after singularity extraction:
`G_smooth(r, r') = [exp(-ikR) - 1] / (4πR)` with limit `-ik/(4π)` as `R → 0`. Used for self‑cell integration with singularity subtraction.

**Parameters:** same as `greens`.

**Returns:** `ComplexF64` value of G_smooth.

---

### `grad_greens(r, rp, k)`

Gradient of `G` with respect to observation point `r`:
`∇G = dG/dR * R̂ = [(-ik - 1/R) G] * R̂` where `R̂ = (r - r') / |r - r'|`.

**Parameters:** same as `greens`.

**Returns:** `CVec3` (complex 3‑vector) `∇G`.

---

## Quadrature

### `tri_quad_rule(order)`

Return quadrature points and weights for Gaussian quadrature on the reference triangle with vertices `(0,0)`, `(1,0)`, `(0,1)`.

**Parameters:**
- `order::Int`: quadrature order (supported: 1, 3, 4, 7)

**Returns:** tuple `(xi, w)` where:
- `xi::Vector{SVector{2,Float64}}`: barycentric coordinates `(ξ₁, ξ₂)`
- `w::Vector{Float64}`: corresponding weights (already includes the Jacobian factor of `1/2` for the unit reference triangle).

**Note:** To integrate over a physical triangle of area `A`, use `∫ f dA ≈ Σ w_q f(ξ_q) * 2A`.

---

### `tri_quad_points(mesh, t, xi)`

Map reference‑triangle quadrature points `xi` to physical coordinates on triangle `t`.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `t::Int`: triangle index (1‑based)
- `xi::Vector{SVector{2,Float64}}`: barycentric coordinates from `tri_quad_rule`

**Returns:** `Vector{Vec3}` of physical coordinates.

---

## Singular Integration

### `analytical_integral_1overR(P, V1, V2, V3)`

Analytical integral `∫ 1/|r - P| dS` over a flat triangle with vertices `V1`, `V2`, `V3`. Used for the `1/R` singularity in EFIE self‑cell terms.

**Parameters:**
- `P::Vec3`: source point (on the triangle)
- `V1, V2, V3::Vec3`: triangle vertices (counter‑clockwise order)

**Returns:** `Float64` value of the integral.

---

### `self_cell_contribution(mesh, rwg, n, tm, quad_pts_tm, rwg_vals_m, rwg_vals_n, div_m, div_n, Am, wq, k)`

Compute the EFIE self‑cell integral for basis functions `m` and `n` on the same triangle `tm` using singularity extraction. Returns the value `(vec_part - scl_part)`, not yet multiplied by `-iωμ₀`.

The integral splits as:
- `I_smooth`: standard product quadrature with `G_smooth`
- `I_singular`: outer quadrature point with analytical inner `∫ 1/R dS'`

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `rwg::RWGData`: RWG basis data
- `n::Int`: basis function index (basis `n`; basis `m` is implied by `rwg_vals_m`)
- `tm::Int`: triangle index (self‑cell)
- `quad_pts_tm::Vector{Vec3}`: quadrature points on triangle `tm`
- `rwg_vals_m::Vector{Vec3}`: RWG basis values at quadrature points for basis `m`
- `rwg_vals_n::Vector{Vec3}`: RWG basis values at quadrature points for basis `n`
- `div_m::Float64`: divergence of basis `m` on triangle `tm`
- `div_n::Float64`: divergence of basis `n` on triangle `tm`
- `Am::Float64`: area of triangle `tm`
- `wq`: quadrature weights (vector of length `Nq`)
- `k`: wavenumber

**Returns:** `ComplexF64` contribution to `Z[m,n]`.

**Note:** This is a low‑level internal helper; most users should call `assemble_Z_efie` instead.

---

## EFIE Assembly

### `assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=376.730313668, mesh_precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12)`

Builds dense EFIE matrix for RWG basis.
Includes self-term singular handling through internal branching.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `rwg::RWGData`: RWG basis data
- `k`: wavenumber (can be complex for complex-step)
- `quad_order::Int=3`: quadrature order on reference triangle
- `eta0::Real=376.730313668`: free-space impedance (Ω)
- `mesh_precheck::Bool=true`: run mesh-quality checks before assembly
- `allow_boundary::Bool=true`: allow boundary edges in mesh
- `require_closed::Bool=false`: require closed surface (no boundary edges)
- `area_tol_rel::Float64=1e-12`: relative tolerance for degenerate triangle detection

**Returns:** `Matrix{ComplexF64}` of size `N×N` where `N = rwg.nedges`.

---

## Impedance Assembly

### `precompute_patch_mass(mesh, rwg, partition; quad_order=3)`

Precomputes patch mass matrices `Mp[p][m,n] = ∫_{Γ_p} f_m · f_n dS`.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `rwg::RWGData`: RWG basis data
- `partition::PatchPartition`: mapping of triangles to patches
- `quad_order::Int=3`: quadrature order on reference triangle

**Returns:** `Vector{SparseMatrixCSC{Float64,Int}}` of length `P`.

---

### `assemble_Z_impedance(Mp, theta)`

Builds impedance block from `Mp` and parameter vector:

```math
\mathbf Z_{\mathrm{imp}} = -\sum_{p=1}^P \theta_p \mathbf M_p.
```

For reactive loading, pass complex coefficients in `theta` (e.g., `1im .* θ`)
or use `assemble_full_Z(...; reactive=true)` for the common real-parameter
reactive workflow.

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: patch mass matrices from `precompute_patch_mass`
- `theta::AbstractVector`: parameter vector (real or complex)

**Returns:** `Matrix{ComplexF64}` of size `N×N`.

---

### `assemble_dZ_dtheta(Mp, p)`

Returns derivative matrix ∂Z/∂θ_p = -M_p (exact, closed-form derivative).

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: patch mass matrices
- `p::Int`: patch index (1‑based)

**Returns:** `Matrix{Float64}` (same type as `Mp[p]`).

---

### `assemble_full_Z(Z_efie, Mp, theta; reactive=false)`

Convenience full system assembly:

```math
\mathbf Z = \mathbf Z_{\mathrm{EFIE}} - \sum_p c_p \mathbf M_p,
```

with `c_p = θ_p` (resistive, default) or `c_p = iθ_p` (reactive mode).

**Parameters:**
- `Z_efie::Matrix{<:Number}`: EFIE matrix
- `Mp::Vector{<:AbstractMatrix}`: patch mass matrices
- `theta::AbstractVector`: parameter vector (real for resistive, real for reactive mode)
- `reactive::Bool=false`: if `true`, treat `theta` as reactive parameters (multiplied by `im`)

**Returns:** `Matrix{ComplexF64}`.

---

## Linear Solves

### `solve_forward(Z, v)`

Solve `Z I = v` using direct factorization (for small/medium N).

**Parameters:**
- `Z::Matrix{<:Number}`: system matrix
- `v::Vector{<:Number}`: excitation vector

**Returns:** `Vector{ComplexF64}` solution `I`.

---

### `solve_system(Z, rhs)`

General linear solve `Z x = rhs`.

**Parameters:**
- `Z::Matrix{<:Number}`: system matrix
- `rhs::Vector{<:Number}`: right-hand side

**Returns:** `Vector{ComplexF64}` solution `x`.

---

## Conditioning Helpers

### `make_mass_regularizer(Mp)`

Build a Hermitian positive‑semidefinite mass‑based regularizer:
`R = Σ_p M_p`.

**Parameters:** `Mp::Vector{<:AbstractMatrix}`: patch mass matrices.

**Returns:** dense `Matrix{ComplexF64}` `R`.

---

### `make_left_preconditioner(Mp; eps_rel=1e-8)`

Build a simple mass‑based left preconditioner matrix:
`M = R + ϵ I`, where `R = Σ_p M_p`.

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: patch mass matrices
- `eps_rel::Float64=1e-8`: relative diagonal shift scaling

**Returns:** `Matrix{ComplexF64}` `M`.

---

### `select_preconditioner(Mp; mode=:off, preconditioner_M=nothing, n_threshold=256, iterative_solver=false, eps_rel=1e-6)`

Select the effective left preconditioner matrix used by the solver.

**Modes:**
- `:off`: disable preconditioning (unless `preconditioner_M` is provided)
- `:on`: always build/use a mass‑based preconditioner
- `:auto`: enable only when `iterative_solver=true` or `N >= n_threshold`

If `preconditioner_M` is provided, it takes precedence over `mode`.

**Returns:** tuple `(M_eff, enabled, reason)` where:
- `M_eff` is either a dense `Matrix{ComplexF64}` or `nothing`,
- `enabled::Bool` indicates whether preconditioning is active,
- `reason::String` is a short status string for logging/debugging.

---

### `transform_patch_matrices(Mp; preconditioner_M=nothing, preconditioner_factor=nothing)`

Transform derivative blocks under left preconditioning:
`M_p_tilde = M^{-1} M_p`.

When `preconditioner_M === nothing`, returns `Mp` unchanged.
If `preconditioner_factor` is provided, it is reused instead of factorizing
`preconditioner_M`.

**Returns:** tuple `(Mp_tilde, factor)` where `factor` is `nothing` for the unpreconditioned case.

---

### `prepare_conditioned_system(Z_raw, rhs; regularization_alpha=0.0, regularization_R=nothing, preconditioner_M=nothing, preconditioner_factor=nothing)`

Build the linear system used by forward/adjoint solves:

```
Z_reg = Z_raw + αR
Z_eff = M^{-1} Z_reg
rhs_eff = M^{-1} rhs
```

If no regularization/preconditioning is requested, returns `(Z_raw, rhs, nothing)`.

**Returns:** tuple `(Z_eff, rhs_eff, factor)` where `factor` is the LU factorization used for preconditioning (or `nothing`).

---

## Minimal Pattern

```julia
Zef = assemble_Z_efie(mesh, rwg, k)
Mp = precompute_patch_mass(mesh, rwg, part)
Z = assemble_full_Z(Zef, Mp, theta; reactive=true)
I = solve_forward(Z, v)
```

---

## Code Mapping

- EFIE kernel and assembly: `src/EFIE.jl`
- Impedance blocks: `src/Impedance.jl`
- Solves and conditioning: `src/Solve.jl`

---

## Exercises

- Basic: confirm `assemble_full_Z(..., theta=zeros(P))` matches `Z_efie`.
