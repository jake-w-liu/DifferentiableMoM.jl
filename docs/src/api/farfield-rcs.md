# API: Far-Field and RCS

## Purpose

Reference for far-field sampling, objective operators, and scattering metrics.

---

## Spherical Sampling

### `make_sph_grid(Ntheta, Nphi)`

Create a spherical grid using a uniform midpoint rule in θ and φ, with quadrature weights `w = sin(θ) dθ dφ`.

**Parameters:**
- `Ntheta::Int`: number of θ (polar angle) samples
- `Nphi::Int`: number of φ (azimuthal angle) samples

**Returns:** `SphGrid` with fields:
- `rhat::Matrix{Float64}`: `(3, NΩ)` unit direction vectors
- `theta::Vector{Float64}`: polar angles θ ∈ [0, π] (radians)
- `phi::Vector{Float64}`: azimuthal angles φ ∈ [0, 2π) (radians)
- `w::Vector{Float64}`: quadrature weights (steradians)

**Note:** Total number of directions `NΩ = Ntheta * Nphi`. The grid uses midpoint sampling: `θ = (it - 0.5) * dθ`, `φ = (ip - 0.5) * dφ`.

---

## Radiation Operators

### `radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=376.730313668)`

Compute the per‑basis radiation vectors `g_n(r̂_q)` for all basis functions and all grid directions.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `rwg::RWGData`: RWG basis data
- `grid::SphGrid`: spherical sampling grid
- `k`: wavenumber (real or complex)
- `quad_order::Int=3`: quadrature order on reference triangle
- `eta0::Real=376.730313668`: free‑space impedance (Ω)

**Returns:** `Matrix{ComplexF64}` `G_mat` of size `(3*NΩ, N)` such that:
```
G_mat[(3*(q-1)+1):(3*q), n] = g_n(r̂_q) ∈ C³
```
where `N = rwg.nedges` and `NΩ = length(grid.w)`.

**Formula:** `g_n(r̂) = (i k η₀)/(4π) r̂ × [r̂ × ∫ f_n(r') exp(i k r̂·r') dS']`.

---

### `compute_farfield(G_mat, I_coeffs, NΩ)`

Compute the far‑field `E∞(r̂_q) = Σ_n I_n g_n(r̂_q)` for all grid points.

**Parameters:**
- `G_mat::Matrix{ComplexF64}`: radiation‑vector matrix from `radiation_vectors`
- `I_coeffs::Vector{ComplexF64}`: MoM current coefficients
- `NΩ::Int`: number of grid directions (must match `size(G_mat, 1) ÷ 3`)

**Returns:** `Matrix{ComplexF64}` of shape `(3, NΩ)` containing far‑field electric‑field phasors `E∞(r̂_q)`.

---

## Q‑Matrix Helpers

### `pol_linear_x(grid)`

Generate x‑polarized far‑field polarization vectors (θ̂ component for broadside radiation along z).

**Parameters:** `grid::SphGrid`

**Returns:** `Matrix{ComplexF64}` of shape `(3, NΩ)` where column `q` is `θ̂(θ_q, φ_q)`.

---

### `cap_mask(grid; theta_max=π/18)`

Create a mask selecting directions within a cone of half‑angle `theta_max` around the z‑axis (broadside).

**Parameters:**
- `grid::SphGrid`
- `theta_max::Real=π/18`: maximum polar angle (radians)

**Returns:** `BitVector` of length `NΩ` with `true` for directions where `θ ≤ theta_max`.

---

### `build_Q(G_mat, grid, pol; mask=nothing)`

Build the Hermitian positive‑semidefinite matrix `Q` for the quadratic far‑field objective.

**Parameters:**
- `G_mat::Matrix{ComplexF64}`: radiation‑vector matrix `(3*NΩ, N)`
- `grid::SphGrid`: spherical grid with quadrature weights
- `pol::Matrix{ComplexF64}`: `(3, NΩ)` complex polarization vectors (unit, transverse to `r̂`)
- `mask=nothing`: optional `BitVector` of length `NΩ` selecting target directions

**Returns:** `Matrix{ComplexF64}` `Q` of size `N×N`, Hermitian PSD.

**Definition:** `Q_mn = Σ_q w_q [p†(r̂_q)·g_m(r̂_q)]* [p†(r̂_q)·g_n(r̂_q)]`.

---

### `apply_Q(G_mat, grid, pol, I_coeffs; mask=nothing)`

Apply `Q*I` without forming `Q` explicitly (matrix‑free).

**Parameters:**
- `G_mat::Matrix{ComplexF64}`: radiation‑vector matrix
- `grid::SphGrid`: spherical grid
- `pol::Matrix{ComplexF64}`: polarization vectors
- `I_coeffs::Vector{ComplexF64}`: current coefficients
- `mask=nothing`: optional direction mask

**Returns:** `Vector{ComplexF64}` `Q*I` of length `N`.

---

## Diagnostics and RCS

### `radiated_power(E_ff, grid; eta0=376.730313668)`

Compute total radiated power from far‑field pattern:
```
P_rad = 1/(2η₀) ∫ |E∞(r̂)|² dΩ ≈ 1/(2η₀) Σ_q w_q |E∞(r̂_q)|².
```

**Parameters:**
- `E_ff::Matrix{<:Number}`: far‑field matrix `(3, NΩ)`
- `grid::SphGrid`: spherical grid with weights
- `eta0::Float64=376.730313668`: free‑space impedance (Ω)

**Returns:** `Float64` radiated power in watts.

---

### `projected_power(E_ff, grid, pol; mask=nothing)`

Compute polarization‑projected angular power:
```
P = Σ_q w_q |p_q^† E∞(r̂_q)|².
```

When `mask` is provided, only selected angular samples are included.
This is the discrete quantity represented by `I† Q I` when `Q` is constructed with the same `pol` and `mask`.

**Parameters:**
- `E_ff::Matrix{<:Number}`: far‑field matrix `(3, NΩ)`
- `grid::SphGrid`: spherical grid
- `pol::AbstractMatrix{<:Complex}`: polarization vectors `(3, NΩ)`
- `mask=nothing`: optional direction mask

**Returns:** `Float64` projected power in watts.

---

### `input_power(I, v)`

Compute the power delivered to the structure:
```
P_in = -½ Re(I† v).
```

For a PEC scatterer with `Z I = v`, this is the power extracted from the incident field by the induced currents.

**Parameters:**
- `I::Vector{<:Number}`: MoM current coefficients
- `v::Vector{<:Number}`: excitation vector

**Returns:** `Float64` input power in watts.

---

### `energy_ratio(I, v, E_ff, grid; eta0=376.730313668)`

Compute the ratio `P_rad / P_in` as an energy conservation diagnostic.

- For a lossless PEC structure, this should be ≈ 1.
- For an impedance sheet with `Re(Z_s) > 0`, `P_rad/P_in < 1` (absorbed power).

**Parameters:**
- `I::Vector{<:Number}`: current coefficients
- `v::Vector{<:Number}`: excitation vector
- `E_ff::Matrix{<:Number}`: far‑field matrix
- `grid::SphGrid`: spherical grid
- `eta0::Float64=376.730313668`: free‑space impedance

**Returns:** `Float64` energy ratio (dimensionless).

---

### `condition_diagnostics(Z)`

Return condition number and eigenvalue extremes of the MoM matrix.

**Parameters:** `Z::Matrix{<:Number}`: system matrix

**Returns:** named tuple `(cond, sv_max, sv_min)` where:
- `cond::Float64`: condition number `σ_max / σ_min`
- `sv_max::Float64`: largest singular value
- `sv_min::Float64`: smallest singular value

---

### `bistatic_rcs(E_ff; E0=1.0)`

Compute bistatic radar cross section samples from far‑field amplitudes:
```
σ(r̂_q) = 4π |E∞(r̂_q)|² / |E0|².
```

**Parameters:**
- `E_ff::Matrix{<:Number}`: far‑field matrix `(3, NΩ)`
- `E0::Real=1.0`: incident field amplitude (same units as `E_ff`)

**Returns:** `Vector{Float64}` of length `NΩ` containing RCS values in linear units (m²).

---

### `backscatter_rcs(E_ff, grid, k_inc_hat; E0=1.0)`

Return monostatic/backscatter RCS for a plane‑wave incidence direction `k_inc_hat` (unit propagation direction). The backscatter direction is `-k_inc_hat`, mapped to the nearest sample on `grid`.

**Parameters:**
- `E_ff::Matrix{<:Number}`: far‑field matrix `(3, NΩ)`
- `grid::SphGrid`: spherical grid
- `k_inc_hat::Vec3`: incident propagation direction (unit vector)
- `E0::Real=1.0`: incident field amplitude

**Returns:** named tuple `(sigma, index, theta, phi, angular_error_deg)` where:
- `sigma::Float64`: backscatter RCS (m²)
- `index::Int`: grid index of nearest backscatter direction
- `theta::Float64`, `phi::Float64`: spherical coordinates of that direction (radians)
- `angular_error_deg::Float64`: angular mismatch between `-k_inc_hat` and nearest grid direction (degrees)

---

## Excitation

### `assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)`

Assemble the excitation vector `v_m = -⟨f_m, E^inc_t⟩` for a plane wave.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `rwg::RWGData`: RWG basis data
- `k_vec::Vec3`: propagation vector (wavevector)
- `E0`: complex amplitude of incident plane wave
- `pol::Vec3`: polarization vector (real or complex)
- `quad_order::Int=3`: quadrature order on reference triangle

**Returns:** `Vector{ComplexF64}` of length `N = rwg.nedges`.

---

## Analytical Reference

### `mie_s1s2_pec(x, μ; nmax=nothing)`

Compute Mie scattering amplitudes `S₁` and `S₂` for a perfect‑electric‑conducting sphere.

**Parameters:**
- `x::Float64`: size parameter `x = k*a` (dimensionless)
- `μ::Float64`: cosine of scattering angle `μ = cos(γ)` where `γ` is the angle between incident and observation directions
- `nmax=nothing`: maximum order for Mie series (auto-computed if not provided)

**Returns:** tuple `(S1, S2)` where `S1, S2` are `ComplexF64` values.

**Note:** This is a low‑level function. For RCS calculations, use `mie_bistatic_rcs_pec` instead.

---

### `mie_bistatic_rcs_pec(k, a, k_inc_hat, pol_inc, rhat; nmax=nothing)`

Compute PEC‑sphere bistatic RCS (linear units, m²) using Mie series.

**Parameters:**
- `k::Float64`: wavenumber (rad/m)
- `a::Float64`: sphere radius (meters)
- `k_inc_hat::Vec3`: incident propagation direction (unit vector)
- `pol_inc::Vec3`: incident polarization (unit vector, must be orthogonal to `k_inc_hat`)
- `rhat::Vec3`: observation direction (unit vector)
- `nmax=nothing`: maximum order for Mie series (auto-computed if not provided)

**Returns:** `Float64` RCS value in m².

**Note:** The scattering angle is computed internally as the angle between `k_inc_hat` and `rhat`.

---

## Code Mapping

- Far field: `src/FarField.jl`
- Q utilities: `src/QMatrix.jl`
- Diagnostics: `src/Diagnostics.jl`
- Excitation: `src/Excitation.jl`
- Mie reference: `src/Mie.jl`

---

## Exercises

- Basic: compute `energy_ratio` for one PEC case.
- Challenge: compare `bistatic_rcs` and `mie_bistatic_rcs_pec` for a sphere.
