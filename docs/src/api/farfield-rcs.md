# API: Far-Field and RCS

## Purpose

Reference for far-field radiation pattern computation, objective matrix construction (Q-matrices), power diagnostics, and radar cross section (RCS) calculation. These functions transform MoM surface currents into observable quantities: radiation patterns, radiated power, directivity, and scattering cross sections.

---

## Spherical Sampling

### `make_sph_grid(Ntheta, Nphi)`

Create a spherical sampling grid using a uniform midpoint rule in theta and phi, with quadrature weights `w = sin(theta) * d_theta * d_phi`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Ntheta` | `Int` | Number of theta (polar angle) samples. More samples = finer angular resolution in the elevation plane. |
| `Nphi` | `Int` | Number of phi (azimuthal angle) samples. More samples = finer resolution in the azimuthal plane. |

**Returns:** `SphGrid` with fields `rhat`, `theta`, `phi`, `w` (see [types.md](types.md)).

**Total directions:** `N_omega = Ntheta * Nphi`. The grid uses midpoint sampling: `theta = (it - 0.5) * d_theta`, `phi = (ip - 0.5) * d_phi`.

**Choosing resolution:**

| Resolution | `Ntheta` | `Nphi` | Total | Use case |
|-----------|----------|--------|-------|----------|
| Coarse (5 deg / 10 deg) | 36 | 36 | 1,296 | Quick estimates, optimization inner loop |
| Medium (2 deg / 5 deg) | 90 | 72 | 6,480 | Standard RCS and pattern computation |
| Fine (1 deg / 5 deg) | 180 | 72 | 12,960 | Publication-quality patterns, Mie validation |

---

## Radiation Operators

### `radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=376.730313668)`

Compute the per-basis radiation vectors `g_n(r_hat_q)` for all N basis functions at all N_omega grid directions. This is the far-field "transfer matrix": it maps surface current coefficients to far-field amplitudes.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `grid` | `SphGrid` | -- | Spherical sampling grid from `make_sph_grid`. |
| `k` | Real or Complex | -- | Wavenumber (rad/m). |
| `quad_order` | `Int` | `3` | Quadrature order on reference triangle. |
| `eta0` | `Real` | `376.730313668` | Free-space impedance (Ohm). |

**Returns:** `Matrix{ComplexF64}` `G_mat` of size `(3*N_omega, N)`.

**Layout:** `G_mat[(3*(q-1)+1):(3*q), n]` is the 3D far-field vector `g_n(r_hat_q)` for basis `n` at direction `q`.

**Formula:**

```
g_n(r_hat) = (i*k*eta0 / 4*pi) * r_hat x [r_hat x integral{ f_n(r') exp(ik r_hat . r') dS' }]
```

The double cross product extracts the transverse (radiating) component of the far field.

**Performance:** Computing `G_mat` is O(N * N_omega * Nq). For N = 500 and N_omega = 6480, this takes a few seconds.

---

### `compute_farfield(G_mat, I_coeffs, N_omega)`

Compute the far-field pattern `E_inf(r_hat_q) = sum_n I_n * g_n(r_hat_q)` at all grid points.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `G_mat` | `Matrix{ComplexF64}` | Radiation-vector matrix from `radiation_vectors`. |
| `I_coeffs` | `Vector{ComplexF64}` | MoM current coefficients from the forward solve. |
| `N_omega` | `Int` | Number of grid directions (must match `size(G_mat, 1) / 3`). |

**Returns:** `Matrix{ComplexF64}` of shape `(3, N_omega)` containing far-field electric-field phasors `E_inf(r_hat_q)`.

---

## Q-Matrix Helpers

The Q-matrix formulation converts far-field pattern objectives into quadratic forms `J = Re(I' Q I)`, which are differentiable via the adjoint method. This is the bridge between far-field pattern shaping and gradient-based optimization.

### `pol_linear_x(grid)`

Generate x-polarized far-field polarization vectors. At each direction `(theta, phi)`, the polarization vector is `theta_hat(theta, phi)`, which corresponds to x-polarized radiation for broadside (+z direction) observation.

**Parameters:** `grid::SphGrid`

**Returns:** `Matrix{ComplexF64}` of shape `(3, N_omega)`.

---

### `cap_mask(grid; theta_max=pi/18)`

Create a boolean mask selecting directions within a cone of half-angle `theta_max` around the z-axis (broadside). This defines the "target region" for directivity optimization.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `SphGrid` | -- | Spherical grid. |
| `theta_max` | `Real` | `pi/18` (~10 deg) | Maximum polar angle of the cone (radians). Larger values = wider target region = easier optimization but lower peak directivity. |

**Returns:** `BitVector` of length `N_omega` with `true` for directions where `theta <= theta_max`.

**Typical values:**
- `pi/36` (5 deg): Tight pencil beam.
- `pi/18` (10 deg): Standard broadside target.
- `pi/6` (30 deg): Wide beam.

---

### `build_Q(G_mat, grid, pol; mask=nothing)`

Build the Hermitian positive-semidefinite matrix `Q` for the quadratic far-field objective `J = Re(I' Q I)`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `G_mat` | `Matrix{ComplexF64}` | Radiation-vector matrix `(3*N_omega, N)`. |
| `grid` | `SphGrid` | Spherical grid with quadrature weights. |
| `pol` | `Matrix{ComplexF64}` | `(3, N_omega)` polarization vectors (from `pol_linear_x` or custom). |
| `mask` | `BitVector` or `nothing` | Optional mask selecting target directions. If `nothing`, all directions contribute. |

**Returns:** `Matrix{ComplexF64}` `Q` of size `N x N`, Hermitian PSD.

**Mathematical definition:**

```
Q[m,n] = sum_q w_q * conj(p_q' . g_m(r_hat_q)) * (p_q' . g_n(r_hat_q))
```

where `p_q` is the polarization vector, `w_q` is the quadrature weight, and the sum runs over selected directions.

**Building Q for optimization:**

```julia
grid = make_sph_grid(90, 36)
G_mat = radiation_vectors(mesh, rwg, grid, k)
pol = pol_linear_x(grid)

# Q_total: all directions (proportional to total radiated power)
Q_total = build_Q(G_mat, grid, pol)

# Q_target: only broadside cone (power in desired region)
mask = cap_mask(grid; theta_max=pi/18)
Q_target = build_Q(G_mat, grid, pol; mask=mask)

# Directivity ratio: J = (I' Q_target I) / (I' Q_total I)
```

---

### `apply_Q(G_mat, grid, pol, I_coeffs; mask=nothing)`

Apply `Q * I` without forming `Q` explicitly (matrix-free). Useful when N is large and storing the N x N `Q` matrix is expensive.

**Parameters:** Same as `build_Q` plus `I_coeffs::Vector{ComplexF64}`.

**Returns:** `Vector{ComplexF64}` `Q * I` of length `N`.

---

## Diagnostics and RCS

### `radiated_power(E_ff, grid; eta0=376.730313668)`

Compute total radiated power by integrating the far-field pattern over the sphere:

```
P_rad = 1/(2*eta0) * integral{ |E_inf(r_hat)|^2 dOmega }
      = 1/(2*eta0) * sum_q w_q * |E_inf(r_hat_q)|^2
```

**Parameters:**
- `E_ff::Matrix{<:Number}`: Far-field matrix `(3, N_omega)` from `compute_farfield`.
- `grid::SphGrid`: Spherical grid with weights.
- `eta0::Float64=376.730313668`: Free-space impedance.

**Returns:** `Float64` radiated power in watts.

---

### `projected_power(E_ff, grid, pol; mask=nothing)`

Compute polarization-projected angular power:

```
P = sum_q w_q * |p_q' . E_inf(r_hat_q)|^2
```

When `mask` is provided, only selected directions contribute. This is the discrete quantity represented by `Re(I' Q I)` when `Q` is constructed with the same `pol` and `mask`.

**Parameters:**
- `E_ff::Matrix{<:Number}`: Far-field matrix `(3, N_omega)`.
- `grid::SphGrid`: Spherical grid.
- `pol::AbstractMatrix{<:Complex}`: Polarization vectors `(3, N_omega)`.
- `mask`: Optional direction mask.

**Returns:** `Float64` projected power.

---

### `input_power(I, v)`

Compute the power delivered to the structure from the excitation:

```
P_in = -1/2 * Re(I' * v)
```

For a PEC scatterer with `Z I = v`, this is the power extracted from the incident field by the induced currents. For a lossless structure, `P_in` should equal `P_rad` (energy conservation).

**Parameters:**
- `I::Vector{<:Number}`: Current coefficients.
- `v::Vector{<:Number}`: Excitation vector.

**Returns:** `Float64` input power in watts.

---

### `energy_ratio(I, v, E_ff, grid; eta0=376.730313668)`

Compute `P_rad / P_in` as an energy conservation diagnostic.

**Interpretation:**
- `P_rad / P_in ~ 1.0`: Energy is conserved (expected for lossless PEC).
- `P_rad / P_in < 1.0`: Structure has ohmic loss (expected with `Re(Z_s) > 0`).
- `P_rad / P_in > 1.0` or significantly != 1.0 for PEC: Indicates a numerical issue (insufficient mesh density, quadrature order, or far-field grid resolution).

**Parameters:** `I`, `v`, `E_ff`, `grid`, `eta0` (same types as `radiated_power` and `input_power`).

**Returns:** `Float64` energy ratio (dimensionless).

---

### `condition_diagnostics(Z)`

Return condition number and singular value extremes of the MoM matrix. Useful for diagnosing solver issues and understanding preconditioning needs.

**Parameters:** `Z::Matrix{<:Number}`: System matrix.

**Returns:** Named tuple `(cond, sv_max, sv_min)`:
- `cond::Float64`: Condition number `sigma_max / sigma_min`. Values > 10^6 may indicate ill-conditioning.
- `sv_max::Float64`: Largest singular value.
- `sv_min::Float64`: Smallest singular value.

---

### `bistatic_rcs(E_ff; E0=1.0)`

Compute bistatic radar cross section from far-field amplitudes:

```
sigma(r_hat_q) = 4*pi * |E_inf(r_hat_q)|^2 / |E0|^2
```

**Parameters:**
- `E_ff::Matrix{<:Number}`: Far-field matrix `(3, N_omega)`.
- `E0::Real=1.0`: Incident field amplitude (must match the amplitude used in excitation assembly).

**Returns:** `Vector{Float64}` of length `N_omega` containing RCS values in m^2 (linear units, not dBsm).

**To convert to dBsm:** `rcs_dBsm = 10 * log10.(rcs_linear)`

---

### `backscatter_rcs(E_ff, grid, k_inc_hat; E0=1.0)`

Return monostatic (backscatter) RCS for a given incidence direction. The backscatter direction is `-k_inc_hat`, mapped to the nearest sample on the spherical grid.

**Parameters:**
- `E_ff::Matrix{<:Number}`: Far-field matrix `(3, N_omega)`.
- `grid::SphGrid`: Spherical grid.
- `k_inc_hat::Vec3`: Incident propagation direction (unit vector).
- `E0::Real=1.0`: Incident field amplitude.

**Returns:** Named tuple `(sigma, index, theta, phi, angular_error_deg)`:
- `sigma::Float64`: Backscatter RCS in m^2.
- `index::Int`: Grid index of the nearest backscatter direction.
- `theta`, `phi`: Spherical coordinates of that direction (radians).
- `angular_error_deg`: Mismatch between `-k_inc_hat` and the nearest grid direction (degrees). Should be small (< 1 deg) for adequate grid resolution.

---

## Excitation

### `assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)`

Assemble the excitation vector for a plane wave. See [excitation.md](excitation.md) for the full excitation system.

**Parameters:**
- `mesh::TriMesh`, `rwg::RWGData`: Mesh and basis data.
- `k_vec::Vec3`: Wave vector (wavevector, not unit direction).
- `E0`: Amplitude of incident plane wave.
- `pol::Vec3`: Polarization vector.
- `quad_order::Int=3`: Quadrature order.

**Returns:** `Vector{ComplexF64}` of length `N`.

---

## Analytical Reference

### `mie_s1s2_pec(x, mu; nmax=nothing)`

Compute Mie scattering amplitudes `S1` and `S2` for a perfect-electric-conducting (PEC) sphere.

**Parameters:**
- `x::Float64`: Size parameter `x = k * a` (dimensionless, where `a` is sphere radius).
- `mu::Float64`: Cosine of scattering angle `mu = cos(gamma)` where `gamma` is the angle between incident and observation directions.
- `nmax=nothing`: Maximum order for Mie series. Auto-computed from `x` if not provided (uses Wiscombe's criterion).

**Returns:** Tuple `(S1, S2)` of `ComplexF64` scattering amplitudes.

**Note:** This is a low-level function. For RCS values, use `mie_bistatic_rcs_pec` instead.

---

### `mie_bistatic_rcs_pec(k, a, k_inc_hat, pol_inc, rhat; nmax=nothing)`

Compute PEC-sphere bistatic RCS using Mie series. This provides an exact analytical reference for validating MoM results against theory.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | `Float64` | Wavenumber (rad/m). |
| `a` | `Float64` | Sphere radius (meters). |
| `k_inc_hat` | `Vec3` | Incident propagation direction (unit vector). |
| `pol_inc` | `Vec3` | Incident polarization (unit vector, must be orthogonal to `k_inc_hat`). |
| `rhat` | `Vec3` | Observation direction (unit vector). |
| `nmax` | `Nothing` or `Int` | Maximum Mie series order. Auto-computed if `nothing`. |

**Returns:** `Float64` RCS value in m^2.

**Validation workflow:**
```julia
# MoM RCS
rcs_mom = bistatic_rcs(E_ff; E0=1.0)

# Mie reference (same directions)
rcs_mie = [mie_bistatic_rcs_pec(k, a, k_inc_hat, pol, grid.rhat[:, q])
           for q in 1:length(grid.w)]

# Compare
rel_error = abs.(rcs_mom .- rcs_mie) ./ max.(rcs_mie, 1e-30)
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/FarField.jl` | `make_sph_grid`, `radiation_vectors`, `compute_farfield` |
| `src/QMatrix.jl` | `build_Q`, `apply_Q`, `pol_linear_x`, `cap_mask` |
| `src/Diagnostics.jl` | `radiated_power`, `projected_power`, `input_power`, `energy_ratio`, `condition_diagnostics` |
| `src/Excitation.jl` | `assemble_v_plane_wave` |
| `src/Mie.jl` | `mie_s1s2_pec`, `mie_bistatic_rcs_pec` |

---

## Exercises

- **Basic:** Compute `energy_ratio` for a PEC plate scattering problem. Verify it is close to 1.0.
- **Practical:** Compute bistatic RCS for a PEC sphere (ka = 1) and overlay with `mie_bistatic_rcs_pec`. Quantify the maximum relative error.
- **Challenge:** Build `Q_target` and `Q_total` for different `theta_max` cone angles (5, 10, 20, 30 degrees). For each, compute the directivity ratio `(I' Q_target I) / (I' Q_total I)` and plot directivity vs cone angle.
