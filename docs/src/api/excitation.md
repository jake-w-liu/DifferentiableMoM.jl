# API: Excitation System

## Purpose

Reference for incident-field excitation assembly, including plane waves, ports,
near-field sources, imported fields, and pattern-based feeds. The excitation
system provides a unified interface for assembling right-hand-side (RHS)
vectors for the EFIE-MoM system.

For full physics derivations and source-selection guidance, see
`fundamentals/06-excitation-theory-and-usage.md`.

---

## Excitation Types

### `AbstractExcitation`

Abstract base type for all excitations. All concrete excitation types must implement a method for `assemble_excitation`.

---

### `PlaneWaveExcitation`

Plane wave excitation with given propagation vector, amplitude, and polarization.

**Fields:**
- `k_vec::Vec3`: Wave vector (rad/m)
- `E0::Float64`: Amplitude (V/m)
- `pol::Vec3`: Polarization (unit vector)

**Constructor:**
```julia
make_plane_wave(k_vec, E0, pol)
```

---

### `PortExcitation`

Port excitation defined by a set of RWG edges with applied voltage.

**Fields:**
- `port_edges::Vector{Int}`: RWG edges forming the port
- `voltage::ComplexF64`: Port voltage (V)
- `impedance::ComplexF64`: Port impedance (Ω)

**Current implementation note:** `impedance` is stored for model completeness
and downstream workflows, but the present RHS assembly uses only edge-local
voltage excitation (`voltage` and `port_edges`).

---

### `DeltaGapExcitation`

Delta-gap excitation across a single RWG edge.

**Fields:**
- `edge::Int`: RWG edge with delta gap
- `voltage::ComplexF64`: Gap voltage (V)
- `gap_length::Float64`: Physical gap length (m)

**Constructor:**
```julia
make_delta_gap(edge, voltage, gap_length)
```

---

### `DipoleExcitation`

Electric or magnetic dipole source.

**Fields:**
- `position::Vec3`: Dipole position (m)
- `moment::CVec3`: Dipole moment (electric: C·m, magnetic: A·m²)
- `orientation::Vec3`: Dipole orientation metadata (unit vector)
- `type::Symbol`: `:electric` or `:magnetic`
- `frequency::Float64`: source frequency (Hz)

**Constructor:**
```julia
make_dipole(position, moment, orientation, type, frequency=1e9)
```

---

### `LoopExcitation`

Circular loop current source.

**Fields:**
- `center::Vec3`: Loop center (m)
- `normal::Vec3`: Loop normal (unit vector)
- `radius::Float64`: Loop radius (m)
- `current::ComplexF64`: Loop current (A)
- `frequency::Float64`: source frequency (Hz)

**Constructor:**
```julia
make_loop(center, normal, radius, current, frequency=1e9)
```

---

### `ImportedExcitation`

Canonical imported/distributed source model with two supported interpretations.

**Fields:**
- `source_func::Function`: spatial function `source(r) -> CVec3`
- `kind::Symbol`: `:electric_field` or `:surface_current_density`
- `eta_equiv::ComplexF64`: equivalent sheet impedance, used only for `:surface_current_density`
- `min_quad_order::Int`: minimum quadrature-order target used for assembly

**Constructors (recommended):**
```julia
ImportedExcitation(source_func; kind=:electric_field, eta_equiv=376.730313668 + 0im,
                   min_quad_order=3)

make_imported_excitation(source_func; kind=:electric_field, eta_equiv=376.730313668 + 0im,
                         min_quad_order=3)
```

**Interpretation:**
- `kind=:electric_field`  
  `source_func(r)` is the incident electric field `E_inc(r)`.
- `kind=:surface_current_density`  
  `source_func(r)` is an impressed surface current density `J_s(r)`, mapped by
  the local equivalent-sheet approximation
  `E_inc(r) = eta_equiv * J_s(r)`.

This is convenient for prototyping and imported feeds, but it is not a full
radiating-source integral equation for arbitrary volumetric current
distributions.

### Physical scope of `ImportedExcitation`

`ImportedExcitation` is an **RHS field model**. It does not solve an auxiliary
radiation problem for the source itself.

- Use `kind=:electric_field` when you already know `E_inc(r)` at the scatterer.
- Use `kind=:surface_current_density` only as a **local equivalent-sheet
  approximation** (`E_inc ≈ η_equiv J_s`) near sheet-like/source-aperture models.
- Do **not** interpret it as a rigorous replacement for dyadic Green-function
  radiation from arbitrary volume currents.

For rigorous imported sources (horn near fields, external full-wave solvers),
use `ImportedExcitation(...; kind=:electric_field)` (or
`PatternFeedExcitation` when spherical far-field coefficients are available).

### `PatternFeedExcitation`

Incident field synthesized from imported spherical far-field coefficients.

**Fields:**
- `theta::Vector{Float64}`: Polar-angle grid (rad), strictly increasing in `[0, π]`
- `phi::Vector{Float64}`: Azimuth grid (rad), strictly increasing over one open `2π` period
- `Ftheta::Matrix{ComplexF64}`: Far-field coefficient \(F_\theta(\theta,\phi)\)
- `Fphi::Matrix{ComplexF64}`: Far-field coefficient \(F_\phi(\theta,\phi)\)
- `frequency::Float64`: Frequency (Hz)
- `phase_center::Vec3`: Pattern phase-center location (m)
- `convention::Symbol`: `:exp_plus_iwt` or `:exp_minus_iwt` for imported data

The field model is
```math
\mathbf{E}(\mathbf{r})=
\frac{e^{-ikR}}{R}
\left(F_{\theta}(\theta,\phi)\,\hat{\boldsymbol\theta}
      + F_{\phi}(\theta,\phi)\,\hat{\boldsymbol\phi}\right),
\quad R=\|\mathbf{r}-\mathbf{r}_c\|.
```

**Constructors:**
```julia
make_pattern_feed(theta, phi, Ftheta, Fphi, frequency; ...)
make_pattern_feed(Etheta_pattern, Ephi_pattern, frequency; ...)
```

The second form accepts pattern objects exposing `.x`, `.y`, `.U` fields
(e.g., `RadiationPatterns.jl` `Pattern` objects), typically one pattern for
`Eθ` and one for `Eϕ`.

---

### `MultiExcitation`

Combination of multiple excitations with weights.

**Fields:**
- `excitations::Vector{AbstractExcitation}`: List of excitations
- `weights::Vector{ComplexF64}`: Weight for each excitation

**Constructor:**
```julia
make_multi_excitation(excitations, weights=nothing)
```
If `weights` is not provided, equal weights are used.

---

## Core Functions

### `plane_wave_field(r, k_vec, E0, pol)`

Evaluate a plane wave `E^inc(r) = pol * E0 * exp(-i k_vec · r)` at point `r`.

**Parameters:**
- `r::Vec3`: Observation point (meters)
- `k_vec::Vec3`: Wave vector (rad/m)
- `E0`: Amplitude (V/m)
- `pol::Vec3`: Polarization (unit vector)

**Returns:** `CVec3` electric field phasor.

**Convention:** Uses `exp(+iωt)` time convention, hence `exp(-i k_vec · r)` phase.

---

### `assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)`

Assemble excitation vector for a plane wave directly.

**Parameters:**
- `mesh::TriMesh`: Triangle mesh
- `rwg::RWGData`: RWG basis data
- `k_vec::Vec3`: Wave vector (rad/m)
- `E0`: Amplitude (V/m)
- `pol::Vec3`: Polarization (unit vector)
- `quad_order::Int=3`: Quadrature order on reference triangle

**Returns:** `Vector{ComplexF64}` of size `N = rwg.nedges`.

**Note:** Internally creates a `PlaneWaveExcitation` and calls `assemble_excitation`.

---

### `assemble_excitation(mesh, rwg, excitation; quad_order=3)`

Unified excitation assembly for any `AbstractExcitation` subtype.

**Parameters:**
- `mesh::TriMesh`: Triangle mesh
- `rwg::RWGData`: RWG basis data
- `excitation::AbstractExcitation`: Excitation specification
- `quad_order::Int=3`: Quadrature order on reference triangle

**Returns:** `Vector{ComplexF64}` excitation vector `v` where `v[m] = -⟨f_m, E^inc_t⟩`.

**Dispatches** to specialized implementations based on excitation type:
- `PlaneWaveExcitation`: Quadrature integration of plane wave field
- `PortExcitation`: Delta-gap approximation across port edges
- `DeltaGapExcitation`: Single-edge delta gap
- `DipoleExcitation`: Field from electric/magnetic dipole
- `LoopExcitation`: Field from circular current loop
- `ImportedExcitation`: Integration/sampling of imported or distributed source fields
- `PatternFeedExcitation`: Bilinear interpolation of imported spherical pattern data
- `MultiExcitation`: Weighted combination of multiple excitations

---

### `assemble_multiple_excitations(mesh, rwg, excitations; quad_order=3)`

Assemble RHS matrix for multiple excitations (multiple RHS).

**Parameters:**
- `mesh::TriMesh`: Triangle mesh
- `rwg::RWGData`: RWG basis data
- `excitations::Vector{<:AbstractExcitation}`: List of excitations
- `quad_order::Int=3`: Quadrature order on reference triangle

**Returns:** `Matrix{ComplexF64}` of size `N × M` where `N = rwg.nedges` and `M = length(excitations)`. Each column corresponds to one excitation.

---

## Assembly Details

### Plane Wave
For plane waves, the excitation vector is computed via quadrature:
```math
v_m = -\int_{S_m} \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{E}^{\mathrm{inc}}(\mathbf{r}) \, dS
```
where $\mathbf{E}^{\mathrm{inc}}(\mathbf{r}) = \mathbf{p} E_0 e^{-i\mathbf{k}\cdot\mathbf{r}}$ and $S_m$ is the union of the two triangles supporting basis $m$.

### Port and Delta-Gap
Port excitations use a simple edge-localized approximation:
```math
v_m \approx \frac{V}{\ell_m} \delta_{m \in \mathrm{port}}
```
where $\ell_m$ is the edge length and $\delta_{m \in \mathrm{port}}$ is 1 if edge $m$ is in the port edge list.

For a `DeltaGapExcitation`, the implementation uses the specified physical
gap length $g$:
```math
v_m \approx \frac{V}{g}\,\delta_{m,\mathrm{gap}}
```

### Dipole and Loop Sources
Near-field sources compute the incident field from dipole/loop formulas and integrate over the mesh using quadrature.

### Imported/Distributed source (`ImportedExcitation`)
`ImportedExcitation` always contributes as an incident electric field
in the EFIE RHS integral:
```math
v_m = -\int_{S_m} \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{E}^{\mathrm{inc}}(\mathbf{r}) \, dS.
```
For `kind=:electric_field`, `E^{inc} = source(r)` directly.
For `kind=:surface_current_density`, `E^{inc} = eta_equiv * source(r)`.

`min_quad_order` sets a minimum quadrature-order target (mapped to supported
orders `1, 3, 4, 7` through the same internal selector).

Implementation guards:
- the source function output is validated as a finite 3-component vector,
- tuple/array/SVector returns are accepted and converted to complex 3-vectors,
- malformed outputs (wrong length, non-finite entries) throw explicit errors.

Imported fields are sampled at quadrature points and integrated, allowing
incorporation of measured or numerically computed incident fields.

### Pattern Feed (Imported \(E_\theta/E_\phi\) data)
Pattern feeds interpolate complex \((F_\theta,F_\phi)\) data on a spherical
grid, reconstruct \(\mathbf{E}^{inc}\) at quadrature points using the
phase-center geometry, and assemble the same MoM RHS integral.

Use two complex patterns (`Eθ`, `Eϕ`), not a power-only pattern, if polarization
and phase must be preserved.

---

## Examples

### Basic Plane Wave
```julia
k = 2π / 0.1
k_vec = Vec3(0.0, 0.0, -k)
E0 = 1.0
pol = Vec3(1.0, 0.0, 0.0)

# Old method (backward compatible)
v_old = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol)

# New method
pw = make_plane_wave(k_vec, E0, pol)
v_new = assemble_excitation(mesh, rwg, pw)
```

### Pattern Feed from Analytical Dipole Coefficients
```julia
freq = 1.0e9
dip = make_dipole(Vec3(0,0,0), CVec3(0+0im, 0+0im, 1e-12+0im),
                  Vec3(0,0,1), :electric, freq)

theta_deg = collect(0.0:2.0:180.0)
phi_deg = collect(0.0:5.0:355.0)

pat = make_analytic_dipole_pattern_feed(dip, theta_deg, phi_deg;
                                        angles_in_degrees=true)

v_pat = assemble_excitation(mesh, rwg, pat; quad_order=3)
```

### Pattern Feed from External Horn CSV (RadiationPatterns adapter)
```julia
# Example script:
#   julia --project=. examples/ex_radiationpatterns_adapter.jl
#   julia --project=. examples/ex_horn_pattern_import_demo.jl
#   julia --project=. examples/ex_horn_pattern_import_demo.jl examples/antenna_pattern.csv 28.0 reflector
#
# Workflow:
#   CSV(theta, phi, Etheta, Ephi)
#   -> RadiationPatterns.Pattern(Etheta), RadiationPatterns.Pattern(Ephi)
#   -> make_pattern_feed(Etheta_pattern, Ephi_pattern, freq; ...)
#   -> assemble_excitation(...)
```
The adapter path expects two complex vector patterns (`Eθ`, `Eϕ`) on a shared
`(theta, phi)` grid. The helper scripts also support a local fallback container
with the same `.x/.y/.U` interface if `RadiationPatterns.jl` is unavailable in
the current Julia environment.

### Delta-Gap Antenna Feed
```julia
# Excitation on edge 10 with 1V across 1mm gap
gap = make_delta_gap(10, 1.0, 0.001)
v_gap = assemble_excitation(mesh, rwg, gap)
```

### Multi-Port Excitation
```julia
# Define ports on edges 1-3 and 4-6
port1 = PortExcitation([1, 2, 3], 1.0 + 0.0im, 50.0)
port2 = PortExcitation([4, 5, 6], 0.5 + 0.0im, 50.0)
excitations = [port1, port2]
V = assemble_multiple_excitations(mesh, rwg, excitations)
# V[:,1] is excitation from port1, V[:,2] from port2
```

### Near-Field Dipole Source
```julia
# Electric dipole at (0,0,0.1) oriented in x-direction
dipole = make_dipole(Vec3(0.0, 0.0, 0.1), 
                     CVec3(1e-9, 0.0, 0.0), 
                     Vec3(1.0, 0.0, 0.0), 
                     :electric)
v_dipole = assemble_excitation(mesh, rwg, dipole)
```

### Imported excitation (recommended usage)
```julia
# 1) Direct electric-field function
Efun(r) = CVec3(exp(-1im * k * r[3]), 0.0 + 0im, 0.0 + 0im)
curE = make_imported_excitation(Efun; kind=:electric_field, min_quad_order=3)
v_curE = assemble_excitation(mesh, rwg, curE)

# 2) Impressed surface-current function with local equivalent-sheet map
Lx = 0.1
Jsfun(r) = CVec3(cos(2π * r[1] / Lx) + 0im, 0.0 + 0im, 0.0 + 0im)
curJs = make_imported_excitation(Jsfun;
                                 kind=:surface_current_density,
                                 eta_equiv=120 + 30im,
                                 min_quad_order=4)
v_curJs = assemble_excitation(mesh, rwg, curJs)
```

### Reflector + imported horn pattern
```julia
# End-to-end demo with imported Eθ/Eϕ pattern and parabolic reflector geometry:
#   julia --project=. examples/ex_horn_pattern_import_demo.jl examples/antenna_pattern.csv 28.0 reflector
#
# This script loads external pattern coefficients, builds PatternFeedExcitation,
# assembles RHS on a reflector mesh, solves, and exports bistatic/monostatic cuts.
```

### Combined Excitation
```julia
# Plane wave + delta gap with weights
pw = make_plane_wave(k_vec, E0, pol)
gap = make_delta_gap(10, 1.0, 0.001)
multi = make_multi_excitation([pw, gap], [0.7, 0.3])
v_multi = assemble_excitation(mesh, rwg, multi)
```

### Imported Field
```julia
# Import field from external simulation
function imported_E(r)
    # Interpolate from external data
    return CVec3(1.0, 0.0, 0.0) * exp(-1im * k * r[3])
end
imported = ImportedExcitation(imported_E; kind=:electric_field, min_quad_order=3)
v_imported = assemble_excitation(mesh, rwg, imported)
```

---

## Integration with Optimization

All excitations are compatible with adjoint gradient computation:
- Port voltages can be design parameters
- Source positions/orientations can be design parameters
- Excitation weights in `MultiExcitation` can be optimized

**Example:**
```julia
# Design problem with delta-gap excitation
function objective(theta)
    Z = assemble_full_Z(Z_efie, Mp, theta)
    I = Z \ v_gap  # Solve with delta-gap excitation
    return real(dot(I, Q * I))
end

# Gradient via adjoint method works as before
```

---

## Code Mapping

- Implementation: `src/Excitation.jl`
- Core dependency: `src/Quadrature.jl` for integration

---

## Notes

- For plane waves, use `quad_order=3` for typical accuracy.
- Port/delta-gap excitations are local (affect only specified edges).
- Near-field sources require careful placement relative to mesh.
- Imported fields must be provided as functions compatible with the mesh coordinate system.
- For imported/distributed sources, prefer `make_imported_excitation(...)`
  so kind/scaling are explicit.
- If you need a rigorously radiating impressed-current model, use
  `ImportedExcitation(...; kind=:electric_field)` with externally computed `E_inc`
  (or add a dedicated
  source integral model), rather than assuming `E≈ηJ` outside sheet-like cases.
- CI gates currently validate:
  - plane-wave/new API consistency,
  - imported-excitation semantic consistency (`E` vs `ηJ` mapping),
  - dipole/loop power, polarization, and phase checks,
  - pattern-feed amplitude/phase/convention consistency.

---

## Exercises

### Basic
1. Assemble plane-wave excitation with both `assemble_v_plane_wave(...)` and
   `assemble_excitation(..., make_plane_wave(...))`; verify they match.
2. Create a delta-gap excitation on edge 5 and inspect which entries of `v` are non-zero.

### Practical
1. Set up a two-port excitation and solve for currents. Compute S-parameters.
2. Combine plane wave and dipole excitations with weights 0.6 and 0.4.

### Advanced
1. Write a function that imports field data from a CSV file and creates an `ImportedExcitation`.
2. Implement a custom excitation type for waveguide modes and integrate it with the abstract interface.
