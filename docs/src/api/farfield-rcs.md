# API: Far-Field and RCS

## Purpose

Reference for far-field sampling, objective operators, and scattering metrics.

---

## Spherical Sampling

### `make_sph_grid(Ntheta, Nphi)`

Returns `SphGrid` with midpoint angular sampling and weights.

---

## Radiation Operators

### `radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=...)`

Builds `G_mat` where each basis contributes a 3-vector per direction.

### `compute_farfield(G_mat, I_coeffs, NΩ)`

Returns complex `E_ff` with shape `(3, NΩ)`.

---

## Q-Matrix Helpers

- `pol_linear_x(grid)` polarization vectors.
- `cap_mask(grid; theta_max=...)` angular mask.
- `build_Q(G_mat, grid, pol; mask=nothing)` objective matrix.
- `apply_Q(...)` matrix-free application of `Q` to vector.

---

## Diagnostics and RCS

- `radiated_power(E_ff, grid; eta0=...)`
- `projected_power(E_ff, grid, pol; mask=nothing, eta0=...)`
- `input_power(I, v)`
- `energy_ratio(I, v, E_ff, grid; eta0=...)`
- `bistatic_rcs(E_ff; E0=1.0)`
- `backscatter_rcs(E_ff, grid; E0=1.0, rhat_inc=...)`

---

## Analytical Reference

- `mie_s1s2_pec(...)`
- `mie_bistatic_rcs_pec(...)`

Useful for sphere benchmark checks.

---

## Code Mapping

- Far field: `src/FarField.jl`
- Q utilities: `src/QMatrix.jl`
- Diagnostics: `src/Diagnostics.jl`
- Mie reference: `src/Mie.jl`

---

## Exercises

- Basic: compute `energy_ratio` for one PEC case.
- Challenge: compare `bistatic_rcs` and `mie_bistatic_rcs_pec` for a sphere.
