# Far-Field, Q, and RCS

## Purpose

Connect solved current coefficients to physically interpretable observables:
far-field vectors, directional quadratic objectives (`Q` matrices), and RCS
metrics.

---

## Learning Goals

After this chapter, you should be able to:

1. Compute far-field samples from MoM currents.
2. Build and use `Q` for directional/polarization objectives.
3. Compute bistatic and monostatic RCS from far-field data.

---

## 1) Far-Field Sampling Pipeline

Given solved coefficients $\mathbf I$:

1. Build spherical grid: `make_sph_grid(Ntheta,Nphi)`.
2. Precompute radiation vectors: `radiation_vectors(...)`.
3. Evaluate far field: `compute_farfield(G_mat, I, NΩ)`.

Form:

```math
\mathbf E^\infty(\hat{\mathbf r}_q)
=
\sum_{n=1}^{N} I_n\,\mathbf g_n(\hat{\mathbf r}_q).
```

---

## 2) Build Quadratic Objective Matrix `Q`

For chosen polarization vectors $\mathbf p_q$ and angular mask:

```math
Q_{mn}
=
\sum_q w_q\,
\big(\mathbf p_q^\dagger\mathbf g_m(\hat{\mathbf r}_q)\big)^*
\big(\mathbf p_q^\dagger\mathbf g_n(\hat{\mathbf r}_q)\big).
```

Package calls:

```julia
pol = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=π/18)
Q = build_Q(G_mat, grid, pol; mask=mask)
J = compute_objective(I, Q)
```

`Q` is Hermitian PSD by construction for this workflow.

---

## 3) RCS Metrics

From far-field samples:

- `bistatic_rcs(E_ff; E0=1.0)` computes per-direction bistatic RCS.
- `backscatter_rcs(E_ff, grid)` extracts near-backscatter sample from grid.

Use dB conversion when plotting:

```julia
σ = bistatic_rcs(E)
σ_dB = 10 .* log10.(max.(σ, 1e-30))
```

---

## 4) Minimal Example

```julia
grid = make_sph_grid(60, 120)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
E = compute_farfield(G, I, length(grid.w))

pol = pol_linear_x(grid)
Qcap = build_Q(G, grid, pol; mask=cap_mask(grid; theta_max=20π/180))
Jcap = compute_objective(I, Qcap)

σ_bi = bistatic_rcs(E)
σ_mono = backscatter_rcs(E, grid)
```

---

## 5) Practical Interpretation

Keep these distinctions clear:

1. `J` from `Q` is an objective-style integrated metric.
2. `RCS` is a scattering metric in area units ($\mathrm m^2$), often shown in dBsm.
3. Beam-centric comparisons (main lobe/sidelobe behavior) are usually better
   aligned with steering objectives than global null-dominated residuals.

---

## Code Mapping

- Far-field grid and radiation vectors: `src/FarField.jl`
- Q construction: `src/QMatrix.jl`
- RCS and power diagnostics: `src/Diagnostics.jl`
- Sphere reference benchmark: `examples/ex_pec_sphere_mie_benchmark.jl`

---

## Exercises

- Basic: compare `J` for two cone half-angles.
- Challenge: compute one $\phi$-cut from `σ_bi` and identify main-lobe and
  strongest sidelobe levels.
