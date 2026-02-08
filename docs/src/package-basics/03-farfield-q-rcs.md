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

Given solved coefficients ``\mathbf I``:

1. Build spherical grid: `make_sph_grid(Ntheta,Nphi)`.
2. Precompute radiation vectors: `radiation_vectors(...)`.
3. Evaluate far field: `compute_farfield(G_mat, I, NΩ)`.

Form:

```math
\mathbf E^\infty(\hat{\mathbf r}_q)
=
\sum_{n=1}^{N} I_n\,\mathbf g_n(\hat{\mathbf r}_q).
```

### ASCII Diagram: Far-Field Computation Pipeline

```
    Current coefficients I → Far-field E∞ at direction (θ,φ)
    
    Steps:
    
    1. Define spherical grid: (θ_q, φ_q), q = 1..NΩ
       with weights w_q = sinθ_q Δθ Δφ
    
    2. Precompute radiation vectors g_n(θ_q, φ_q):
       g_n = radiation pattern of RWG basis n
    
    3. Compute far field via superposition:
    
       E∞(θ_q, φ_q) = Σ_n I_n · g_n(θ_q, φ_q)
    
    ┌─────────────────────────────────────────────────────────┐
    │         Matrix form representation                      │
    │                                                         │
    │   G = [g_1(θ_1) ... g_N(θ_1);                          │
    │        ...         ...        ;                         │
    │        g_1(θ_NΩ) ... g_N(θ_NΩ)]  (size: 3NΩ × N)        │
    │                                                         │
    │   E∞ = G I                                             │
    └─────────────────────────────────────────────────────────┘

    Physical interpretation:
    - Each RWG basis has characteristic radiation pattern
    - Total far field = weighted sum of basis patterns
    - Linear in current coefficients I
```

---

## 2) Build Quadratic Objective Matrix `Q`

For chosen polarization vectors ``\mathbf p_q`` and angular mask:

```math
Q_{mn}
=
\sum_q w_q\,
\big(\mathbf p_q^\dagger\mathbf g_m(\hat{\mathbf r}_q)\big)^*
\big(\mathbf p_q^\dagger\mathbf g_n(\hat{\mathbf r}_q)\big).
```

### ASCII Diagram: Q Matrix Construction

```
    Q matrix for directional objective J = I† Q I
    
    Construction steps:
    
    1. Choose polarization vector p_q for each direction (θ_q, φ_q)
       Example: p_q = (1,0,0) for x-polarized far field
    
    2. Apply angular mask to select directions of interest
       Example: cap_mask selects θ ≤ 30° (main beam region)
    
    3. Compute weighted outer product:
    
       Q = Σ_q w_q · (G†·p_q) (G†·p_q)†
    
       where G = radiation matrix from before
    
    ┌─────────────────────────────────────────────────────────┐
    │         Physical interpretation                         │
    │                                                         │
    │   Q measures "how much" current pattern I               │
    │   radiates into target region with target polarization  │
    │                                                         │
    │   J = I† Q I = total power radiated into                │
    │       target region with specified polarization         │
    └─────────────────────────────────────────────────────────┘

    Properties:
    - Q is Hermitian (Q = Q†)
    - Q is positive semidefinite (I† Q I ≥ 0)
    - Size: N × N (same as number of RWG unknowns)
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
2. `RCS` is a scattering metric in area units (``\mathrm m^2``), often shown in dBsm.
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
- Challenge: compute one ``\phi``-cut from `σ_bi` and identify main-lobe and
  strongest sidelobe levels.
