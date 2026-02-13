# Quickstart

## Purpose

Run a minimal end-to-end workflow in a few minutes:
mesh → RWG basis → EFIE solve → far field → objective.

---

## Learning Goals

After this chapter, you should be able to:

1. Run a small PEC forward solve on a plate.
2. Compute far-field quantities and energy diagnostics.
3. Understand where to go next for optimization and validation.

---

## 1) Minimal Forward Solve (PEC Plate)

```julia
using DifferentiableMoM

# Geometry and basis
mesh = make_rect_plate(0.1, 0.1, 6, 6)   # 1λ×1λ at 3 GHz (approx)
rwg  = build_rwg(mesh)

# Frequency
f  = 3e9
c0 = 299792458.0
λ0 = c0 / f
k  = 2π / λ0
η0 = 376.730313668

# EFIE and excitation
Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)
k_vec = Vec3(0.0, 0.0, -k)
pol   = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, pol; quad_order=3)

# Solve currents
I = solve_forward(Z, v)
```

---

## 2) Far Field and Energy Check

```julia
grid  = make_sph_grid(36, 72)
Gmat  = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
Eff   = compute_farfield(Gmat, I, length(grid.w))

Prad = radiated_power(Eff, grid; eta0=η0)
Pin  = input_power(I, v)
println("Energy ratio = ", Prad / Pin)
```

For a PEC scatterer, `Prad/Pin` should be close to 1 (up to discretization and
quadrature error).

---

## 3) Build a Directional Objective Quickly

```julia
pol_ff = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=30 * π / 180)
Q = build_Q(Gmat, grid, pol_ff; mask=mask)
J = compute_objective(I, Q)   # J = real(I' * Q * I)
println("Directional objective = ", J)
```

This is the same `Q`-matrix objective form used by adjoint optimization.

---

## 4) Where to Go Next

- **Forward/mesh fundamentals**: Part II.
- **Adjoint + optimization**: Part III.
- **Validation and benchmarks**: Part IV.
- **Runnable full workflows**: Tutorials section.

---

## 5) High-Value Example Commands

```bash
julia --project=. examples/04_beam_steering.jl
julia --project=. examples/02_pec_sphere_mie.jl
julia --project=. examples/06_aircraft_rcs.jl ../Airplane.obj 3.0 0.001 300
```

---

## Code Mapping

- Forward assembly: `src/EFIE.jl`, `src/Excitation.jl`, `src/Solve.jl`
- Far-field and objectives: `src/FarField.jl`, `src/QMatrix.jl`, `src/Diagnostics.jl`
- Example implementations: `examples/04_beam_steering.jl`, `examples/01_pec_plate_basics.jl`

---

## Exercises

- Basic: double `Ntheta`/`Nphi` and observe the energy-ratio change.
- Challenge: replace the PEC solve by an impedance-loaded solve using
  `assemble_full_Z` with a nonzero `theta` vector.
