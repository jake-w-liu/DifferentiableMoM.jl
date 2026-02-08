# Forward Pipeline

## Purpose

Show the exact sequence used to compute scattering from geometry and excitation:
mesh ``\rightarrow`` RWG ``\rightarrow`` EFIE matrix ``\rightarrow`` solve currents
``\rightarrow`` far-field observables.

---

## Learning Goals

After this chapter, you should be able to:

1. Assemble and solve forward systems for PEC or impedance sheets.
2. Understand where regularization/preconditioning enter the pipeline.
3. Map each step to package functions.

---

## 1) Pipeline Overview

Forward solve flow:

1. `build_rwg(mesh)` builds basis data.
2. `assemble_Z_efie(mesh,rwg,k)` builds EFIE operator.
3. `precompute_patch_mass(...)` and `assemble_full_Z(...)` add impedance terms.
4. `assemble_v_plane_wave(...)` builds excitation.
5. `solve_forward`/`solve_system` computes current coefficients.

Mathematically:

```math
\mathbf Z(\theta)\mathbf I=\mathbf v.
```

---

## 2) PEC Forward Solve

```julia
using DifferentiableMoM

mesh = make_rect_plate(0.1, 0.1, 6, 6)
rwg = build_rwg(mesh)

f = 3e9
c0 = 299792458.0
k = 2π*f/c0
η0 = 376.730313668

Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)
v = assemble_v_plane_wave(mesh, rwg, Vec3(0,0,-k), 1.0, Vec3(1,0,0); quad_order=3)
I = solve_forward(Z, v)
```

---

## 3) Impedance-Loaded Solve

Patch matrices are precomputed once:

```julia
part = PatchPartition(fill(1, ntriangles(mesh)), 1)
Mp = precompute_patch_mass(mesh, rwg, part; quad_order=3)
theta = [100.0]               # Ω (resistive) or use reactive=true for iθ

Zfull = assemble_full_Z(Z, Mp, theta; reactive=true)
I = solve_forward(Zfull, v)
```

This is the same operator form used inside optimization loops.

---

## 4) Optional Conditioning in Forward Solves

For conditioned workflows:

```julia
M_eff, enabled, _ = select_preconditioner(Mp; mode=:auto, iterative_solver=true)
Zeff, rhs_eff, _ = prepare_conditioned_system(Zfull, v; preconditioner_M=M_eff)
I = solve_system(Zeff, rhs_eff)
```

Conditioning improves numerical behavior but does not change dense memory
complexity class.

---

## 5) Forward Diagnostics

After solving, run:

- `condition_diagnostics(Zfull)`
- residual check `norm(Zfull*I - v)/norm(v)`
- energy ratio via far field (`energy_ratio`)

These are your first correctness gates before optimization.

---

## 6) Common Mistakes

1. Mismatched units (geometry not in meters).
2. Excitation vector not recomputed when wave settings change.
3. Using conditioned forward solve but unconditioned adjoint solve later.

---

## Code Mapping

- EFIE assembly: `src/EFIE.jl`
- Excitation: `src/Excitation.jl`
- System assembly/conditioning: `src/Solve.jl`, `src/Impedance.jl`
- Diagnostics: `src/Diagnostics.jl`

---

## Exercises

- Basic: run PEC and impedance-loaded forward solves on the same mesh and
  compare current norms.
- Challenge: enable `:auto` preconditioning and verify residual and objective
  remain consistent.
