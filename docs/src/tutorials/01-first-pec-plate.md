# Tutorial: First PEC Plate

## Purpose

Run your first physically meaningful MoM simulation on a PEC plate and verify
that the output passes basic sanity checks (solve residual and energy balance).

---

## Learning Goals

After this tutorial, you should be able to:

1. Assemble and solve a PEC EFIE system.
2. Compute far-field power and energy ratio.
3. Read solver output and identify red flags.

---

## 1) Run the Scripted Version

From repository root:

```bash
julia --project=. examples/ex_convergence.jl
```

This script executes a refinement sweep and writes:

- `data/convergence_study.csv`

---

## 2) Minimal Manual Workflow

```julia
using DifferentiableMoM

f = 3e9
c0 = 299792458.0
λ = c0 / f
k = 2π / λ
η0 = 376.730313668

mesh = make_rect_plate(0.1, 0.1, 4, 4)
rwg  = build_rwg(mesh)

Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)
v = assemble_v_plane_wave(mesh, rwg, Vec3(0,0,-k), 1.0, Vec3(1,0,0); quad_order=3)
I = solve_forward(Z, v)

grid = make_sph_grid(36, 72)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
E = compute_farfield(G, I, length(grid.w))

println("residual = ", norm(Z*I - v) / norm(v))
println("P_rad/P_in = ", energy_ratio(I, v, E, grid; eta0=η0))
```

---

## 3) Expected Behavior

- Residual should be small (typically near machine precision with direct solve).
- `P_rad/P_in` should be close to 1 for PEC.
- If energy ratio is very far from 1, first check mesh quality and grid
  resolution.

---

## 4) Common Failure Modes

- **Mesh quality error**: run `mesh_quality_report(mesh)` and inspect counts.
- **Memory pressure**: reduce mesh size (`Nx`, `Ny`) or coarsen imported meshes.
- **Unstable far-field numbers**: increase angular grid density.

---

## Code Mapping

- Core forward solve: `src/EFIE.jl`, `src/Excitation.jl`, `src/Solve.jl`
- Energy checks: `src/Diagnostics.jl`
- Reference script: `examples/ex_convergence.jl`

---

## Exercises

- Basic: rerun at `Nx=6` and compare condition number vs `Nx=4`.
- Challenge: replace the plate by an imported OBJ mesh and repeat the energy
  check after repair.
