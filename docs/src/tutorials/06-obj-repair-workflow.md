# Tutorial: OBJ Repair Workflow

## Purpose

Learn a robust mesh-preparation workflow for imported CAD models before running
MoM simulation.

---

## Learning Goals

After this tutorial, you should be able to:

1. Diagnose common OBJ mesh quality failures.
2. Repair meshes to satisfy solver prechecks.
3. Export and visualize repaired/coarsened simulation meshes.

---

## 1) Repair an OBJ Mesh

```bash
julia --project=. examples/ex_repair_obj_mesh.jl ../Airplane.obj ../Airplane_repaired.obj
```

The script reports:

- boundary/non-manifold/orientation counts (before and after),
- numbers of removed invalid/degenerate triangles,
- number of orientation flips.

---

## 2) Visualize Repaired vs Simulation Mesh

```bash
julia --project=. examples/ex_visualize_simulation_mesh.jl \
  data/airplane_repaired.obj data/airplane_coarse.obj figs/airplane_mesh_preview
```

This generates side-by-side wireframe previews with shared axis limits.

---

## 3) Programmatic Repair API

```julia
using DifferentiableMoM

mesh = read_obj_mesh("Airplane.obj")
result = repair_mesh_for_simulation(
    mesh;
    allow_boundary=true,
    require_closed=false,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=true,
    strict_nonmanifold=true,
)
mesh_ok = result.mesh
```

---

## 4) When to Coarsen

Estimate dense matrix memory before solving:

```julia
rwg = build_rwg(mesh_ok)
println(estimate_dense_matrix_gib(rwg.nedges), " GiB")
```

If too large, use:

```julia
coarse = coarsen_mesh_to_target_rwg(mesh_ok, 400)
mesh_sim = coarse.mesh
```

---

## 5) Quality Checklist Before Solve

- no invalid triangles,
- no degenerate triangles,
- no non-manifold edges,
- no orientation conflicts,
- boundary policy consistent with your problem (`allow_boundary` vs closed).

Use `assert_mesh_quality` to enforce this checklist.

---

## Code Mapping

- OBJ IO and repair: `src/Mesh.jl`
- Wireframe plotting: `src/Visualization.jl`
- Scripts: `examples/ex_repair_obj_mesh.jl`, `examples/ex_visualize_simulation_mesh.jl`

---

## Exercises

- Basic: run repair on two different OBJ meshes and compare reports.
- Challenge: intentionally relax one repair flag, then show how/where the
  simulation pipeline fails.
