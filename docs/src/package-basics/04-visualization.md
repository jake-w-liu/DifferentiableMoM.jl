# Visualization

## Purpose

This chapter treats visualization as part of the *numerical verification loop*:
before assembly/solve, you confirm geometry integrity, orientation plausibility,
and scale realism from package-native plotting utilities.

---

## Learning Goals

After this chapter, you should be able to:

1. Use package plotting tools to validate repair/coarsening results quickly.
2. Understand why shared axis limits and equal aspect ratio are required for
   physically meaningful visual comparisons.
3. Export reproducible preview artifacts for reports and experiment logs.

---

## 1) Why Plotting Is a Technical Check

Many forward-solve failures are geometry failures in disguise:

- unintended topology changes after repair,
- over-aggressive coarsening,
- unit-scale mistakes (mm treated as m, etc.),
- disconnected surface fragments.

A mesh preview often reveals these issues faster than matrix diagnostics.

---

## 2) What the Visualization Utilities Actually Do

Package-level APIs:

- `plot_mesh_wireframe(mesh; ...)`
- `plot_mesh_comparison(mesh_a, mesh_b; ...)`
- `save_mesh_preview(mesh_a, mesh_b, out_prefix; ...)`

Internally:

1. `mesh_wireframe_segments(mesh)` converts unique edges into line segments
   `(p1, p2, NaN)` for robust 3D path plotting.
2. `_realistic_axis_limits([mesh_a, mesh_b])` computes a **shared cubic frame**:
   - gather global min/max per axis,
   - take `max_span = maximum(maxs - mins)`,
   - set all axes to center ± `max_span/2` (plus padding).
3. `plot_mesh_comparison` uses `aspect_ratio=:equal` on both panels.

This prevents false visual distortion from independent axis autoscaling.

---

## 3) Step-by-Step: Single-Mesh Sanity View

```julia
using DifferentiableMoM

mesh = read_obj_mesh("airplane_repaired.obj")

p = plot_mesh_wireframe(
    mesh;
    title = "Repaired mesh",
    color = :steelblue,
    camera = (30, 30),
    linewidth = 0.7,
)
display(p)
```

What to check visually:

1. gross geometry and connectivity,
2. expected length scale (meters in this package),
3. no obvious detached fragments.

---

## 4) Step-by-Step: Repaired vs Simulation Mesh

This is the most useful diagnostic before a large run.

```julia
mesh_rep = read_obj_mesh("airplane_repaired.obj")
mesh_sim = read_obj_mesh("airplane_coarse.obj")

p = plot_mesh_comparison(
    mesh_rep,
    mesh_sim;
    title_a = "Repaired mesh",
    title_b = "Simulation mesh",
    color_a = :steelblue,
    color_b = :darkorange,
    camera = (30, 30),
    size = (1200, 520),
    pad_frac = 0.04,
)
display(p)
```

Because limits are shared and aspect ratio is equal, you can reliably answer:

- did coarsening preserve the dominant shape?
- are appendages (wings/tail/fins) still represented?
- is the model scale still realistic?

---

## 5) Save Artifacts for Reproducibility

```julia
preview = save_mesh_preview(
    mesh_rep,
    mesh_sim,
    "figs/airplane_mesh_preview";
    title_a = "Repaired mesh",
    title_b = "Simulation mesh",
    camera = (30, 30),
)

println(preview.png_path)
println(preview.pdf_path)
```

The returned paths make it easy to log generated assets in scripts/CI.

---

## 6) Annotated Workflow Before Running EFIE

```julia
using DifferentiableMoM

# (A) Import + repair
mesh_raw = read_obj_mesh("Airplane.obj")
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true)
mesh_rep = rep.mesh

# (B) Coarsen to target unknown count
coarse = coarsen_mesh_to_target_rwg(mesh_rep, 600)
mesh_sim = coarse.mesh

# (C) Visual check (shared limits + equal aspect)
plot_mesh_comparison(mesh_rep, mesh_sim;
    title_a = "Repaired (high detail)",
    title_b = "Simulation (coarsened)",
    aspect_ratio = :equal
)

# (D) Then build RWG and solve
rwg = build_rwg(mesh_sim; precheck=true, allow_boundary=true)
```

This ordering avoids expensive solves on obviously bad meshes.

---

## 7) Practical Interpretation Tips

1. **“Broken-looking” coarsened mesh** can be acceptable if global scattering
   observables are your target and topology remains valid.
2. **Large geometric distortion** in critical features means coarsening is too
   aggressive for your intended observable.
3. Always pair visual checks with `mesh_quality_report` counts; the two are
   complementary.

---

## Code Mapping

- Plot implementations: `src/Visualization.jl`
- Wireframe segment generation: `src/Mesh.jl`
- End-to-end examples:
  - `examples/ex_visualize_simulation_mesh.jl`
  - `examples/ex_airplane_rcs.jl`

---

## Exercises

- Basic: render one repaired mesh and annotate `(Nv, Nt, Nedges)` in title.
- Intermediate: compare two target-RWG coarsened meshes and justify which one
  you would trust for monostatic RCS.
- Challenge: create a scripted pre-solve gate that fails if visual preview and
  quality report suggest severe geometric loss.
