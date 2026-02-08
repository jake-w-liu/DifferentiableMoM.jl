# API: Visualization

## Purpose

Reference for package-level mesh plotting helpers used in OBJ workflows and
simulation previews.

---

## `plot_mesh_wireframe`

```julia
plot_mesh_wireframe(mesh; color=:steelblue, title="Mesh", camera=(30,30), linewidth=0.7, ...)
```

Creates a 3D wireframe from mesh edges.

Important defaults:

- labeled metric axes (`x,y,z` in meters),
- boxed frame and grid,
- camera controls exposed as kwargs.

---

## `plot_mesh_comparison`

```julia
plot_mesh_comparison(mesh_a, mesh_b; title_a="Mesh A", title_b="Mesh B", ...)
```

Creates side-by-side wireframes with:

- shared axis limits,
- equal aspect ratio,
- independent colors/titles.

Useful for repaired-vs-coarsened mesh checks.

---

## `save_mesh_preview`

```julia
save_mesh_preview(mesh_a, mesh_b, out_prefix; kwargs...)
```

Writes both:

- `out_prefix * ".png"`
- `out_prefix * ".pdf"`

Returns `(plot, png_path, pdf_path)`.

---

## Typical Usage

```julia
mesh_rep = read_obj_mesh("data/airplane_repaired.obj")
mesh_coa = read_obj_mesh("data/airplane_coarse.obj")
preview = save_mesh_preview(mesh_rep, mesh_coa, "figs/airplane_mesh_preview")
println(preview.png_path)
```

---

## Notes

- Wireframe plotting is intentionally lightweight and robust for large meshes.
- Axis limits are computed from mesh extents with small padding.

---

## Code Mapping

- Implementation: `src/Visualization.jl`
- Mesh segment source: `src/Mesh.jl` (`mesh_wireframe_segments`)
- Example usage: `examples/ex_visualize_simulation_mesh.jl`, `examples/ex_airplane_rcs.jl`

---

## Exercises

- Basic: plot one mesh with two different camera settings.
- Challenge: compare a repaired mesh and two coarsening levels in separate
  previews and discuss geometric fidelity tradeoffs.
