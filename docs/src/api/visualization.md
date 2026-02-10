# API: Visualization

## Purpose

Reference for package‑level mesh plotting helpers used in OBJ workflows and simulation previews.

---

## `plot_mesh_wireframe(mesh; kwargs...)`

Create a 3D wireframe plot of a mesh using package‑native mesh segments.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `color=:steelblue`: line color
- `title::AbstractString="Mesh"`: plot title
- `camera::Tuple{Real,Real}=(30, 30)`: 3D camera view angles (azimuth, elevation)
- `linewidth::Real=0.7`: line width
- `xlims`, `ylims`, `zlims`: optional axis limits (default: auto‑scaled from mesh)
- `kwargs...`: forwarded to Plots.jl `plot` command

**Returns:** Plots.jl plot object.

**Features:**
- labeled metric axes (`x`, `y`, `z` in meters)
- boxed frame and grid
- equal aspect ratio (when used with `plot_mesh_comparison`)

---

## `plot_mesh_comparison(mesh_a, mesh_b; kwargs...)`

Create side‑by‑side 3D wireframe plots with shared axis limits and equal aspect ratio.

**Parameters:**
- `mesh_a::TriMesh`, `mesh_b::TriMesh`: meshes to compare
- `title_a::AbstractString="Mesh A"`, `title_b::AbstractString="Mesh B"`: titles for each subplot
- `color_a=:steelblue`, `color_b=:darkorange`: line colors for each mesh
- `camera::Tuple{Real,Real}=(30, 30)`: 3D camera view angles (same for both subplots)
- `size::Tuple{Int,Int}=(1200, 520)`: total figure size (width, height) in pixels
- `pad_frac::Float64=0.04`: fractional padding around combined mesh bounding box
- `kwargs...`: forwarded to `plot_mesh_wireframe`

**Returns:** Plots.jl plot object with `layout=(1,2)`.

**Features:**
- shared axis limits computed from both meshes
- equal aspect ratio enforced
- independent colors and titles

---

## `save_mesh_preview(mesh_a, mesh_b, out_prefix; kwargs...)`

Generate and save side‑by‑side mesh preview plots as PNG and PDF.

**Parameters:**
- `mesh_a::TriMesh`, `mesh_b::TriMesh`: meshes to compare
- `out_prefix::AbstractString`: output file prefix (without extension)
- `kwargs...`: forwarded to `plot_mesh_comparison`

**Returns:** named tuple `(plot, png_path, pdf_path)` where:
- `plot`: the Plots.jl plot object
- `png_path`: full path to saved PNG file (`out_prefix * ".png"`)
- `pdf_path`: full path to saved PDF file (`out_prefix * ".pdf"`)

**Note:** Creates parent directories if they do not exist.

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
- Requires `Plots.jl` backend (e.g., GR) to be installed and loaded.

---

## Code Mapping

- Implementation: `src/Visualization.jl`
- Mesh segment source: `src/Mesh.jl` (`mesh_wireframe_segments`)
- Example usage: `examples/ex_visualize_simulation_mesh.jl`, `examples/ex_obj_rcs_pipeline.jl`

---

## Exercises

- Basic: plot one mesh with two different camera settings.
- Challenge: compare a repaired mesh and two coarsening levels in separate previews and discuss geometric fidelity tradeoffs.
