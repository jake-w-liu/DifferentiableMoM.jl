# Tutorial: Airplane RCS

## Purpose

Real‑world electromagnetic simulations involve complex CAD geometries that require robust preprocessing before MoM analysis. This tutorial demonstrates a complete workflow for scattering analysis of an airplane model:

- **Import an OBJ mesh** (typical CAD output) and scale it to physical dimensions.
- **Repair mesh defects** (non‑manifold edges, degenerate triangles, flipped normals) that would break RWG basis functions.
- **Coarsen the mesh** to stay within dense‑matrix memory limits while preserving geometric features.
- **Solve PEC scattering** for a given incidence direction and compute bistatic RCS cuts.
- **Visualize results** with heuristic plots that highlight monostatic and bistatic signatures.

**Key insight:** The bottleneck for large platforms is not the solver time but the $O(N^2)$ memory of the dense MoM matrix. Strategic coarsening allows you to explore scattering trends with manageable unknown counts (e.g., $N \lesssim 500$), providing qualitative insight before investing in fast‑method accelerators.

---

## Learning Goals

After this tutorial, you should be able to:

1. **Prepare imported meshes** using `read_obj_mesh`, `repair_mesh_for_simulation`, and `coarsen_mesh_to_target_rwg`.
2. **Estimate memory requirements** with `estimate_dense_matrix_gib(N)` and choose a feasible target RWG count.
3. **Run the automated airplane RCS pipeline** with custom frequency, scaling, and coarsening parameters.
4. **Interpret the output CSVs and plots** to extract monostatic RCS and bistatic pattern trends.
5. **Trade off accuracy vs speed** by adjusting the target RWG count and understanding coarsening artifacts.
6. **Generate mesh preview images** that compare original and coarsened geometry.

---

## Workflow Overview

The airplane RCS demo follows a defensive pipeline designed to handle real‑world CAD imperfections:

```
OBJ import → scale to meters → repair defects → coarsen to target RWG →
EFIE assembly → forward solve → far‑field → RCS cut + monostatic sample →
CSV outputs + mesh preview + plotting script
```

Each step is logged, and intermediate meshes are saved as OBJ files for inspection.

---

## Step‑by‑Step Walkthrough

### 1) Obtain an Airplane OBJ File

The example expects an OBJ file named `Airplane.obj` in the parent directory. You can use any closed‑surface triangulated mesh (e.g., from Blender, CAD export, or online repositories). For a quick test, you can create a simple placeholder sphere OBJ using the built‑in icosphere generator (see Tutorial 4).

### 2) Run the Pipeline with Default Parameters

From the project root:

```bash
julia --project=. examples/ex_obj_rcs_pipeline.jl ../Airplane.obj 3.0 0.001 300
```

**Arguments:**

| Position | Default | Meaning |
|----------|---------|---------|
| 1 | `../Airplane.obj` | Path to input OBJ file |
| 2 | `3.0` | Frequency in GHz |
| 3 | `0.001` | Scaling factor to convert OBJ units to meters (multiplies vertex coordinates) |
| 4 | `300` | Target number of RWG basis functions after coarsening |

**Outputs** are written to `data/` and `figs/` directories (created automatically).

### 3) Understand the Console Output

The script prints a detailed log:

```
────────────────────────────────────
Airplane PEC RCS Demo
────────────────────────────────────
Input OBJ   : ../Airplane.obj
Frequency   : 3.0 GHz
Scale to m  : 0.001
Target RWG  : 300

── Imported mesh ──
  Vertices: 12345 -> 12000 (scaled/repaired)
  Triangles: 24684 -> 24000
  RWG (before coarsen): 36000
  Dense matrix size estimate: 9.7 GiB
  Repaired winding flips: 12

── Coarsening mesh for dense solve feasibility ──
  Coarsened vertices : 800
  Coarsened triangles: 1600
  RWG after coarsen  : 280
  Dense matrix estimate: 0.6 GiB
  Coarsening iterations: 4, target gap: 20

── Solving PEC scattering ──
  λ0 = 0.09993 m
  Unknowns = 280
  Assembly time: 2.34 s
  Solve time: 0.12 s
  Relative residual: 3.2e-14
  Far-field time: 1.56 s

── Outputs ──
  Repaired mesh: data/airplane_repaired.obj
  Coarsened mesh: data/airplane_coarse.obj
  Mesh preview PNG: figs/airplane_mesh_preview.png
  Mesh preview PDF: figs/airplane_mesh_preview.pdf
  Bistatic φ≈0° cut: data/airplane_bistatic_rcs_phi0.csv
  Monostatic backscatter: data/airplane_monostatic_rcs.csv
  Run summary: data/airplane_rcs_summary.csv
  Monostatic σ = 0.042 m² (‑13.8 dBsm)
```

**Key metrics:**

- **Dense matrix estimate**: If > available RAM, coarsening is essential.
- **Coarsening gap**: Difference between target and achieved RWG count (small is good).
- **Relative residual**: Should be < 1e‑10 for a well‑conditioned solve.
- **Monostatic RCS**: Reported in both m² and dBsm (dB relative to 1 m²).

### 4) Inspect Generated Meshes

Open the saved OBJ files in a mesh viewer (e.g., MeshLab, Blender) to verify repair and coarsening quality. The preview PNG shows side‑by‑side wireframes.

### 5) Plot RCS Results

A separate plotting script reads the CSV files and creates a two‑panel figure:

```bash
julia --project=. examples/ex_obj_rcs_pipeline.jl
```

This produces `figs/airplane_rcs_heuristic.{png,pdf}` with:

- **Top panel**: Bistatic RCS cut (φ ≈ 0°) in dB scale, with monostatic sample highlighted.
- **Bottom panel**: Same cut in linear scale.

The plot is labeled as “heuristic” because coarsening and limited angular resolution affect accuracy; nevertheless, it reveals scattering trends (specular lobes, nulls) characteristic of the platform.

---

## Detailed Explanation of Each Step

### Mesh Import and Scaling

```julia
mesh_in = read_obj_mesh(input_path)
mesh_scaled = TriMesh(mesh_in.xyz .* scale_to_m, copy(mesh_in.tri))
```

CAD models often use arbitrary units (mm, inches, dimensionless). The `scale_to_m` parameter converts to meters, which are required for physical wavenumber $k = 2\pi/\lambda$.

### Mesh Repair

`repair_mesh_for_simulation` performs a series of sanity fixes:

| Operation | Purpose |
|-----------|---------|
| `drop_degenerate=true` | Remove triangles with zero area |
| `fix_orientation=true` | Ensure consistent outward‑pointing normals |
| `strict_nonmanifold=true` | Remove non‑manifold edges (where >2 triangles meet) |
| `allow_boundary=true` | Permit open surfaces (e.g., aircraft without interior) |
| `require_closed=false` | Do not insist on watertight mesh |

The function returns a named tuple with before/after quality reports and
repair counters (flipped triangles, removed invalid/degenerate/non-manifold
triangles, and final mesh quality status).

### Coarsening to Target RWG Count

If the repaired mesh yields more RWG functions than `target_rwg`, the script calls `coarsen_mesh_to_target_rwg`. This iterative algorithm:

1. Computes a curvature‑weighted edge‑collapse priority.
2. Collapses the lowest‑priority edge that does not violate quality thresholds.
3. Rebuilds RWG basis and checks count.
4. Repeats until `RWG ≤ target_rwg` or max iterations reached.

Coarsening preserves sharp features (wing edges, tail fins) while reducing triangle density on flat regions. The `area_tol_rel` parameter prevents collapse that would change local area by more than a fraction.

**Trade‑off:** Aggressive coarsening reduces accuracy but keeps memory feasible. A good rule of thumb is to target $N$ such that `estimate_dense_matrix_gib(N)` is ≤ 0.5× available RAM.

### EFIE Assembly and Solve

Standard MoM workflow with default quadrature order 3. The forward solve uses direct factorization (LU) because $N$ is small after coarsening.

### Far‑Field and RCS Computation

A spherical grid with 121 θ points (1.5° resolution) and 36 φ points (10° resolution) is used for performance. The bistatic RCS is computed for all directions; a φ ≈ 0° cut is extracted for plotting. Monostatic RCS is sampled at the backscatter direction (θ = 180°, φ = 0°).

### CSV Outputs

Three CSV files are created:

1. `airplane_bistatic_rcs_phi0.csv` – θ, σ(m²), σ(dBsm) for the φ ≈ 0° cut.
2. `airplane_monostatic_rcs.csv` – backscatter σ with angular error (difference between sampled direction and exact backscatter).
3. `airplane_rcs_summary.csv` – run parameters, timings, and key metrics for archival.

### Mesh Preview

`save_mesh_preview` generates a side‑by‑side wireframe image comparing repaired and coarsened meshes, with camera angles chosen to show overall shape.

---

## Interpretation Guidelines

### Coarsening Artifacts

| Artifact | Effect on RCS | Mitigation |
|----------|---------------|------------|
| **Smoothed curvatures** (e.g., rounded fuselage) | Shifts specular lobe angles, reduces creeping‑wave contributions | Increase `target_rwg`; use curvature‑weighted coarsening (already default) |
| **Lost small features** (antenna, wingtips) | Eliminates high‑frequency scattering | Preserve feature edges via manual mesh preprocessing |
| **Increased triangle aspect ratio** | May degrade EFIE accuracy, raise condition number | Set stricter `area_tol_rel` (e.g., 1e‑4) |

### RCS Pattern Features

- **Specular lobe**: Strong peak near θ = 0° (forward scattering) and θ = 180° (backscatter). Width inversely proportional to electrical size.
- **Nulls**: Directions where scattered field cancels; sensitive to geometry details.
- **Sidelobes**: Secondary maxima caused by diffraction at edges (wing, tail).

With coarse meshes, nulls may fill in and sidelobes may be suppressed. The overall pattern shape should still resemble the expected scattering physics.

### Monostatic RCS Validity

The reported monostatic value is a **single sample** on the angular grid. The `angular_error_deg` column gives the angular distance to the exact backscatter direction; if > grid resolution, consider interpolating between adjacent grid points.

---

## Troubleshooting

### Error: “Mesh has non‑manifold edges after repair”

**Cause:** CAD export contains topological errors that automatic repair cannot fix.

**Solutions:**

- Manually clean mesh in Blender/MeshLab before import.
- Set `strict_nonmanifold=false` (may produce unphysical currents).
- Use `drop_nonmanifold_triangles(mesh)` to remove offending triangles.

### Error: Coarsening fails to reach target RWG within max iterations

**Cause:** Mesh is already near minimal triangle count for its genus.

**Solutions:**

- Increase `target_rwg` to a more realistic value.
- Reduce `area_tol_rel` to allow more aggressive collapses.
- Manually simplify mesh in external tool (e.g., MeshLab’s quadratic edge collapse).

### Warning: “Mesh preview generation failed”

**Cause:** Visualization backend (GLMakie) not installed or OpenGL issues.

**Solutions:**

- Ignore – preview is optional.
- Install GLMakie: `] add GLMakie` and rerun.
- Use `plot_mesh_wireframe` separately to generate plots.

### Poor Residual (> 1e‑10)

**Cause:** Ill‑conditioned $Z$ matrix due to coarsening‑induced poorly‑shaped triangles.

**Solutions:**

- Enable preconditioning: modify script to call `make_left_preconditioner`.
- Increase `target_rwg` to improve mesh quality.

### Monostatic RCS varies wildly with small incidence‑angle changes

**Cause:** Coarse angular grid misses sharp backscatter peaks.

**Solutions:**

- Use finer spherical grid (`make_sph_grid(361, 72)`).
- Interpolate RCS near backscatter direction using neighboring samples.

---

## Code Mapping

| Task | Function | Source File | Key Lines |
|------|----------|-------------|-----------|
| **OBJ import** | `read_obj_mesh` | `src/Mesh.jl` | 100–120 |
| **Mesh repair** | `repair_mesh_for_simulation` | `src/Mesh.jl` | 300–350 |
| **Coarsening** | `coarsen_mesh_to_target_rwg` | `src/Mesh.jl` | 400–450 |
| **RWG building** | `build_rwg` | `src/RWG.jl` | 50–80 |
| **Memory estimate** | `estimate_dense_matrix_gib` | `src/Mesh.jl` | 200–220 |
| **Mesh preview** | `save_mesh_preview` | `src/Visualization.jl` | 150–200 |
| **Complete pipeline (repair + solve + plot)** | `ex_obj_rcs_pipeline.jl` | `examples/` | full script (`full`, `repair`, `plot` subcommands) |
| **Plotting-only mode** | `ex_obj_rcs_pipeline.jl plot ...` | `examples/` | same script, plot subcommand |

---

## Exercises

### Basic (60 minutes)

1. **Run the pipeline** with a simple sphere OBJ (generate with `write_obj_mesh`). Compare monostatic RCS with the Mie solution (Tutorial 4). How does coarsening affect accuracy?
2. **Vary target RWG** (200, 400, 600) and plot monostatic RCS vs unknown count. Is there convergence?
3. **Inspect coarsening artifacts**: Load the generated `<tag>_coarse.obj` (for the default run: `data/demo_aircraft_coarse.obj`) in a mesh viewer and identify regions where triangle density is disproportionately reduced.

### Practical (90 minutes)

1. **Import a real CAD model** (e.g., vehicle, drone). Adjust `scale_to_m` to achieve plausible physical size (e.g., 5 m wingspan). Run the pipeline and discuss plausible vs unphysical RCS features.
2. **Implement adaptive coarsening**: Modify the script to coarsen until memory estimate ≤ 2 GiB (instead of fixed RWG target). Use `estimate_dense_matrix_gib` in the loop.
3. **Add incidence‑angle sweep**: Modify `ex_obj_rcs_pipeline.jl` (`run_full`) to compute monostatic RCS for 5 incidence directions (θ = 0°, 45°, 90°, 135°, 180°). Plot RCS vs incidence angle.

### Advanced (2 hours)

1. **Compare with fast‑method prototype**: Replace the dense direct solve with an iterative solver (GMRES) and left preconditioning. Measure time/memory vs dense solve for N = 500, 1000, 2000.
2. **Quantify coarsening error**: For a sphere, compute MAE(dB) between RCS of original mesh and coarsened mesh (same frequency). Plot error vs coarsening ratio (triangles‑after / triangles‑before).
3. **Extend to impedance boundary condition**: Modify the pipeline to simulate a partially‑absorbing airplane (e.g., carbon‑composite skin) by adding resistive impedance sheets. Use `assemble_full_Z` with `reactive=false` and small θ values (~100 Ω). Compare RCS patterns with PEC case.

---

## Tutorial Checklist

Before applying the pipeline to your own CAD models, ensure you can:

- [ ] **Run the demo** with the provided sphere OBJ and obtain plausible RCS outputs.
- [ ] **Interpret memory estimates** and adjust target RWG accordingly.
- [ ] **Identify mesh defects** in the repair log and know how to fix them externally.
- [ ] **Explain trade‑offs** of coarsening: which scattering features are preserved, which are lost.
- [ ] **Generate and interpret** the heuristic RCS plots, noting the effect of angular sampling.

---

## Further Reading

- **Paper Section 6.2** – Complex‑platform workflow and coarsening strategy.
- **`src/Mesh.jl`** – Implementation of mesh repair and coarsening algorithms.
- **`src/Visualization.jl`** – Mesh preview and wireframe plotting utilities.
- **Tutorial 6: OBJ Repair Workflow** – Deep dive into mesh repair and quality metrics.
- **Advanced Workflows, Chapter 2: Large‑Problem Strategy** – Scaling beyond dense‑matrix limits.
