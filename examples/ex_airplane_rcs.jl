# ex_airplane_rcs.jl — Airplane PEC scattering demo with automatic repair/coarsening
# and package-level mesh preview generation.
#
# Usage:
#   julia --project=. examples/ex_airplane_rcs.jl [input_obj] [freq_GHz] [scale_to_m] [target_rwg]
#
# Examples:
#   julia --project=. examples/ex_airplane_rcs.jl ../Airplane.obj
#   julia --project=. examples/ex_airplane_rcs.jl ../Airplane.obj 3.0 0.001 300

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
const FIGDIR = joinpath(@__DIR__, "..", "figs")
mkpath(DATADIR)
mkpath(FIGDIR)

input_path = length(ARGS) >= 1 ? ARGS[1] : joinpath("..", "Airplane.obj")
freq_ghz = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 3.0
scale_to_m = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-3
target_rwg = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 300

println("="^68)
println("Airplane PEC RCS Demo")
println("="^68)
println("Input OBJ   : $input_path")
println("Frequency   : $(freq_ghz) GHz")
println("Scale to m  : $scale_to_m")
println("Target RWG  : $target_rwg")

mesh_in = read_obj_mesh(input_path)
mesh_scaled = TriMesh(mesh_in.xyz .* scale_to_m, copy(mesh_in.tri))

repair0 = repair_mesh_for_simulation(
    mesh_scaled;
    allow_boundary=true,
    require_closed=false,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=true,
    strict_nonmanifold=true,
)
mesh0 = repair0.mesh
rwg0 = build_rwg(mesh0; precheck=true, allow_boundary=true)

println("\n── Imported mesh ──")
println("  Vertices: $(nvertices(mesh_in)) -> $(nvertices(mesh0)) (scaled/repaired)")
println("  Triangles: $(ntriangles(mesh_in)) -> $(ntriangles(mesh0))")
println("  RWG (before coarsen): $(rwg0.nedges)")
println("  Dense matrix size estimate: $(estimate_dense_matrix_gib(rwg0.nedges)) GiB")
println("  Repaired winding flips: $(length(repair0.flipped_triangles))")

mesh_use = mesh0
rwg_use = rwg0
if rwg0.nedges > target_rwg
    println("\n── Coarsening mesh for dense solve feasibility ──")
    coarse_result = coarsen_mesh_to_target_rwg(
        mesh0,
        target_rwg;
        max_iters=10,
        allow_boundary=true,
        require_closed=false,
        area_tol_rel=1e-12,
        strict_nonmanifold=true,
    )
    mesh_use = coarse_result.mesh
    rwg_use = build_rwg(mesh_use; precheck=true, allow_boundary=true)
    println("  Coarsened vertices : $(nvertices(mesh_use))")
    println("  Coarsened triangles: $(ntriangles(mesh_use))")
    println("  RWG after coarsen  : $(rwg_use.nedges)")
    println("  Dense matrix estimate: $(estimate_dense_matrix_gib(rwg_use.nedges)) GiB")
    println("  Coarsening iterations: $(coarse_result.iterations), target gap: $(coarse_result.best_gap)")
else
    println("\nNo coarsening needed.")
end

mesh_repaired_out = joinpath(DATADIR, "airplane_repaired.obj")
mesh_coarse_out = joinpath(DATADIR, "airplane_coarse.obj")
write_obj_mesh(mesh_repaired_out, mesh0; header="Scaled+repaired from $input_path")
write_obj_mesh(mesh_coarse_out, mesh_use; header="Coarsened for RCS demo from $input_path")

preview_png = ""
preview_pdf = ""
try
    seg_rep = mesh_wireframe_segments(mesh0)
    seg_coa = mesh_wireframe_segments(mesh_use)
    preview = save_mesh_preview(
        mesh0,
        mesh_use,
        joinpath(FIGDIR, "airplane_mesh_preview");
        title_a="Repaired mesh\nV=$(nvertices(mesh0)), T=$(ntriangles(mesh0)), E=$(seg_rep.n_edges)",
        title_b="Simulation mesh\nV=$(nvertices(mesh_use)), T=$(ntriangles(mesh_use)), E=$(seg_coa.n_edges)",
        color_a=:steelblue,
        color_b=:darkorange,
        camera=(30, 30),
        size=(1200, 520),
        linewidth=0.7,
        guidefontsize=10,
        tickfontsize=8,
        titlefontsize=10,
    )
    global preview_png = preview.png_path
    global preview_pdf = preview.pdf_path
catch err
    @warn "Mesh preview generation failed; continuing without preview files." exception=(err, catch_backtrace())
end

freq = freq_ghz * 1e9
c0 = 299792458.0
lambda0 = c0 / freq
k = 2π / lambda0
eta0 = 376.730313668

println("\n── Solving PEC scattering ──")
println("  λ0 = $(round(lambda0, digits=5)) m")
println("  Unknowns = $(rwg_use.nedges)")

t_asm = @elapsed Z_efie = assemble_Z_efie(mesh_use, rwg_use, k; quad_order=3, eta0=eta0)
println("  Assembly time: $(round(t_asm, digits=3)) s")

k_vec = Vec3(0.0, 0.0, -k)
pol = Vec3(1.0, 0.0, 0.0)
E0 = 1.0
v = assemble_v_plane_wave(mesh_use, rwg_use, k_vec, E0, pol; quad_order=3)

t_solve = @elapsed I = solve_forward(Z_efie, v)
res = norm(Z_efie * I - v) / max(norm(v), 1e-30)
println("  Solve time: $(round(t_solve, digits=3)) s")
println("  Relative residual: $res")

grid = make_sph_grid(121, 36)
NΩ = length(grid.w)
t_ff = @elapsed begin
    G_mat = radiation_vectors(mesh_use, rwg_use, grid, k; quad_order=3, eta0=eta0)
    E_ff = compute_farfield(G_mat, I, NΩ)
end
println("  Far-field time: $(round(t_ff, digits=3)) s")

σ = bistatic_rcs(E_ff; E0=E0)
khat_inc = k_vec / norm(k_vec)
bs = backscatter_rcs(E_ff, grid, khat_inc; E0=E0)

phi_target = grid.phi[argmin(abs.(grid.phi))]
phi_idx = [q for q in 1:NΩ if abs(grid.phi[q] - phi_target) < 1e-12]
perm = sortperm(grid.theta[phi_idx])
phi_sorted = phi_idx[perm]

df_cut = DataFrame(
    phi_cut_deg = fill(rad2deg(phi_target), length(phi_sorted)),
    theta_deg = rad2deg.(grid.theta[phi_sorted]),
    sigma_m2 = σ[phi_sorted],
    sigma_dBsm = 10 .* log10.(max.(σ[phi_sorted], 1e-30)),
)
df_bs = DataFrame(
    sigma_m2 = [bs.sigma],
    sigma_dBsm = [10 * log10(max(bs.sigma, 1e-30))],
    theta_obs_deg = [rad2deg(bs.theta)],
    phi_obs_deg = [rad2deg(bs.phi)],
    sample_angular_error_deg = [bs.angular_error_deg],
)
df_summary = DataFrame(
    input_obj = [input_path],
    freq_ghz = [freq_ghz],
    scale_to_m = [scale_to_m],
    rwg_unknowns = [rwg_use.nedges],
    vertices = [nvertices(mesh_use)],
    triangles = [ntriangles(mesh_use)],
    assembly_time_s = [t_asm],
    solve_time_s = [t_solve],
    farfield_time_s = [t_ff],
    residual = [res],
    monostatic_sigma_m2 = [bs.sigma],
    monostatic_sigma_dBsm = [10 * log10(max(bs.sigma, 1e-30))],
)

csv_cut = joinpath(DATADIR, "airplane_bistatic_rcs_phi0.csv")
csv_bs = joinpath(DATADIR, "airplane_monostatic_rcs.csv")
csv_summary = joinpath(DATADIR, "airplane_rcs_summary.csv")
CSV.write(csv_cut, df_cut)
CSV.write(csv_bs, df_bs)
CSV.write(csv_summary, df_summary)

println("\n── Outputs ──")
println("  Repaired mesh: $mesh_repaired_out")
println("  Coarsened mesh: $mesh_coarse_out")
if !isempty(preview_png)
    println("  Mesh preview PNG: $preview_png")
    println("  Mesh preview PDF: $preview_pdf")
end
println("  Bistatic φ≈0° cut: $csv_cut")
println("  Monostatic backscatter: $csv_bs")
println("  Run summary: $csv_summary")
println("  Monostatic σ = $(df_bs.sigma_m2[1]) m² ($(round(df_bs.sigma_dBsm[1], digits=3)) dBsm)")

println("\nDone.")
