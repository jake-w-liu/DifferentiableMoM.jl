# ex_pec_sphere_mie_benchmark.jl — MoM vs Mie benchmark for PEC sphere RCS
#
# Run with fallback sphere mesh (default 2.0 GHz):
#   julia --project=. examples/ex_pec_sphere_mie_benchmark.jl
#
# Run with external OBJ sphere mesh:
#   julia --project=. examples/ex_pec_sphere_mie_benchmark.jl path/to/sphere.obj
#
# Run with external OBJ and custom frequency (GHz):
#   julia --project=. examples/ex_pec_sphere_mie_benchmark.jl path/to/sphere.obj 3.0

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using Statistics
using CSV
using DataFrames
using PlotlySupply

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
const FIGDIR = joinpath(@__DIR__, "..", "figs")
mkpath(DATADIR)
mkpath(FIGDIR)

function _save_pair(fig, basepath::AbstractString; width::Int, height::Int)
    png = basepath * ".png"
    pdf = basepath * ".pdf"
    savefig(fig, png; width = width, height = height)
    savefig(fig, pdf; width = width, height = height)
    return (png_path = png, pdf_path = pdf)
end

function write_icosphere_obj(path::AbstractString; radius::Float64=0.05, subdivisions::Int=2)
    ϕ = (1 + sqrt(5.0)) / 2
    verts0 = [
        (-1.0,  ϕ, 0.0), ( 1.0,  ϕ, 0.0), (-1.0, -ϕ, 0.0), ( 1.0, -ϕ, 0.0),
        ( 0.0, -1.0, ϕ), ( 0.0,  1.0, ϕ), ( 0.0, -1.0,-ϕ), ( 0.0,  1.0,-ϕ),
        ( ϕ, 0.0, -1.0), ( ϕ, 0.0,  1.0), (-ϕ, 0.0, -1.0), (-ϕ, 0.0,  1.0),
    ]
    faces0 = [
        (1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        (2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        (4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        (5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    ]

    verts = [Vec3(v...) / norm(Vec3(v...)) for v in verts0]
    faces = collect(faces0)

    for _ in 1:subdivisions
        edge_mid = Dict{Tuple{Int,Int},Int}()
        new_faces = NTuple{3,Int}[]

        function midpoint_index(i::Int, j::Int)
            key = i < j ? (i, j) : (j, i)
            if haskey(edge_mid, key)
                return edge_mid[key]
            end
            vmid = (verts[i] + verts[j]) / 2
            vmid /= norm(vmid)
            push!(verts, vmid)
            idx = length(verts)
            edge_mid[key] = idx
            return idx
        end

        for (i, j, k) in faces
            a = midpoint_index(i, j)
            b = midpoint_index(j, k)
            c = midpoint_index(k, i)
            push!(new_faces, (i, a, c))
            push!(new_faces, (j, b, a))
            push!(new_faces, (k, c, b))
            push!(new_faces, (a, b, c))
        end

        faces = new_faces
    end

    open(path, "w") do io
        println(io, "# Icosphere fallback mesh")
        for v in verts
            println(io, "v $(radius*v[1]) $(radius*v[2]) $(radius*v[3])")
        end
        for (i, j, k) in faces
            println(io, "f $i $j $k")
        end
    end
end

function estimate_sphere_radius(mesh::TriMesh)
    ctr = vec(mean(mesh.xyz, dims=2))
    radii = [norm(Vec3(mesh.xyz[:, i]) - Vec3(ctr)) for i in 1:nvertices(mesh)]
    return mean(radii), std(radii), Vec3(ctr)
end

println("="^60)
println("PEC Sphere Benchmark: MoM vs Mie")
println("="^60)

mesh_path = if isempty(ARGS)
    fallback = joinpath(DATADIR, "sphere_fallback_icosa.obj")
    write_icosphere_obj(fallback; radius=0.05, subdivisions=2)
    println("No mesh path provided. Using fallback OBJ: $fallback")
    fallback
else
    ARGS[1]
end

mesh = read_obj_mesh(mesh_path)
rwg = build_rwg(mesh)

freq = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) * 1e9 : 2e9
c0 = 299792458.0
λ0 = c0 / freq
k = 2π / λ0
eta0 = 376.730313668

k_vec = Vec3(0.0, 0.0, -k)
khat_inc = k_vec / norm(k_vec)
pol = Vec3(1.0, 0.0, 0.0)

println("\n── Geometry/mesh ──")
println("  Mesh: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles, $(rwg.nedges) RWG")
println("  Frequency: $(round(freq/1e9, digits=3)) GHz")
a_est, a_std, ctr = estimate_sphere_radius(mesh)
println("  Estimated center: ($(ctr[1]), $(ctr[2]), $(ctr[3])) m")
println("  Estimated radius: $(round(a_est, digits=6)) m  (std=$(round(a_std, digits=6)) m)")

println("\n── MoM solve ──")
Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
v_legacy = assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, pol; quad_order=3)
v = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, 1.0, pol); quad_order=3)
rhs_diff = norm(v - v_legacy) / max(norm(v_legacy), 1e-30)
println("  RHS new-vs-legacy relative difference: $rhs_diff")
I = solve_forward(Z, v)
res = norm(Z * I - v) / max(norm(v), 1e-30)
println("  Relative residual: $res")

grid = make_sph_grid(181, 72)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
E_ff = compute_farfield(G_mat, I, length(grid.w))
σ_mom = bistatic_rcs(E_ff; E0=1.0)

phi_target = grid.phi[argmin(grid.phi)]
phi_cut_idx = [q for q in 1:length(grid.w) if abs(grid.phi[q] - phi_target) < 1e-12]
perm = sortperm(grid.theta[phi_cut_idx])
cut_idx = phi_cut_idx[perm]

θ = grid.theta[cut_idx]
γ = acos.(clamp.(dot.(Ref(khat_inc), [Vec3(grid.rhat[:, q]) for q in cut_idx]), -1.0, 1.0))

σ_mie = zeros(Float64, length(cut_idx))
for (i, q) in enumerate(cut_idx)
    rhat = Vec3(grid.rhat[:, q])
    σ_mie[i] = mie_bistatic_rcs_pec(k, a_est, khat_inc, pol, rhat)
end

σ_mom_cut = σ_mom[cut_idx]
dB_mom = 10 .* log10.(max.(σ_mom_cut, 1e-30))
dB_mie = 10 .* log10.(max.(σ_mie, 1e-30))
ΔdB = dB_mom .- dB_mie

rmse_db = sqrt(mean(abs2, ΔdB))
mae_db = mean(abs.(ΔdB))
max_abs_db = maximum(abs.(ΔdB))

σ_bs_mom = backscatter_rcs(E_ff, grid, khat_inc; E0=1.0).sigma
σ_bs_mie = mie_bistatic_rcs_pec(k, a_est, khat_inc, pol, -khat_inc)
Δbs_db = 10 * log10(max(σ_bs_mom, 1e-30)) - 10 * log10(max(σ_bs_mie, 1e-30))

println("\n── MoM vs Mie summary (φ=$(round(rad2deg(phi_target), digits=2))° cut) ──")
println("  MAE(dB):  $mae_db")
println("  RMSE(dB): $rmse_db")
println("  Max |Δ|(dB): $max_abs_db")
println("  Backscatter Δ(dB): $Δbs_db")

df_cmp = DataFrame(
    phi_cut_deg = fill(rad2deg(phi_target), length(cut_idx)),
    theta_global_deg = rad2deg.(θ),
    gamma_deg = rad2deg.(γ),
    sigma_mom_m2 = σ_mom_cut,
    sigma_mie_m2 = σ_mie,
    sigma_mom_dBsm = dB_mom,
    sigma_mie_dBsm = dB_mie,
    delta_dB = ΔdB,
)
csv_cmp = joinpath(DATADIR, "sphere_mie_benchmark_phi_cut.csv")
CSV.write(csv_cmp, df_cmp)

df_summary = DataFrame(
    metric = ["mae_db", "rmse_db", "max_abs_db", "backscatter_delta_db", "radius_est_m", "radius_std_m"],
    value = [mae_db, rmse_db, max_abs_db, Δbs_db, a_est, a_std],
)
csv_sum = joinpath(DATADIR, "sphere_mie_benchmark_summary.csv")
CSV.write(csv_sum, df_summary)

width = 900
height = 840
titles = reshape(["PEC sphere: MoM vs Mie (φ=$(round(rad2deg(phi_target), digits=2))° cut)", "Deviation"], 2, 1)
sf = subplots(2, 1; sync = false, width = width, height = height, subplot_titles = titles)

plot_scatter!(
    sf,
    rad2deg.(γ),
    [dB_mom, dB_mie];
    color = ["blue", "red"],
    dash = ["", "dash"],
    legend = ["MoM", "Mie (PEC sphere)"],
    xlabel = "Scattering angle γ (deg)",
    ylabel = "Bistatic RCS (dBsm)",
)

subplot!(sf, 2, 1)
plot_scatter!(
    sf,
    rad2deg.(γ),
    ΔdB;
    color = "black",
    legend = "Δ(dB) = MoM - Mie",
    xlabel = "Scattering angle γ (deg)",
    ylabel = "Δ(dB)",
)
add_hline!(sf.plot, 0.0; row = 2, col = 1, line_color = "gray", line_dash = "dot", line_width = 1.2)

subplot_legends!(sf; position = :topright)
plot_out = _save_pair(sf.plot, joinpath(FIGDIR, "sphere_mie_benchmark_phi_cut"); width = width, height = height)
png_path = plot_out.png_path
pdf_path = plot_out.pdf_path

println("\n── Outputs ──")
println("  Comparison CSV: $csv_cmp")
println("  Summary CSV:    $csv_sum")
println("  Heuristic plot PNG: $png_path")
println("  Heuristic plot PDF: $pdf_path")
println("\nBenchmark complete.")
