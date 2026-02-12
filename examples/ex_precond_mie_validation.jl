# ex_precond_mie_validation.jl — Mie sphere accuracy with GMRES solver
#
# Validates that the randomized-preconditioned GMRES solver produces
# bistatic RCS results matching both the direct LU solver and Mie theory.
#
# Key metrics: MAE(dB), RMSE(dB), max|Δ|(dB) of RCS vs Mie reference.
#
# Run: julia --project=. examples/ex_precond_mie_validation.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

println("="^60)
println("Mie Sphere Validation: Direct LU vs GMRES")
println("="^60)

# Generate icosphere mesh
function write_icosphere_obj(path::AbstractString; radius::Float64=0.05, subdivisions::Int=2)
    ϕ = (1 + sqrt(5.0)) / 2
    verts = [
        (-1.0, ϕ, 0.0), (1.0, ϕ, 0.0), (-1.0, -ϕ, 0.0), (1.0, -ϕ, 0.0),
        (0.0, -1.0, ϕ), (0.0, 1.0, ϕ), (0.0, -1.0, -ϕ), (0.0, 1.0, -ϕ),
        (ϕ, 0.0, -1.0), (ϕ, 0.0, 1.0), (-ϕ, 0.0, -1.0), (-ϕ, 0.0, 1.0),
    ]
    verts = [Vec3(v...) / norm(Vec3(v...)) for v in verts]
    faces = [
        (1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        (2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        (4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        (5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    ]
    for _ in 1:subdivisions
        edge_mid = Dict{Tuple{Int,Int},Int}()
        new_faces = NTuple{3,Int}[]
        function midpoint_index(i::Int, j::Int)
            key = i < j ? (i, j) : (j, i)
            haskey(edge_mid, key) && return edge_mid[key]
            vmid = (verts[i] + verts[j]) / 2
            vmid /= norm(vmid)
            push!(verts, vmid)
            edge_mid[key] = length(verts)
            return length(verts)
        end
        for (i, j, k) in faces
            a = midpoint_index(i, j)
            b = midpoint_index(j, k)
            c = midpoint_index(k, i)
            push!(new_faces, (i, a, c), (j, b, a), (k, c, b), (a, b, c))
        end
        faces = new_faces
    end
    open(path, "w") do io
        for v in verts; println(io, "v $(radius*v[1]) $(radius*v[2]) $(radius*v[3])"); end
        for (i, j, k) in faces; println(io, "f $i $j $k"); end
    end
end

# Parameters
freq = 2.0e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668
radius = 0.05  # 5 cm sphere

# Mesh
obj_path = joinpath(DATADIR, "tmp_mie_precond_sphere.obj")
write_icosphere_obj(obj_path; radius=radius, subdivisions=2)
mesh = read_obj_mesh(obj_path)
rwg  = build_rwg(mesh)
N    = rwg.nedges

# Sphere effective radius
ctr = vec(sum(mesh.xyz, dims=2) ./ nvertices(mesh))
radii = [norm(Vec3(mesh.xyz[:, i]) - Vec3(ctr)) for i in 1:nvertices(mesh)]
a_eff = sum(radii) / length(radii)

println("\n── Setup ──")
println("  Frequency: $(freq/1e9) GHz,  ka = $(round(k*a_eff, digits=3))")
println("  Mesh: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles, $N RWG")

# Assemble
k_vec = Vec3(0.0, 0.0, -k)
khat  = k_vec / norm(k_vec)
pol   = Vec3(1.0, 0.0, 0.0)

Z = Matrix{ComplexF64}(assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0))
v = Vector{ComplexF64}(assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, pol; quad_order=3))

# Far-field grid
grid = make_sph_grid(181, 72)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)

# ── Direct LU solve ──
println("\n── Direct LU solve ──")
t_direct = @elapsed I_direct = Z \ v
E_direct = compute_farfield(G_mat, I_direct, length(grid.w))
σ_direct = bistatic_rcs(E_direct; E0=1.0)
println("  Time: $(round(t_direct, sigdigits=3))s")

# ── GMRES + randomized preconditioner ──
precond_ranks = [10, 20]
results = DataFrame(
    method    = String[],
    precond_k = Int[],
    time_s    = Float64[],
    iters     = Int[],
    mae_dB    = Float64[],
    rmse_dB   = Float64[],
    maxabs_dB = Float64[],
    sol_rel_err = Float64[],
)

for k_rank in precond_ranks
    println("\n── GMRES + preconditioner rank=$k_rank ──")
    t_gmres = @elapsed begin
        P = build_randomized_preconditioner(Z, k_rank; seed=42)
        I_gmres, stats = solve_gmres(Z, v; preconditioner=P, tol=1e-10, maxiter=500)
    end
    E_gmres = compute_farfield(G_mat, I_gmres, length(grid.w))
    σ_gmres = bistatic_rcs(E_gmres; E0=1.0)

    sol_err = norm(I_gmres - I_direct) / max(norm(I_direct), 1e-30)
    println("  Time: $(round(t_gmres, sigdigits=3))s  iters: $(stats.niter)")
    println("  Solution rel error vs direct: $(round(sol_err, sigdigits=3))")

    # Compare RCS against Mie theory on phi=0 cut
    phi_target = grid.phi[argmin(grid.phi)]
    idx_cut = [q for q in 1:length(grid.w) if abs(grid.phi[q] - phi_target) < 1e-12]
    idx_cut = idx_cut[sortperm(grid.theta[idx_cut])]

    σ_mie = [mie_bistatic_rcs_pec(k, a_eff, khat, pol, Vec3(grid.rhat[:, q]))
             for q in idx_cut]

    dB_gmres = 10 .* log10.(max.(σ_gmres[idx_cut], 1e-30))
    dB_mie   = 10 .* log10.(max.(σ_mie, 1e-30))
    ΔdB = dB_gmres .- dB_mie

    mae  = sum(abs.(ΔdB)) / length(ΔdB)
    rmse = sqrt(sum(abs2, ΔdB) / length(ΔdB))
    maxabs = maximum(abs.(ΔdB))

    println("  vs Mie:  MAE=$(round(mae, sigdigits=3)) dB  RMSE=$(round(rmse, sigdigits=3)) dB  max|Δ|=$(round(maxabs, sigdigits=3)) dB")

    push!(results, (method="gmres_k$(k_rank)", precond_k=k_rank,
                     time_s=t_gmres, iters=stats.niter,
                     mae_dB=mae, rmse_dB=rmse, maxabs_dB=maxabs,
                     sol_rel_err=sol_err))
end

# Direct solver vs Mie (reference)
phi_target = grid.phi[argmin(grid.phi)]
idx_cut = [q for q in 1:length(grid.w) if abs(grid.phi[q] - phi_target) < 1e-12]
idx_cut = idx_cut[sortperm(grid.theta[idx_cut])]
σ_mie_ref = [mie_bistatic_rcs_pec(k, a_eff, khat, pol, Vec3(grid.rhat[:, q]))
             for q in idx_cut]
dB_dir = 10 .* log10.(max.(σ_direct[idx_cut], 1e-30))
dB_mie_ref = 10 .* log10.(max.(σ_mie_ref, 1e-30))
ΔdB_dir = dB_dir .- dB_mie_ref
mae_dir = sum(abs.(ΔdB_dir)) / length(ΔdB_dir)
rmse_dir = sqrt(sum(abs2, ΔdB_dir) / length(ΔdB_dir))
maxabs_dir = maximum(abs.(ΔdB_dir))

push!(results, (method="direct_LU", precond_k=0,
                 time_s=t_direct, iters=0,
                 mae_dB=mae_dir, rmse_dB=rmse_dir, maxabs_dB=maxabs_dir,
                 sol_rel_err=0.0))

# Save
CSV.write(joinpath(DATADIR, "precond_mie_validation.csv"), results)

# Save RCS cut for plotting
dB_mie_cut = 10 .* log10.(max.(σ_mie_ref, 1e-30))
df_rcs = DataFrame(
    theta_deg = rad2deg.(grid.theta[idx_cut]),
    rcs_mie_dB = dB_mie_cut,
    rcs_direct_dB = dB_dir,
)
CSV.write(joinpath(DATADIR, "precond_mie_rcs_cut.csv"), df_rcs)

println("\n" * "="^60)
println("MIE VALIDATION COMPLETE")
println("="^60)
println("\nResults saved to: $(joinpath(DATADIR, "precond_mie_validation.csv"))")
