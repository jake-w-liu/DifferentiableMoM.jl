# ex_pec_sphere_rcs.jl — PEC sphere scattering example (bistatic + monostatic RCS)
#
# Run with an external OBJ mesh:
#   julia --project=. examples/ex_pec_sphere_rcs.jl path/to/sphere.obj
#
# Run with the built-in fallback sphere mesh:
#   julia --project=. examples/ex_pec_sphere_rcs.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

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

println("="^60)
println("PEC Sphere Scattering Example (RCS)")
println("="^60)

mesh_path = if isempty(ARGS)
    fallback_path = joinpath(DATADIR, "sphere_fallback_icosa.obj")
    write_icosphere_obj(fallback_path; radius=0.05, subdivisions=2)
    println("No mesh path provided. Using fallback OBJ: $fallback_path")
    fallback_path
else
    ARGS[1]
end

mesh = read_obj_mesh(mesh_path)
rwg = build_rwg(mesh)

println("\n── Mesh summary ──")
println("  Path: $mesh_path")
println("  Vertices: $(nvertices(mesh))")
println("  Triangles: $(ntriangles(mesh))")
println("  RWG basis functions: $(rwg.nedges)")

freq = 3e9
c0 = 299792458.0
λ0 = c0 / freq
k = 2π / λ0
eta0 = 376.730313668

println("\n── Solving PEC scattering ──")
println("  Frequency: $(freq/1e9) GHz (λ=$(round(λ0, digits=4)) m)")

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
k_vec = Vec3(0.0, 0.0, -k)
pol = Vec3(1.0, 0.0, 0.0)
E0 = 1.0
v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)
I = solve_forward(Z_efie, v)

res = norm(Z_efie * I - v) / max(norm(v), 1e-30)
println("  Relative solve residual: $res")

grid = make_sph_grid(181, 72)
NΩ = length(grid.w)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
E_ff = compute_farfield(G_mat, I, NΩ)
σ = bistatic_rcs(E_ff; E0=E0)

# Single φ cut closest to 0°
phi_target = grid.phi[argmin(grid.phi)]
phi0_idx = [q for q in 1:NΩ if abs(grid.phi[q] - phi_target) < 1e-12]
perm = sortperm(grid.theta[phi0_idx])
phi0_sorted = phi0_idx[perm]

df_bistatic = DataFrame(
    phi_cut_deg = fill(rad2deg(phi_target), length(phi0_sorted)),
    theta_deg = rad2deg.(grid.theta[phi0_sorted]),
    sigma_m2 = σ[phi0_sorted],
    sigma_dBsm = 10 .* log10.(max.(σ[phi0_sorted], 1e-30)),
)
csv_bistatic = joinpath(DATADIR, "sphere_bistatic_rcs_phi0.csv")
CSV.write(csv_bistatic, df_bistatic)

# Monostatic (backscatter for this incidence direction)
khat_inc = k_vec / norm(k_vec)
bs = backscatter_rcs(E_ff, grid, khat_inc; E0=E0)
df_mono = DataFrame(
    sigma_m2 = [bs.sigma],
    sigma_dBsm = [10 * log10(max(bs.sigma, 1e-30))],
    theta_obs_deg = [rad2deg(bs.theta)],
    phi_obs_deg = [rad2deg(bs.phi)],
    sample_angular_error_deg = [bs.angular_error_deg],
)
csv_mono = joinpath(DATADIR, "sphere_monostatic_rcs.csv")
CSV.write(csv_mono, df_mono)

println("\n── Outputs ──")
println("  Bistatic φ≈0° cut: $csv_bistatic")
println("  Monostatic backscatter: $csv_mono")
println("  Monostatic σ = $(df_mono.sigma_m2[1]) m² ($(round(df_mono.sigma_dBsm[1], digits=3)) dBsm)")

println("\nExample complete.")
