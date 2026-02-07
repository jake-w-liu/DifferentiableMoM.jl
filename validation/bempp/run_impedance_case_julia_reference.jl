#!/usr/bin/env julia

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "..", "data")
mkpath(DATADIR)

function parse_float_flag(args::Vector{String}, name::String, default::Float64)
    for i in eachindex(args)
        if args[i] == name
            i < length(args) || error("Missing value for $(name)")
            return parse(Float64, args[i + 1])
        end
    end
    return default
end

function parse_int_flag(args::Vector{String}, name::String, default::Int)
    for i in eachindex(args)
        if args[i] == name
            i < length(args) || error("Missing value for $(name)")
            return parse(Int, args[i + 1])
        end
    end
    return default
end

freq = 3e9
c0 = 299792458.0
lambda0 = c0 / freq
k = 2π / lambda0
eta0 = 376.730313668

Lx = 4 * lambda0
Ly = 4 * lambda0
Nx, Ny = 12, 12

theta_uniform = parse_float_flag(ARGS, "--theta-ohm", 200.0)  # Zs = i*theta_uniform [ohm]
n_theta = parse_int_flag(ARGS, "--n-theta", 180)
n_phi = parse_int_flag(ARGS, "--n-phi", 72)

mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg = build_rwg(mesh)
Nt = ntriangles(mesh)
N = rwg.nedges

println("Julia impedance-reference case")
println("  Mesh: N=$N RWG, Nt=$Nt triangles")

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

theta_vec = fill(theta_uniform, Nt)
Z_imp = assemble_full_Z(Z_efie, Mp, theta_vec; reactive=true)

k_vec = Vec3(0.0, 0.0, -k)
E0 = 1.0
pol_inc = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3)
I_imp = Z_imp \ v

grid = make_sph_grid(n_theta, n_phi)
NΩ = length(grid.w)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
E_ff_imp = compute_farfield(G_mat, I_imp, NΩ)

ff_power_imp = [real(dot(E_ff_imp[:, q], E_ff_imp[:, q])) for q in 1:NΩ]
P_sphere_imp = sum(ff_power_imp[q] * grid.w[q] for q in 1:NΩ)
D_imp = [4π * ff_power_imp[q] / P_sphere_imp for q in 1:NΩ]
dir_imp_dBi = 10 .* log10.(max.(D_imp, 1e-30))

theta_steer = 30.0 * π / 180
phi_steer = 0.0
steer_rhat = Vec3(sin(theta_steer) * cos(phi_steer),
                  sin(theta_steer) * sin(phi_steer),
                  cos(theta_steer))
in_target = BitVector([begin
    rh = Vec3(grid.rhat[:, q])
    angle = acos(clamp(dot(rh, steer_rhat), -1.0, 1.0))
    angle <= 5.0 * π / 180
end for q in 1:NΩ])

df_ff = DataFrame(
    theta_deg = rad2deg.(grid.theta),
    phi_deg = rad2deg.(grid.phi),
    dir_julia_imp_dBi = dir_imp_dBi,
    in_target = in_target,
)
CSV.write(joinpath(DATADIR, "julia_impedance_farfield.csv"), df_ff)

dphi = 2π / 72
phi0_idx = [q for q in 1:NΩ if min(grid.phi[q], 2π - grid.phi[q]) <= dphi / 2 + 1e-10]
df_cut = DataFrame(
    theta_deg = rad2deg.(grid.theta[phi0_idx]),
    dir_julia_imp_dBi = dir_imp_dBi[phi0_idx],
)
CSV.write(joinpath(DATADIR, "julia_impedance_cut_phi0.csv"), df_cut)

println("Saved data/julia_impedance_farfield.csv")
println("Saved data/julia_impedance_cut_phi0.csv")
