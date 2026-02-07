# ex_convergence.jl — Mesh refinement convergence and conditioning study
#
# Demonstrates:
# 1. EFIE condition number growth under mesh refinement
# 2. Far-field pattern convergence
# 3. Gradient accuracy under refinement
# 4. Energy conservation diagnostics
#
# Run: julia --project=. examples/ex_convergence.jl

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
println("Mesh Refinement Convergence & Conditioning Study")
println("="^60)

# Problem parameters
freq = 3e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668

Lx, Ly = 0.1, 0.1   # 10 cm × 10 cm plate (1λ × 1λ)

# Mesh refinement levels
mesh_sizes = [2, 3, 4, 5, 6, 8]

# Plane wave excitation
k_vec = Vec3(0.0, 0.0, -k)
E0    = 1.0
pol_inc = Vec3(1.0, 0.0, 0.0)

# Far-field grid (shared across refinement levels)
grid = make_sph_grid(16, 32)
NΩ = length(grid.w)
pol_mat = pol_linear_x(grid)

# Storage for convergence data
results = DataFrame(
    Nx     = Int[],
    Nv     = Int[],
    Nt     = Int[],
    N_rwg  = Int[],
    cond_Z = Float64[],
    P_rad  = Float64[],
    P_in   = Float64[],
    energy_ratio = Float64[],
    J_pec  = Float64[],
    max_grad_err = Float64[]
)

println("\n── Running refinement sweep ──\n")

for Nx in mesh_sizes
    Ny = Nx

    mesh = make_rect_plate(Lx, Ly, Nx, Ny)
    rwg  = build_rwg(mesh)
    N    = rwg.nedges
    Nt   = ntriangles(mesh)

    # EFIE assembly
    Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
    cond_info = condition_diagnostics(Z_efie)

    # Excitation
    v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3)

    # PEC solve
    I_pec = Z_efie \ v

    # Far-field
    G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
    E_ff = compute_farfield(G_mat, I_pec, NΩ)

    # Energy diagnostics
    P_rad = radiated_power(E_ff, grid)
    P_in  = input_power(I_pec, v)
    e_ratio = P_rad / P_in

    # Objective with broadside Q
    mask = cap_mask(grid; theta_max=30 * π / 180)
    Q = build_Q(G_mat, grid, pol_mat; mask=mask)
    J_pec = real(dot(I_pec, Q * I_pec))

    # Gradient check: impedance at one patch
    partition = PatchPartition(collect(1:Nt), Nt)
    Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
    theta_test = fill(200.0, Nt)
    Z_full = assemble_full_Z(Z_efie, Mp, theta_test)
    I_imp  = Z_full \ v
    lambda = Z_full' \ (Q * I_imp)
    g_adj  = gradient_impedance(Mp, I_imp, lambda)

    # FD verification on a few parameters
    function J_of_theta(theta_vec)
        Z_t = copy(Z_efie)
        for p in eachindex(theta_vec)
            Z_t .-= theta_vec[p] .* Mp[p]
        end
        I_t = Z_t \ v
        return real(dot(I_t, Q * I_t))
    end

    n_check = min(5, Nt)
    max_gerr = 0.0
    for p in 1:n_check
        g_fd = fd_grad(J_of_theta, theta_test, p; h=1e-5)
        rel_err = abs(g_adj[p] - g_fd) / max(abs(g_adj[p]), 1e-30)
        max_gerr = max(max_gerr, rel_err)
    end

    push!(results, (Nx, nvertices(mesh), Nt, N,
                    cond_info.cond, P_rad, P_in, e_ratio,
                    J_pec, max_gerr))

    println("  Nx=$Nx: Nv=$(nvertices(mesh)), Nt=$Nt, N=$N, " *
            "cond=$(round(cond_info.cond, sigdigits=3)), " *
            "P_rad/P_in=$(round(e_ratio, sigdigits=4)), " *
            "grad_err=$(round(max_gerr, sigdigits=3))")
end

CSV.write(joinpath(DATADIR, "convergence_study.csv"), results)

# ─────────────────────────────────────────────────
# Analysis summary
# ─────────────────────────────────────────────────
println("\n── Summary ──")
println("  Condition numbers: $(round.(results.cond_Z, sigdigits=3))")
println("  Energy ratios:     $(round.(results.energy_ratio, sigdigits=4))")
println("  Max gradient errors: $(round.(results.max_grad_err, sigdigits=3))")

# Check condition number growth rate
if length(results.cond_Z) >= 2
    # Condition number should grow roughly as O(h^{-2}) for EFIE
    h_vals = Lx ./ results.Nx
    # Fit log-log slope
    log_h = log.(h_vals)
    log_kappa = log.(results.cond_Z)
    n_pts = length(log_h)
    if n_pts >= 3
        # Simple linear regression
        x_mean = sum(log_h) / n_pts
        y_mean = sum(log_kappa) / n_pts
        slope = sum((log_h .- x_mean) .* (log_kappa .- y_mean)) /
                sum((log_h .- x_mean).^2)
        println("  Condition number growth rate: κ ~ h^($(round(slope, sigdigits=2)))")
        println("  (Expected: ~h^{-2} to ~h^{-3} for EFIE)")
    end
end

# Gradient accuracy should remain good across refinements
if all(results.max_grad_err .< 1e-3)
    println("  Gradient verification PASSED across all mesh levels")
else
    println("  WARNING: gradient accuracy degraded at fine meshes")
end

println("\n" * "="^60)
println("CONVERGENCE STUDY COMPLETE")
println("="^60)
println("Data saved to: $(joinpath(DATADIR, "convergence_study.csv"))")
