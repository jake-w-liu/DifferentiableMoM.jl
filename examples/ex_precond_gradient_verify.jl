# ex_precond_gradient_verify.jl — Gradient accuracy with iterative solver
#
# Verifies that the adjoint gradient computed via GMRES + randomized
# preconditioner matches the finite-difference reference.
#
# Sweeps GMRES tolerance to show gradient error vs solver tolerance,
# demonstrating that gradient accuracy is controlled by GMRES convergence.
#
# Run: julia --project=. examples/ex_precond_gradient_verify.jl

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
println("Gradient Accuracy: Direct vs GMRES at Various Tolerances")
println("="^60)

# Problem setup
freq = 3e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668

Lx, Ly = 0.1, 0.1
Nx, Ny = 4, 4

mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges
Nt   = ntriangles(mesh)

println("\n── Setup ──")
println("  Mesh: $Nx×$Ny  N=$N  Nt=$Nt")

# Assembly
Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

k_vec = Vec3(0.0, 0.0, -k)
v = Vector{ComplexF64}(assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, Vec3(1.0, 0.0, 0.0); quad_order=3))

grid = make_sph_grid(16, 32)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
pol_mat = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=30 * π / 180)
Q = build_Q(G_mat, grid, pol_mat; mask=mask)

# Test point
theta_test = fill(200.0, Nt)
Z_full = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_test))

# FD reference objective function
function J_of_theta(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= theta_vec[p] .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

# Direct gradient (ground truth)
println("\n── Direct solver gradient ──")
I_direct = Z_full \ v
lam_direct = Z_full' \ (Q * I_direct)
g_direct = gradient_impedance(Mp, I_direct, lam_direct)

# FD gradient (cross-check)
n_check = min(Nt, 10)
g_fd = zeros(n_check)
for p in 1:n_check
    g_fd[p] = fd_grad(J_of_theta, theta_test, p; h=1e-5)
end
max_fd_err = maximum(abs.(g_direct[1:n_check] .- g_fd) ./ max.(abs.(g_direct[1:n_check]), 1e-30))
println("  Direct gradient vs FD: max rel err = $max_fd_err")

# GMRES gradient at various tolerances
gmres_tols = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
precond_rank = 15

results = DataFrame(
    gmres_tol      = Float64[],
    max_grad_err_vs_direct = Float64[],
    max_grad_err_vs_fd     = Float64[],
    mean_grad_err_vs_fd    = Float64[],
    fwd_iters      = Int[],
    adj_iters      = Int[],
)

println("\n── GMRES gradient at various tolerances ──")
for tol in gmres_tols
    P = build_randomized_preconditioner(Z_full, precond_rank; seed=42)

    I_gm, stats_fwd = solve_gmres(Z_full, v;
                                    preconditioner=P, tol=tol, maxiter=1000)
    rhs_adj = Q * I_gm
    lam_gm, stats_adj = solve_gmres_adjoint(Z_full, Vector{ComplexF64}(rhs_adj);
                                              preconditioner=P, tol=tol, maxiter=1000)
    g_gmres = gradient_impedance(Mp, I_gm, lam_gm)

    # Error vs direct
    err_vs_direct = maximum(abs.(g_gmres[1:n_check] .- g_direct[1:n_check]) ./
                            max.(abs.(g_direct[1:n_check]), 1e-30))

    # Error vs FD
    errs_vs_fd = [abs(g_gmres[p] - g_fd[p]) / max(abs(g_fd[p]), 1e-30) for p in 1:n_check]
    max_err_fd = maximum(errs_vs_fd)
    mean_err_fd = sum(errs_vs_fd) / n_check

    push!(results, (gmres_tol=tol,
                     max_grad_err_vs_direct=err_vs_direct,
                     max_grad_err_vs_fd=max_err_fd,
                     mean_grad_err_vs_fd=mean_err_fd,
                     fwd_iters=stats_fwd.niter,
                     adj_iters=stats_adj.niter))

    println("  tol=$tol  max_err_vs_direct=$(round(err_vs_direct, sigdigits=3))  " *
            "max_err_vs_fd=$(round(max_err_fd, sigdigits=3))  " *
            "fwd_iters=$(stats_fwd.niter)  adj_iters=$(stats_adj.niter)")
end

# Also test reactive impedance
println("\n── Reactive impedance gradient ──")
theta_reac = fill(150.0, Nt)
Z_reac = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_reac; reactive=true))

function J_of_theta_reac(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= (1im * theta_vec[p]) .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

I_reac = Z_reac \ v
lam_reac = Z_reac' \ (Q * I_reac)
g_reac_direct = gradient_impedance(Mp, I_reac, lam_reac; reactive=true)

P_reac = build_randomized_preconditioner(Z_reac, precond_rank; seed=42)
I_reac_gm, _ = solve_gmres(Z_reac, v; preconditioner=P_reac, tol=1e-10, maxiter=500)
lam_reac_gm, _ = solve_gmres_adjoint(Z_reac, Vector{ComplexF64}(Q * I_reac_gm);
                                       preconditioner=P_reac, tol=1e-10, maxiter=500)
g_reac_gmres = gradient_impedance(Mp, I_reac_gm, lam_reac_gm; reactive=true)

reac_errs = Float64[]
for p in 1:n_check
    g_fd_r = fd_grad(J_of_theta_reac, theta_reac, p; h=1e-5)
    push!(reac_errs, abs(g_reac_gmres[p] - g_fd_r) / max(abs(g_fd_r), 1e-30))
end
println("  Reactive max rel err (GMRES vs FD): $(maximum(reac_errs))")
println("  Reactive GMRES vs direct: $(norm(g_reac_gmres - g_reac_direct) / max(norm(g_reac_direct), 1e-30))")

# Save
CSV.write(joinpath(DATADIR, "precond_gradient_verification.csv"), results)

# Per-parameter breakdown at tightest tolerance
g_fd_full = [fd_grad(J_of_theta, theta_test, p; h=1e-5) for p in 1:n_check]
P_best = build_randomized_preconditioner(Z_full, precond_rank; seed=42)
I_best, _ = solve_gmres(Z_full, v; preconditioner=P_best, tol=1e-12, maxiter=1000)
lam_best, _ = solve_gmres_adjoint(Z_full, Vector{ComplexF64}(Q * I_best);
                                    preconditioner=P_best, tol=1e-12, maxiter=1000)
g_best = gradient_impedance(Mp, I_best, lam_best)

df_detail = DataFrame(
    param = 1:n_check,
    g_direct = g_direct[1:n_check],
    g_gmres  = g_best[1:n_check],
    g_fd     = g_fd_full,
    rel_err_gmres_vs_fd = [abs(g_best[p] - g_fd_full[p]) / max(abs(g_fd_full[p]), 1e-30) for p in 1:n_check],
    rel_err_direct_vs_fd = [abs(g_direct[p] - g_fd_full[p]) / max(abs(g_fd_full[p]), 1e-30) for p in 1:n_check],
)
CSV.write(joinpath(DATADIR, "precond_gradient_detail.csv"), df_detail)

println("\n" * "="^60)
println("GRADIENT VERIFICATION COMPLETE")
println("="^60)
