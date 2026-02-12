# ex_precond_inverse_design.jl — Beam-steering with randomized preconditioning
#
# Compares the beam-steering metasurface optimization using:
#   1. Direct LU solver (baseline from Paper 1)
#   2. GMRES + randomized preconditioner
#
# Shows wall-clock speedup and confirms both reach the same optimum.
#
# Run: julia --project=. examples/ex_precond_inverse_design.jl

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
println("Beam-Steering: Direct LU vs GMRES + Randomized Preconditioning")
println("="^60)

# Problem parameters
freq = 3e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668

# Plate: 4λ × 4λ
Lx = 4 * lambda0
Ly = 4 * lambda0
Nx, Ny = 12, 12

mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges
Nt   = ntriangles(mesh)

println("\n── Setup ──")
println("  Frequency: $(freq/1e9) GHz,  λ = $(round(lambda0*100, digits=2)) cm")
println("  Plate: $(round(Lx/lambda0, digits=1))λ × $(round(Ly/lambda0, digits=1))λ")
println("  Mesh: $N RWG basis functions, $Nt patches")

# Assembly
Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
k_vec = Vec3(0.0, 0.0, -k)
v = Vector{ComplexF64}(assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, Vec3(1.0, 0.0, 0.0); quad_order=3))

# Far-field objective: steer to 30°
theta_steer = 30.0 * π / 180
grid = make_sph_grid(180, 72)
NΩ = length(grid.w)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
pol_mat = pol_linear_x(grid)

steer_rhat = Vec3(sin(theta_steer), 0.0, cos(theta_steer))
mask = BitVector([begin
    rh = Vec3(grid.rhat[:, q])
    angle = acos(clamp(dot(rh, steer_rhat), -1.0, 1.0))
    angle <= 5.0 * π / 180
end for q in 1:NΩ])

Q_target = build_Q(G_mat, grid, pol_mat; mask=mask)
Q_total = build_Q(G_mat, grid, pol_mat)

# Initial impedance: phase gradient
tri_centers = [triangle_center(mesh, t) for t in 1:Nt]
cx = [tc[1] for tc in tri_centers]
x_center = (minimum(cx) + maximum(cx)) / 2
x_halfspan = (maximum(cx) - minimum(cx)) / 2
theta_init = [300.0 * (c - x_center) / x_halfspan for c in cx]
theta_bound = 500.0

maxiter_opt = 100
tol_opt = 1e-10

# PEC reference
I_pec = Z_efie \ v
P_target_pec = real(dot(I_pec, Q_target * I_pec))
P_total_pec  = real(dot(I_pec, Q_total * I_pec))
J_pec = P_target_pec / P_total_pec
println("  J_pec = $(round(J_pec*100, digits=2))%")

# ─────────────────────────────────────────────────
# Run 1: Direct LU optimization
# ─────────────────────────────────────────────────
println("\n── Optimization: Direct LU ──")
t_direct = @elapsed begin
    theta_opt_dir, trace_dir = optimize_directivity(
        Z_efie, Mp, v, Q_target, Q_total, copy(theta_init);
        maxiter=maxiter_opt, tol=tol_opt,
        alpha0=1e8, verbose=false,
        reactive=true,
        lb=-theta_bound, ub=theta_bound,
        solver=:direct,
    )
end
J_dir = trace_dir[end].J
println("  Time: $(round(t_direct, sigdigits=4))s")
println("  Iterations: $(length(trace_dir))")
println("  J_opt = $(round(J_dir*100, digits=2))%")

# ─────────────────────────────────────────────────
# Run 2: GMRES (no preconditioner)
# ─────────────────────────────────────────────────
println("\n── Optimization: GMRES (no preconditioner) ──")
t_gmres_nop = @elapsed begin
    theta_opt_gm_nop, trace_gm_nop = optimize_directivity(
        Z_efie, Mp, v, Q_target, Q_total, copy(theta_init);
        maxiter=maxiter_opt, tol=tol_opt,
        alpha0=1e8, verbose=false,
        reactive=true,
        lb=-theta_bound, ub=theta_bound,
        solver=:gmres, precond_rank=0,
        gmres_tol=1e-8, gmres_maxiter=300,
    )
end
J_gm_nop = trace_gm_nop[end].J
println("  Time: $(round(t_gmres_nop, sigdigits=4))s")
println("  Iterations: $(length(trace_gm_nop))")
println("  J_opt = $(round(J_gm_nop*100, digits=2))%")
println("  Speedup vs direct: $(round(t_direct / t_gmres_nop, sigdigits=3))x")
rel_J_nop = abs(J_gm_nop - J_dir) / max(abs(J_dir), 1e-30)
println("  J_ratio agreement: $rel_J_nop")

df_trace_nop = DataFrame(
    iter  = [t.iter for t in trace_gm_nop],
    J     = [t.J for t in trace_gm_nop],
    gnorm = [t.gnorm for t in trace_gm_nop],
)
CSV.write(joinpath(DATADIR, "precond_inverse_design_trace_gmres_noprecond.csv"), df_trace_nop)

# ─────────────────────────────────────────────────
# Run 3: GMRES + randomized preconditioner
# ─────────────────────────────────────────────────
for k_rank in [10, 20]
    println("\n── Optimization: GMRES + preconditioner rank=$k_rank ──")
    t_gmres = @elapsed begin
        theta_opt_gm, trace_gm = optimize_directivity(
            Z_efie, Mp, v, Q_target, Q_total, copy(theta_init);
            maxiter=maxiter_opt, tol=tol_opt,
            alpha0=1e8, verbose=false,
            reactive=true,
            lb=-theta_bound, ub=theta_bound,
            solver=:gmres, precond_rank=k_rank, precond_seed=42,
            gmres_tol=1e-8, gmres_maxiter=300,
        )
    end
    J_gm = trace_gm[end].J
    println("  Time: $(round(t_gmres, sigdigits=4))s")
    println("  Iterations: $(length(trace_gm))")
    println("  J_opt = $(round(J_gm*100, digits=2))%")
    println("  Speedup vs direct: $(round(t_direct / t_gmres, sigdigits=3))x")

    # Solution agreement
    rel_J_diff = abs(J_gm - J_dir) / max(abs(J_dir), 1e-30)
    rel_theta_diff = norm(theta_opt_gm - theta_opt_dir) / max(norm(theta_opt_dir), 1e-30)
    println("  J_ratio agreement: $rel_J_diff")
    println("  θ_opt agreement: $rel_theta_diff")

    # Save trace
    df_trace_gm = DataFrame(
        iter  = [t.iter for t in trace_gm],
        J     = [t.J for t in trace_gm],
        gnorm = [t.gnorm for t in trace_gm],
    )
    CSV.write(joinpath(DATADIR, "precond_inverse_design_trace_gmres_k$(k_rank).csv"), df_trace_gm)
end

# Save direct trace
df_trace_dir = DataFrame(
    iter  = [t.iter for t in trace_dir],
    J     = [t.J for t in trace_dir],
    gnorm = [t.gnorm for t in trace_dir],
)
CSV.write(joinpath(DATADIR, "precond_inverse_design_trace_direct.csv"), df_trace_dir)

# Post-optimization far-field comparison
Z_opt_dir = assemble_full_Z(Z_efie, Mp, theta_opt_dir; reactive=true)
I_opt_dir = Z_opt_dir \ v
E_ff_dir = compute_farfield(G_mat, I_opt_dir, NΩ)
ff_power_dir = [real(dot(E_ff_dir[:, q], E_ff_dir[:, q])) for q in 1:NΩ]
P_sphere_dir = sum(ff_power_dir[q] * grid.w[q] for q in 1:NΩ)
D_dir = [4π * ff_power_dir[q] / P_sphere_dir for q in 1:NΩ]

# Phi=0 cut
dphi = 2π / 72
phi0_idx = [q for q in 1:NΩ if min(grid.phi[q], 2π - grid.phi[q]) <= dphi/2 + 1e-10]
perm = sortperm(grid.theta[phi0_idx])
phi0_sorted = phi0_idx[perm]

df_cut = DataFrame(
    theta_deg   = rad2deg.(grid.theta[phi0_sorted]),
    D_direct_dBi = 10 .* log10.(max.(D_dir[phi0_sorted], 1e-30)),
)
CSV.write(joinpath(DATADIR, "precond_inverse_design_farfield_cut.csv"), df_cut)

# Summary
println("\n" * "="^60)
println("INVERSE DESIGN COMPARISON COMPLETE")
println("="^60)
println("  Direct LU:  J=$(round(J_dir*100,digits=2))%  time=$(round(t_direct,sigdigits=3))s")
println("  Results saved to $DATADIR/precond_inverse_design_*.csv")
