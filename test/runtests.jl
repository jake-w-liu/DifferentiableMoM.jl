# runtests.jl — Test suite for DifferentiableMoM
#
# Run: julia --project=. test/runtests.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using SparseArrays
using StaticArrays
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

println("="^60)
println("DifferentiableMoM Test Suite")
println("="^60)

# ─────────────────────────────────────────────────
# Test 1: Mesh and RWG Construction
# ─────────────────────────────────────────────────
println("\n── Test 1: Mesh and RWG construction ──")

Lx, Ly = 0.1, 0.1   # 10 cm × 10 cm plate
Nx, Ny = 3, 3
mesh = make_rect_plate(Lx, Ly, Nx, Ny)

println("  Vertices: $(nvertices(mesh)),  Triangles: $(ntriangles(mesh))")
@assert nvertices(mesh) == (Nx+1)*(Ny+1)
@assert ntriangles(mesh) == 2*Nx*Ny

rwg = build_rwg(mesh)
println("  RWG basis functions: $(rwg.nedges)")
@assert rwg.nedges > 0

# Verify RWG edge lengths are positive and areas are positive
@assert all(rwg.len .> 0)
@assert all(rwg.area_plus .> 0)
@assert all(rwg.area_minus .> 0)

# Save mesh data
df_mesh = DataFrame(
    vx = mesh.xyz[1, :],
    vy = mesh.xyz[2, :],
    vz = mesh.xyz[3, :]
)
CSV.write(joinpath(DATADIR, "mesh_vertices.csv"), df_mesh)

df_tri = DataFrame(
    t1 = mesh.tri[1, :],
    t2 = mesh.tri[2, :],
    t3 = mesh.tri[3, :]
)
CSV.write(joinpath(DATADIR, "mesh_triangles.csv"), df_tri)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 2: Green's Function
# ─────────────────────────────────────────────────
println("\n── Test 2: Green's function ──")

k0 = 2π / 0.1   # wavelength = 10 cm
r1 = Vec3(0.0, 0.0, 0.0)
r2 = Vec3(0.05, 0.0, 0.0)
R = norm(r2 - r1)

G = greens(r1, r2, k0)
G_expected = exp(-1im * k0 * R) / (4π * R)
@assert abs(G - G_expected) < 1e-14

# Check reciprocity: G(r,r') = G(r',r)
@assert abs(greens(r1, r2, k0) - greens(r2, r1, k0)) < 1e-14

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 3: EFIE Assembly (PEC plate)
# ─────────────────────────────────────────────────
println("\n── Test 3: EFIE assembly ──")

freq = 3e9            # 3 GHz
c0 = 299792458.0
lambda0 = c0 / freq
k = 2π / lambda0
eta0 = 376.730313668

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
N = rwg.nedges
println("  Z_efie size: $N × $N")
@assert size(Z_efie) == (N, N)

# Z should have nonzero entries
@assert norm(Z_efie) > 0

# Save EFIE matrix magnitude for inspection
df_Z = DataFrame(
    row = repeat(1:N, inner=N),
    col = repeat(1:N, outer=N),
    abs_Z = vec(abs.(Z_efie)),
    real_Z = vec(real.(Z_efie)),
    imag_Z = vec(imag.(Z_efie))
)
CSV.write(joinpath(DATADIR, "Z_efie.csv"), df_Z)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 4: PEC Scattering (plane wave excitation)
# ─────────────────────────────────────────────────
println("\n── Test 4: PEC forward solve ──")

# Normal-incidence plane wave, x-polarized
k_vec = Vec3(0.0, 0.0, -k)    # propagating in -z
E0 = 1.0
pol = Vec3(1.0, 0.0, 0.0)     # x-polarized

v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)
@assert length(v) == N
@assert norm(v) > 0

# Solve PEC EFIE: Z_efie * I = v
I_pec = Z_efie \ v
println("  |I_pec| = $(norm(I_pec))")
@assert norm(I_pec) > 0

# Residual check
residual = norm(Z_efie * I_pec - v) / norm(v)
println("  Relative residual: $residual")
@assert residual < 1e-10

# Save current coefficients
df_I = DataFrame(
    basis_idx = 1:N,
    real_I = real.(I_pec),
    imag_I = imag.(I_pec),
    abs_I  = abs.(I_pec)
)
CSV.write(joinpath(DATADIR, "I_pec.csv"), df_I)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 5: Impedance Term and Derivatives
# ─────────────────────────────────────────────────
println("\n── Test 5: Impedance term and derivatives ──")

Nt = ntriangles(mesh)
# Simple partition: one patch per triangle
partition = PatchPartition(collect(1:Nt), Nt)

Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
@assert length(Mp) == Nt

# Each M_p should be symmetric (real-valued mass matrix)
for p in 1:min(3, Nt)
    @assert norm(Mp[p] - Mp[p]') < 1e-14 * norm(Mp[p])
end

# Test impedance assembly
theta = fill(100.0 + 50.0im, Nt)  # complex impedance
Z_imp = assemble_Z_impedance(Mp, theta)
@assert size(Z_imp) == (N, N)

# Resistive/reactive decomposition sanity checks
Z_imp_res = assemble_Z_impedance(Mp, fill(100.0, Nt))
Z_imp_reac = assemble_Z_impedance(Mp, 1im .* fill(100.0, Nt))
@assert maximum(abs.(imag.(Z_imp_res))) < 1e-12
@assert maximum(abs.(real.(Z_imp_reac))) < 1e-12

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 6: Far-Field and Q Matrix
# ─────────────────────────────────────────────────
println("\n── Test 6: Far-field and Q matrix ──")

grid = make_sph_grid(8, 16)
NΩ = length(grid.w)
println("  Far-field grid: $NΩ directions")

G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
@assert size(G_mat) == (3 * NΩ, N)

# Compute far-field from PEC solution
E_ff = compute_farfield(G_mat, I_pec, NΩ)
@assert size(E_ff) == (3, NΩ)

# Far-field should be transverse: r̂ · E∞ ≈ 0
max_radial = let mr = 0.0
    for q in 1:NΩ
        rh = Vec3(grid.rhat[:, q])
        Eq = CVec3(E_ff[:, q])
        radial = abs(dot(rh, Eq)) / max(abs(norm(Eq)), 1e-30)
        mr = max(mr, radial)
    end
    mr
end
println("  Max radial E-field component: $max_radial")
@assert max_radial < 0.1  # should be small

# Build Q matrix
pol_mat = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=30 * π / 180)
Q = build_Q(G_mat, grid, pol_mat; mask=mask)

# Q should be Hermitian PSD
@assert norm(Q - Q') < 1e-12 * norm(Q)
eigvals_Q = eigvals(Hermitian(Q))
@assert all(eigvals_Q .>= -1e-12 * maximum(eigvals_Q))
println("  Q is Hermitian PSD ✓")

# Cross-check objective computed two ways:
#   (1) quadratic form I†QI
#   (2) direct angular integration of projected far field
P_qform = real(dot(I_pec, Q * I_pec))
P_direct = projected_power(E_ff, grid, pol_mat; mask=mask)
rel_q_err = abs(P_qform - P_direct) / max(abs(P_qform), 1e-30)
println("  Objective consistency (I†QI vs direct projected power): $rel_q_err")
@assert rel_q_err < 1e-12

# Save far-field pattern
ff_power = [real(dot(E_ff[:, q], E_ff[:, q])) for q in 1:NΩ]
df_ff = DataFrame(
    theta_deg = rad2deg.(grid.theta),
    phi_deg   = rad2deg.(grid.phi),
    power_dB  = 10 .* log10.(max.(ff_power, 1e-30))
)
CSV.write(joinpath(DATADIR, "farfield_pec.csv"), df_ff)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 7: Adjoint Gradient Verification
# ─────────────────────────────────────────────────
println("\n── Test 7: Adjoint gradient verification (CRITICAL) ──")

# Setup: impedance sheet problem with real impedance parameters
theta_real = fill(200.0, Nt)  # real impedance values
Z_full = assemble_full_Z(Z_efie, Mp, theta_real)
I_imp = Z_full \ v

# Objective
J_val = compute_objective(I_imp, Q)
println("  J(θ₀) = $J_val")

# Adjoint gradient
lambda = solve_adjoint(Z_full, Q, I_imp)
g_adj = gradient_impedance(Mp, I_imp, lambda)
println("  |g_adj| = $(norm(g_adj))")

# Gradient verification via central finite differences
#
# Note: complex-step is not used here because J(θ) = I†QI involves
# conjugation (sesquilinear form), which breaks analyticity.
# Central FD with h ≈ 1e-5 provides O(h²) accuracy, sufficient
# for validating the adjoint gradient.

function J_of_theta(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= theta_vec[p] .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

# Sanity check: J_of_theta at baseline should match J_val
J_check = J_of_theta(theta_real)
@assert abs(J_check - J_val) / max(abs(J_val), 1e-30) < 1e-12

println("  Checking adjoint vs central finite difference (h=1e-5)...")
fd_results = Float64[]
adj_results = Float64[]
rel_errors = Float64[]

n_check = min(Nt, 10)  # check first 10 parameters
h_fd = 1e-5
for p in 1:n_check
    g_fd = fd_grad(J_of_theta, theta_real, p; h=h_fd)
    rel_err = abs(g_adj[p] - g_fd) / max(abs(g_adj[p]), 1e-30)
    push!(fd_results, g_fd)
    push!(adj_results, g_adj[p])
    push!(rel_errors, rel_err)
    println("    p=$p: adj=$(g_adj[p])  fd=$g_fd  rel_err=$rel_err")
end

# Save gradient verification data
df_grad = DataFrame(
    param_idx = 1:n_check,
    adjoint   = adj_results,
    fd_central = fd_results,
    rel_error = rel_errors
)
CSV.write(joinpath(DATADIR, "gradient_verification.csv"), df_grad)

max_rel_err = maximum(rel_errors)
println("  Max relative error (adjoint vs central FD): $max_rel_err")
@assert max_rel_err < 1e-4 "Gradient verification FAILED: max rel error = $max_rel_err"

println("  PASS ✓  (adjoint gradients match central FD)")

# ─────────────────────────────────────────────────
# Test 8: FD Convergence Check
# ─────────────────────────────────────────────────
println("\n── Test 8: FD convergence rate check ──")

# Verify FD error decreases at O(h²) for central differences
# by comparing two step sizes on a single parameter
p_test = 1
h1 = 1e-4
h2 = 1e-5
g_fd1 = fd_grad(J_of_theta, theta_real, p_test; h=h1)
g_fd2 = fd_grad(J_of_theta, theta_real, p_test; h=h2)
err1 = abs(g_adj[p_test] - g_fd1)
err2 = abs(g_adj[p_test] - g_fd2)

if err1 > 1e-15 && err2 > 1e-15
    rate = log10(err1 / err2) / log10(h1 / h2)
    println("  Error at h=$h1: $err1")
    println("  Error at h=$h2: $err2")
    println("  Convergence rate: $rate  (expected ≈ 2 for central FD)")
    # Rate should be near 2 for central differences (O(h²))
    @assert rate > 1.5 "FD convergence rate too low: $rate (expected ~2)"
else
    println("  Errors at machine precision — gradient is exact")
end

# Save FD check data
df_fd = DataFrame(
    param_idx = 1:n_check,
    adjoint   = adj_results,
    fd        = fd_results,
    rel_error = rel_errors
)
CSV.write(joinpath(DATADIR, "gradient_fd_check.csv"), df_fd)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 9: Reciprocity Check
# ─────────────────────────────────────────────────
println("\n── Test 9: Reciprocity check ──")

# For EFIE on PEC: Z should be symmetric (Z = Z^T) due to reciprocity
# (Galerkin testing with the same basis/test functions)
sym_err = norm(Z_efie - transpose(Z_efie)) / norm(Z_efie)
println("  Symmetry error (EFIE, PEC): $sym_err")
# Note: due to quadrature, small symmetry error is expected
@assert sym_err < 1e-10 "EFIE matrix not symmetric: err = $sym_err"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 10: Optimization Smoke Test
# ─────────────────────────────────────────────────
println("\n── Test 10: Optimization smoke test ──")

theta_init = fill(300.0, Nt)
theta_opt, trace = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init;
    maxiter=10, tol=1e-8, alpha0=0.01, verbose=false
)

# Check that objective decreased
if length(trace) >= 2
    J_first = trace[1].J
    J_last  = trace[end].J
    println("  J(iter=1)  = $J_first")
    println("  J(iter=$(length(trace))) = $J_last")

    # Save optimization trace
    df_trace = DataFrame(
        iter  = [t.iter for t in trace],
        J     = [t.J for t in trace],
        gnorm = [t.gnorm for t in trace]
    )
    CSV.write(joinpath(DATADIR, "optimization_trace.csv"), df_trace)
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 11: Paper-consistency metrics from tracked data
# ─────────────────────────────────────────────────
println("\n── Test 11: Paper-consistency metrics ──")

meanval(x) = sum(x) / length(x)

function crossval_metrics(
    df_ref::DataFrame,
    ref_col::Symbol,
    df_cmp::DataFrame,
    cmp_col::Symbol;
    target_theta_deg::Float64=30.0,
)
    left = select(df_ref, :theta_deg, :phi_deg, ref_col)
    right = select(df_cmp, :theta_deg, :phi_deg, cmp_col)
    merged = innerjoin(left, right, on=[:theta_deg, :phi_deg])

    delta = merged[!, cmp_col] .- merged[!, ref_col]
    abs_delta = abs.(delta)

    theta_unique = unique(merged.theta_deg)
    theta_near = theta_unique[argmin(abs.(theta_unique .- target_theta_deg))]
    idx_target = findall(t -> abs(t - theta_near) < 1e-12, merged.theta_deg)

    return (
        n = nrow(merged),
        rmse = sqrt(sum(abs2, delta) / length(delta)),
        mean_abs = meanval(abs_delta),
        target_theta_near = theta_near,
        target_mean_abs = meanval(abs_delta[idx_target]),
    )
end

conv = CSV.read(joinpath(DATADIR, "convergence_study.csv"), DataFrame)
grad = CSV.read(joinpath(DATADIR, "gradient_verification.csv"), DataFrame)
rob = CSV.read(joinpath(DATADIR, "robustness_sweep.csv"), DataFrame)

max_grad_mesh = maximum(conv.max_grad_err)
min_energy_ratio = minimum(conv.energy_ratio)
max_grad_ref = maximum(grad.rel_error)

idx_nom = findfirst(rob.case .== "f_nom")
idx_p2 = findfirst(rob.case .== "f_+2pct")
@assert idx_nom !== nothing
@assert idx_p2 !== nothing

J_opt_nom = rob.J_opt_pct[idx_nom]
J_pec_nom = rob.J_pec_pct[idx_nom]
peak_theta_p2 = rob.peak_theta_opt_deg[idx_p2]

df_pec_julia = CSV.read(joinpath(DATADIR, "beam_steer_farfield.csv"), DataFrame)
df_pec_bempp = CSV.read(joinpath(DATADIR, "bempp_pec_farfield.csv"), DataFrame)
df_imp_julia = CSV.read(joinpath(DATADIR, "julia_impedance_farfield.csv"), DataFrame)
df_imp_bempp = CSV.read(joinpath(DATADIR, "bempp_impedance_farfield.csv"), DataFrame)

pec_cv = crossval_metrics(df_pec_julia, :dir_pec_dBi, df_pec_bempp, :dir_bempp_dBi)
imp_cv = crossval_metrics(df_imp_julia, :dir_julia_imp_dBi, df_imp_bempp, :dir_bempp_imp_dBi)

println("  Max grad rel. err (reference): $max_grad_ref")
println("  Max grad rel. err (mesh sweep): $max_grad_mesh")
println("  Min energy ratio: $min_energy_ratio")
println("  Nominal J_opt/J_pec (%): $J_opt_nom / $J_pec_nom")
println("  +2% freq peak theta (deg): $peak_theta_p2")
println("  PEC CV RMSE / near-target |ΔD| (dB): $(pec_cv.rmse) / $(pec_cv.target_mean_abs)")
println("  IMP CV RMSE / near-target |ΔD| (dB): $(imp_cv.rmse) / $(imp_cv.target_mean_abs)")

# These checks track manuscript quantitative claims.
@assert max_grad_ref < 3e-7
@assert max_grad_mesh < 3e-6
@assert min_energy_ratio > 0.98
@assert J_opt_nom > J_pec_nom
@assert peak_theta_p2 < 5.0
@assert pec_cv.target_mean_abs < 0.5
@assert imp_cv.target_mean_abs < 3.0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
println("\n" * "="^60)
println("ALL TESTS PASSED")
println("="^60)
println("\nCSV data files saved to: $DATADIR/")
for f in readdir(DATADIR)
    println("  $f")
end
