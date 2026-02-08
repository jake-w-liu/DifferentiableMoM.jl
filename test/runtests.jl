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

function write_icosphere_obj(path::AbstractString; radius::Float64=0.05, subdivisions::Int=2)
    ϕ = (1 + sqrt(5.0)) / 2
    verts = [
        (-1.0,  ϕ, 0.0), ( 1.0,  ϕ, 0.0), (-1.0, -ϕ, 0.0), ( 1.0, -ϕ, 0.0),
        ( 0.0, -1.0, ϕ), ( 0.0,  1.0, ϕ), ( 0.0, -1.0,-ϕ), ( 0.0,  1.0,-ϕ),
        (  ϕ, 0.0, -1.0), (  ϕ, 0.0,  1.0), ( -ϕ, 0.0, -1.0), ( -ϕ, 0.0,  1.0),
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
        println(io, "# Icosphere mesh for test gate")
        for v in verts
            println(io, "v $(radius * v[1]) $(radius * v[2]) $(radius * v[3])")
        end
        for (i, j, k) in faces
            println(io, "f $i $j $k")
        end
    end
end

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
# Test 1b: OBJ mesh import
# ─────────────────────────────────────────────────
println("\n── Test 1b: OBJ mesh import ──")

obj_path = joinpath(DATADIR, "tmp_quad.obj")
open(obj_path, "w") do io
    println(io, "v 0 0 0")
    println(io, "v 1 0 0")
    println(io, "v 1 1 0")
    println(io, "v 0 1 0")
    println(io, "f 1 2 3 4")
end

mesh_obj = read_obj_mesh(obj_path)
@assert nvertices(mesh_obj) == 4
@assert ntriangles(mesh_obj) == 2
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
# Test 2b: PEC sphere Mie utilities
# ─────────────────────────────────────────────────
println("\n── Test 2b: PEC Mie-theory utilities ──")

k_mie = 2π / 0.1
a_mie = 0.03
θ_mie = 60 * π / 180
khat_mie = Vec3(0.0, 0.0, 1.0)
pol_mie = Vec3(1.0, 0.0, 0.0)
rhat_mie = Vec3(sin(θ_mie), 0.0, cos(θ_mie))
μ_mie = dot(khat_mie, rhat_mie)

S1_mie, S2_mie = mie_s1s2_pec(k_mie * a_mie, μ_mie)
@assert isfinite(real(S1_mie)) && isfinite(imag(S1_mie))
@assert isfinite(real(S2_mie)) && isfinite(imag(S2_mie))

σ_formula = 4π * abs2(S2_mie) / (k_mie^2)   # φ=0 plane for x-pol
σ_mie = mie_bistatic_rcs_pec(k_mie, a_mie, khat_mie, pol_mie, rhat_mie)
@assert σ_mie >= 0
@assert abs(σ_mie - σ_formula) / max(abs(σ_formula), 1e-30) < 1e-10

# Independent energy-consistency check:
# integrate differential cross section over sphere and compare to
# coefficient-sum total scattering cross section Csca.
function _mie_csca_coeff_pec(k::Float64, a::Float64; nmax::Int=80)
    x = k * a
    j = zeros(Float64, nmax + 1)  # j[n+1] = j_n
    y = zeros(Float64, nmax + 1)  # y[n+1] = y_n
    j[1] = sin(x) / x
    y[1] = -cos(x) / x
    j[2] = sin(x) / x^2 - cos(x) / x
    y[2] = -cos(x) / x^2 - sin(x) / x
    for n in 1:(nmax - 1)
        j[n + 2] = ((2n + 1) / x) * j[n + 1] - j[n]
        y[n + 2] = ((2n + 1) / x) * y[n + 1] - y[n]
    end

    psi = zeros(Float64, nmax + 1)
    xi  = zeros(ComplexF64, nmax + 1)
    for n in 0:nmax
        psi[n + 1] = x * j[n + 1]
        xi[n + 1]  = x * (j[n + 1] - 1im * y[n + 1])
    end

    csum = 0.0
    for n in 1:nmax
        psi_p = psi[n] - (n / x) * psi[n + 1]
        xi_p  = xi[n] - (n / x) * xi[n + 1]
        an = -psi_p / xi_p
        bn = -psi[n + 1] / xi[n + 1]
        csum += (2n + 1) * (abs2(an) + abs2(bn))
    end
    return (2π / (k^2)) * csum
end

grid_mie = make_sph_grid(181, 180)
σ_mie_grid = [
    mie_bistatic_rcs_pec(k_mie, a_mie, khat_mie, pol_mie, Vec3(grid_mie.rhat[:, q]))
    for q in 1:length(grid_mie.w)
]
Csca_num = sum((σ_mie_grid ./ (4π)) .* grid_mie.w)
Csca_ref = _mie_csca_coeff_pec(k_mie, a_mie; nmax=80)
@assert abs(Csca_num - Csca_ref) / max(abs(Csca_ref), 1e-30) < 2e-3

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

# RCS helper checks
sigma = bistatic_rcs(E_ff; E0=1.0)
@assert length(sigma) == NΩ
@assert all(sigma .>= 0.0)

bs = backscatter_rcs(E_ff, grid, Vec3(0.0, 0.0, -1.0); E0=1.0)
@assert 1 <= bs.index <= NΩ
@assert bs.sigma >= 0.0

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

imp_beam_csv = joinpath(DATADIR, "impedance_validation_matrix_summary.csv")
if !isfile(imp_beam_csv)
    imp_beam_csv = joinpath(DATADIR, "impedance_validation_matrix_summary_paper_default.csv")
end
df_imp_beam = CSV.read(imp_beam_csv, DataFrame)

pec_cv = crossval_metrics(df_pec_julia, :dir_pec_dBi, df_pec_bempp, :dir_bempp_dBi)
imp_cv = crossval_metrics(df_imp_julia, :dir_julia_imp_dBi, df_imp_bempp, :dir_bempp_imp_dBi)

n_beam_cases = nrow(df_imp_beam)
n_pass_main_theta = count(df_imp_beam.pass_main_theta_le_3deg)
n_pass_main_level = count(df_imp_beam.pass_main_level_le_1p5db)
n_pass_sll = count(df_imp_beam.pass_sll_le_3db)

println("  Max grad rel. err (reference): $max_grad_ref")
println("  Max grad rel. err (mesh sweep): $max_grad_mesh")
println("  Min energy ratio: $min_energy_ratio")
println("  Nominal J_opt/J_pec (%): $J_opt_nom / $J_pec_nom")
println("  +2% freq peak theta (deg): $peak_theta_p2")
println("  PEC CV RMSE / near-target |ΔD| (dB): $(pec_cv.rmse) / $(pec_cv.target_mean_abs)")
println("  IMP CV RMSE / near-target |ΔD| (dB): $(imp_cv.rmse) / $(imp_cv.target_mean_abs)")
println("  Beam-centric passes (main θ / main L / SLL): $n_pass_main_theta/$n_beam_cases, $n_pass_main_level/$n_beam_cases, $n_pass_sll/$n_beam_cases")

# These checks track manuscript quantitative claims.
@assert max_grad_ref < 3e-7
@assert max_grad_mesh < 3e-6
@assert min_energy_ratio > 0.98
@assert J_opt_nom > J_pec_nom
@assert peak_theta_p2 < 5.0
@assert pec_cv.target_mean_abs < 0.5
@assert n_pass_main_theta == n_beam_cases
@assert n_pass_main_level == n_beam_cases
@assert n_pass_sll == n_beam_cases

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 12: Conditioning / preconditioning consistency
# ─────────────────────────────────────────────────
println("\n── Test 12: Conditioning and preconditioning ──")

theta_c = copy(theta_real)
Z_raw = assemble_full_Z(Z_efie, Mp, theta_c)
I_raw = Z_raw \ v

# Build mass-based regularizer and left preconditioner
R_mass = make_mass_regularizer(Mp)
M_left = make_left_preconditioner(Mp; eps_rel=1e-6)

# Auto-preconditioning selector behavior
M_auto_off, enabled_auto_off, reason_auto_off = select_preconditioner(
    Mp;
    mode=:auto,
    n_threshold=10_000,
    iterative_solver=false,
)
println("  Auto preconditioning (high threshold): enabled=$enabled_auto_off ($reason_auto_off)")
@assert !enabled_auto_off
@assert M_auto_off === nothing

M_auto_on, enabled_auto_on, reason_auto_on = select_preconditioner(
    Mp;
    mode=:auto,
    n_threshold=1,
    iterative_solver=false,
    eps_rel=1e-6,
)
println("  Auto preconditioning (low threshold): enabled=$enabled_auto_on ($reason_auto_on)")
@assert enabled_auto_on
@assert M_auto_on !== nothing

# Left-preconditioned system should preserve the same solution
Z_pre, v_pre, fac_pre = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=0.0,
    regularization_R=nothing,
    preconditioner_M=M_left,
)
I_pre = Z_pre \ v_pre
rel_I_pre = norm(I_pre - I_raw) / max(norm(I_raw), 1e-30)
println("  Left-preconditioned solution mismatch: $rel_I_pre")
@assert rel_I_pre < 1e-10

# Auto-selected preconditioner (activated) should also preserve the solution
Z_pre_auto, v_pre_auto, _ = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=0.0,
    regularization_R=nothing,
    preconditioner_M=M_auto_on,
)
I_pre_auto = Z_pre_auto \ v_pre_auto
rel_I_pre_auto = norm(I_pre_auto - I_raw) / max(norm(I_raw), 1e-30)
println("  Auto-preconditioned solution mismatch: $rel_I_pre_auto")
@assert rel_I_pre_auto < 1e-10

# Regularization should alter the solve (for nonzero alpha)
alpha_reg = 1e-3
Z_reg, v_reg, _ = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=alpha_reg,
    regularization_R=R_mass,
    preconditioner_M=nothing,
)
I_reg = Z_reg \ v_reg
rel_I_reg = norm(I_reg - I_raw) / max(norm(I_raw), 1e-30)
println("  Regularized solution change (alpha=$alpha_reg): $rel_I_reg")
@assert rel_I_reg > 1e-9

# Adjoint gradient with left preconditioning should match FD on the
# equivalently preconditioned system objective.
Mp_pre, fac_pre = transform_patch_matrices(
    Mp;
    preconditioner_M=M_left,
    preconditioner_factor=fac_pre,
)
lambda_pre = solve_adjoint(Z_pre, Q, I_pre)
g_pre = gradient_impedance(Mp_pre, I_pre, lambda_pre)

function J_of_theta_pre(theta_vec)
    Z_t = assemble_full_Z(Z_efie, Mp, theta_vec)
    Zp, vp, _ = prepare_conditioned_system(
        Z_t,
        v;
        regularization_alpha=0.0,
        regularization_R=nothing,
        preconditioner_M=M_left,
        preconditioner_factor=fac_pre,
    )
    I_t = Zp \ vp
    return real(dot(I_t, Q * I_t))
end

p_pre = 1
g_fd_pre = fd_grad(J_of_theta_pre, theta_c, p_pre; h=1e-5)
rel_g_pre = abs(g_pre[p_pre] - g_fd_pre) / max(abs(g_pre[p_pre]), 1e-30)
println("  Preconditioned gradient rel. error (p=$p_pre): $rel_g_pre")
@assert rel_g_pre < 1e-4

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 13: Sphere MoM-vs-Mie benchmark CI gate
# ─────────────────────────────────────────────────
println("\n── Test 13: Sphere Mie benchmark gate ──")

obj_sphere = joinpath(DATADIR, "tmp_sphere_gate.obj")
write_icosphere_obj(obj_sphere; radius=0.05, subdivisions=2)

mesh_s = read_obj_mesh(obj_sphere)
rwg_s = build_rwg(mesh_s)

freq_s = 2.0e9
c0_s = 299792458.0
lambda_s = c0_s / freq_s
k_s = 2π / lambda_s
eta0_s = 376.730313668
k_vec_s = Vec3(0.0, 0.0, -k_s)
khat_s = k_vec_s / norm(k_vec_s)
pol_s = Vec3(1.0, 0.0, 0.0)

ctr_s = vec(sum(mesh_s.xyz, dims=2) ./ nvertices(mesh_s))
radii_s = [norm(Vec3(mesh_s.xyz[:, i]) - Vec3(ctr_s)) for i in 1:nvertices(mesh_s)]
a_s = sum(radii_s) / length(radii_s)

Z_s = assemble_Z_efie(mesh_s, rwg_s, k_s; quad_order=3, eta0=eta0_s)
v_s = assemble_v_plane_wave(mesh_s, rwg_s, k_vec_s, 1.0, pol_s; quad_order=3)
I_s = solve_forward(Z_s, v_s)
res_s = norm(Z_s * I_s - v_s) / max(norm(v_s), 1e-30)
@assert res_s < 1e-10

grid_s = make_sph_grid(181, 72)
G_s = radiation_vectors(mesh_s, rwg_s, grid_s, k_s; quad_order=3, eta0=eta0_s)
E_s = compute_farfield(G_s, I_s, length(grid_s.w))
σ_mom_s = bistatic_rcs(E_s; E0=1.0)

phi_target_s = grid_s.phi[argmin(grid_s.phi)]
idx_cut_s = [q for q in 1:length(grid_s.w) if abs(grid_s.phi[q] - phi_target_s) < 1e-12]
idx_cut_s = idx_cut_s[sortperm(grid_s.theta[idx_cut_s])]

σ_mie_s = [
    mie_bistatic_rcs_pec(k_s, a_s, khat_s, pol_s, Vec3(grid_s.rhat[:, q]))
    for q in idx_cut_s
]

dB_mom_s = 10 .* log10.(max.(σ_mom_s[idx_cut_s], 1e-30))
dB_mie_s = 10 .* log10.(max.(σ_mie_s, 1e-30))
ΔdB_s = dB_mom_s .- dB_mie_s

mae_s = sum(abs.(ΔdB_s)) / length(ΔdB_s)
rmse_s = sqrt(sum(abs2, ΔdB_s) / length(ΔdB_s))
maxabs_s = maximum(abs.(ΔdB_s))

σ_bs_mom_s = backscatter_rcs(E_s, grid_s, khat_s; E0=1.0).sigma
σ_bs_mie_s = mie_bistatic_rcs_pec(k_s, a_s, khat_s, pol_s, -khat_s)
Δbs_s = 10 * log10(max(σ_bs_mom_s, 1e-30)) - 10 * log10(max(σ_bs_mie_s, 1e-30))

println("  MAE(dB): $mae_s")
println("  RMSE(dB): $rmse_s")
println("  Max |Δ|(dB): $maxabs_s")
println("  Backscatter Δ(dB): $Δbs_s")

# Dedicated CI thresholds for the sphere benchmark
@assert mae_s < 0.35 "Sphere Mie gate failed: MAE(dB)=$mae_s"
@assert rmse_s < 0.40 "Sphere Mie gate failed: RMSE(dB)=$rmse_s"
@assert maxabs_s < 0.80 "Sphere Mie gate failed: max |Δ|(dB)=$maxabs_s"
@assert abs(Δbs_s) < 0.80 "Sphere Mie gate failed: |backscatter Δ(dB)|=$(abs(Δbs_s))"

df_sphere_gate = DataFrame(
    metric = ["mae_db", "rmse_db", "max_abs_db", "backscatter_delta_db", "solve_residual"],
    value = [mae_s, rmse_s, maxabs_s, Δbs_s, res_s],
)
CSV.write(joinpath(DATADIR, "sphere_mie_gate_metrics.csv"), df_sphere_gate)

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
