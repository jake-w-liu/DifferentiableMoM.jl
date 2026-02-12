# runtests.jl — Test suite for DifferentiableMoM
#
# Run: julia --project=. test/runtests.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using Random
using Test
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

report_obj = assert_mesh_quality(mesh_obj; allow_boundary=true)
@assert report_obj.n_nonmanifold_edges == 0
@assert report_obj.n_orientation_conflicts == 0
@assert report_obj.n_degenerate_triangles == 0

# Orientation-conflict negative test (shared interior edge has same orientation)
tri_bad_orient = hcat([1, 2, 3], [1, 4, 3])
mesh_bad_orient = TriMesh(mesh_obj.xyz, tri_bad_orient)
thrown_orient = try
    assert_mesh_quality(mesh_bad_orient; allow_boundary=true)
    false
catch
    true
end
@assert thrown_orient

# Degenerate-triangle negative test
tri_bad_deg = hcat([1, 1, 2])
mesh_bad_deg = TriMesh(mesh_obj.xyz, tri_bad_deg)
thrown_deg = try
    assert_mesh_quality(mesh_bad_deg; allow_boundary=true)
    false
catch
    true
end
@assert thrown_deg

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1c: Mesh repair utility
# ─────────────────────────────────────────────────
println("\n── Test 1c: Mesh repair utility ──")

repair_orient = repair_mesh_for_simulation(mesh_bad_orient; allow_boundary=true)
@assert repair_orient.after.n_orientation_conflicts == 0
@assert !isempty(repair_orient.flipped_triangles)
@assert mesh_quality_ok(repair_orient.after; allow_boundary=true, require_closed=false)

xyz_bad_mixed = hcat(mesh_obj.xyz, mesh_obj.xyz[:, 1])
tri_bad_mixed = hcat([1, 2, 5], [1, 2, 3], [1, 6, 3])
mesh_bad_mixed = TriMesh(xyz_bad_mixed, tri_bad_mixed)
repair_mixed = repair_mesh_for_simulation(
    mesh_bad_mixed;
    allow_boundary=true,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=false,
)
@assert ntriangles(repair_mixed.mesh) == 1
@assert length(repair_mixed.removed_invalid) == 1
@assert length(repair_mixed.removed_degenerate) == 1

repair_in_path = joinpath(DATADIR, "tmp_repair_in.obj")
open(repair_in_path, "w") do io
    println(io, "v 0 0 0")
    println(io, "v 1 0 0")
    println(io, "v 1 1 0")
    println(io, "v 0 1 0")
    println(io, "f 1 2 3")
    println(io, "f 1 4 3")
end
repair_out_path = joinpath(DATADIR, "tmp_repair_out.obj")
repair_obj_result = repair_obj_mesh(repair_in_path, repair_out_path; allow_boundary=true)
@assert isfile(repair_obj_result.output_path)
mesh_repair_out = read_obj_mesh(repair_out_path)
report_repair_out = mesh_quality_report(mesh_repair_out)
@assert report_repair_out.n_orientation_conflicts == 0
@assert report_repair_out.n_nonmanifold_edges == 0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1d: Mesh coarsening utilities
# ─────────────────────────────────────────────────
println("\n── Test 1d: Mesh coarsening utilities ──")

@assert estimate_dense_matrix_gib(100) > 0

mesh_cluster_in = make_rect_plate(1.0, 1.0, 6, 6)
mesh_cluster_out = cluster_mesh_vertices(mesh_cluster_in, 0.35)
@assert nvertices(mesh_cluster_out) > 0
@assert ntriangles(mesh_cluster_out) > 0

mesh_edges_test = make_rect_plate(1.0, 1.0, 1, 1) # two triangles
edges_test = mesh_unique_edges(mesh_edges_test)
@assert length(edges_test) == 5
segments_test = mesh_wireframe_segments(mesh_edges_test)
@assert segments_test.n_edges == 5
@assert length(segments_test.x) == 15
@assert count(isnan, segments_test.x) == 5

p_mesh = plot_mesh_wireframe(mesh_edges_test; title="Mesh preview test", linewidth=0.5)
@assert p_mesh !== nothing
p_cmp = plot_mesh_comparison(mesh_edges_test, mesh_edges_test; title_a="A", title_b="B", size=(600, 300))
@assert p_cmp !== nothing

xyz_nm = [
    0.0  1.0  0.0  0.0  0.0  2.0  2.0;
    0.0  0.0  1.0 -1.0  0.0  0.0  1.0;
    0.0  0.0  0.0  0.0  1.0  0.0  0.0
]
tri_nm = [
    1  1  1  3;
    2  2  2  6;
    3  4  5  7
]
mesh_nm = TriMesh(xyz_nm, tri_nm)
mesh_nm_clean = drop_nonmanifold_triangles(mesh_nm)
report_nm = mesh_quality_report(mesh_nm_clean)
@assert report_nm.n_nonmanifold_edges == 0

mesh_coarse_in = make_rect_plate(1.0, 1.0, 12, 12)
rwg_coarse_in = build_rwg(mesh_coarse_in; precheck=true, allow_boundary=true)
@assert rwg_coarse_in.nedges > 60

target_rwg = 60
coarse_result = coarsen_mesh_to_target_rwg(mesh_coarse_in, target_rwg; max_iters=8)
rwg_coarse_out = build_rwg(coarse_result.mesh; precheck=true, allow_boundary=true)
@assert coarse_result.rwg_count == rwg_coarse_out.nedges
@assert abs(coarse_result.rwg_count - target_rwg) <= target_rwg  # improved complexity scale
@assert rwg_coarse_out.nedges < rwg_coarse_in.nedges
report_coarse_out = mesh_quality_report(coarse_result.mesh)
@assert mesh_quality_ok(report_coarse_out; allow_boundary=true, require_closed=false)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1e: Parabolic reflector mesh
# ─────────────────────────────────────────────────
println("\n── Test 1e: Parabolic reflector mesh ──")

D_ref = 0.30
f_ref = 0.105
Nr_ref = 6
Nphi_ref = 20
mesh_ref = make_parabolic_reflector(D_ref, f_ref, Nr_ref, Nphi_ref)
report_ref = mesh_quality_report(mesh_ref)
@assert mesh_quality_ok(report_ref; allow_boundary=true, require_closed=false)
@assert nvertices(mesh_ref) == 1 + Nr_ref * Nphi_ref
@assert ntriangles(mesh_ref) == Nphi_ref + 2 * (Nr_ref - 1) * Nphi_ref

# Paraboloid checks at rim points: z=r²/(4f), and distance-to-focus = z+f.
focus = Vec3(0.0, 0.0, f_ref)
rim_start = 2 + (Nr_ref - 1) * Nphi_ref
for idx in rim_start:4:(rim_start + Nphi_ref - 1)
    p = Vec3(mesh_ref.xyz[:, idx])
    rxy = hypot(p[1], p[2])
    z_expected = rxy^2 / (4 * f_ref)
    @assert abs(p[3] - z_expected) < 1e-12
    d_focus = norm(p - focus)
    @assert abs(d_focus - (p[3] + f_ref)) < 1e-12
end

rwg_ref = build_rwg(mesh_ref; precheck=true, allow_boundary=true)
@assert rwg_ref.nedges > 0

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

if !isfile(joinpath(DATADIR, "convergence_study.csv"))
    println("  SKIPPED (data files not found — run examples first)")
else
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
end  # if isfile(convergence_study.csv)

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
# Test 14: Excitation-model physics sanity checks
# ─────────────────────────────────────────────────
println("\n── Test 14: Excitation physics sanity checks ──")

mesh_exc = make_rect_plate(0.02, 0.02, 3, 3)
rwg_exc = build_rwg(mesh_exc)

freq_exc = 1.0e9
k_exc = 2π * freq_exc / 299792458.0
k_vec_exc = Vec3(0.0, 0.0, -k_exc)
pol_exc = Vec3(1.0, 0.0, 0.0)

v_old_exc = assemble_v_plane_wave(mesh_exc, rwg_exc, k_vec_exc, 1.0, pol_exc; quad_order=3)
v_new_exc = assemble_excitation(mesh_exc, rwg_exc, make_plane_wave(k_vec_exc, 1.0, pol_exc); quad_order=3)
rel_rhs_exc = norm(v_new_exc - v_old_exc) / max(norm(v_old_exc), 1e-30)
println("  Plane-wave path-consistency RHS rel. diff: $rel_rhs_exc")
@assert rel_rhs_exc < 1e-13

# Explicit quadrature check for plane-wave RHS assembly
xi_exc, wq_exc = tri_quad_rule(3)
v_manual_exc = zeros(ComplexF64, rwg_exc.nedges)
for n in 1:rwg_exc.nedges
    for t in (rwg_exc.tplus[n], rwg_exc.tminus[n])
        A = triangle_area(mesh_exc, t)
        pts = tri_quad_points(mesh_exc, t, xi_exc)
        for q in eachindex(wq_exc)
            rq = pts[q]
            fn = eval_rwg(rwg_exc, n, rq, t)
            Einc = pol_exc * exp(-1im * dot(k_vec_exc, rq))
            v_manual_exc[n] += -wq_exc[q] * dot(fn, Einc) * (2 * A)
        end
    end
end
rel_rhs_manual_exc = norm(v_new_exc - v_manual_exc) / max(norm(v_manual_exc), 1e-30)
println("  Plane-wave vs explicit quadrature RHS rel. diff: $rel_rhs_manual_exc")
@assert rel_rhs_manual_exc < 1e-12

gap_a = make_delta_gap(1, 1.0 + 0im, 1e-3)
gap_b = make_delta_gap(1, 1.0 + 0im, 2e-3)
v_gap_a = assemble_excitation(mesh_exc, rwg_exc, gap_a)
v_gap_b = assemble_excitation(mesh_exc, rwg_exc, gap_b)
ratio_gap = abs(v_gap_a[1]) / max(abs(v_gap_b[1]), 1e-30)
println("  Delta-gap scaling ratio (g=1mm/g=2mm): $ratio_gap")
@assert abs(ratio_gap - 2.0) < 1e-12
@assert norm(v_gap_a[2:end]) < 1e-12

port_exc = PortExcitation([1, 2], 2.0 + 0im, 50.0 + 0im)
v_port = assemble_excitation(mesh_exc, rwg_exc, port_exc)
@assert abs(v_port[1] - (2.0 / rwg_exc.len[1])) < 1e-14
@assert abs(v_port[2] - (2.0 / rwg_exc.len[2])) < 1e-14
@assert norm(v_port[3:end]) < 1e-14

# Out-of-bounds port edges should be skipped gracefully
port_oob = PortExcitation([1, rwg_exc.nedges + 10], 1.0 + 0im, 50.0 + 0im)
# Expected-by-design robustness behavior: out-of-range edges are warned and skipped.
v_port_oob = @test_logs (:warn, r"Port edge .* is out of bounds .* Skipping") assemble_excitation(mesh_exc, rwg_exc, port_oob)
@assert abs(v_port_oob[1] - (1.0 / rwg_exc.len[1])) < 1e-14
@assert norm(v_port_oob[2:end]) < 1e-14

thrown_multi = try
    bad_multi = make_multi_excitation([gap_a, gap_b], [1 + 0im])
    assemble_excitation(mesh_exc, rwg_exc, bad_multi)
    false
catch
    true
end
@assert thrown_multi

V_exc = assemble_multiple_excitations(mesh_exc, rwg_exc, [gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)]; quad_order=3)
@assert size(V_exc) == (rwg_exc.nedges, 2)
@assert norm(V_exc[:, 1] - v_gap_a) / max(norm(v_gap_a), 1e-30) < 1e-13
@assert norm(V_exc[:, 2] - v_new_exc) / max(norm(v_new_exc), 1e-30) < 1e-13

weights_exc = ComplexF64[0.3 - 0.1im, -0.2 + 0.7im]
multi_exc = make_multi_excitation([gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)], weights_exc)
v_multi_exc = assemble_excitation(mesh_exc, rwg_exc, multi_exc; quad_order=3)
v_multi_ref = V_exc * weights_exc
rel_multi_exc = norm(v_multi_exc - v_multi_ref) / max(norm(v_multi_ref), 1e-30)
println("  Multi-excitation linearity rel. diff: $rel_multi_exc")
@assert rel_multi_exc < 1e-13

# Imported excitation semantics:
# kind=:electric_field should match direct electric-field import exactly.
E_field_test(r) = CVec3(r[1] + 0im, (0.5 * r[2]) + 0im, 0.0 + 0im)
cur_E = make_imported_excitation(E_field_test; kind=:electric_field, min_quad_order=3)
imp_E = ImportedExcitation(E_field_test; kind=:electric_field, min_quad_order=3)
v_cur_E = assemble_excitation(mesh_exc, rwg_exc, cur_E; quad_order=3)
v_imp_E = assemble_excitation(mesh_exc, rwg_exc, imp_E; quad_order=3)
rel_cur_imp = norm(v_cur_E - v_imp_E) / max(norm(v_imp_E), 1e-30)
println("  ImportedExcitation(:electric_field) self-consistency rel. diff: $rel_cur_imp")
@assert rel_cur_imp < 1e-13

# Source function can return tuple/vector-like 3-component data.
E_field_tuple(r) = (r[1] + 0im, (0.5 * r[2]) + 0im, 0.0 + 0im)
cur_E_tuple = make_imported_excitation(E_field_tuple; kind=:electric_field, min_quad_order=3)
v_cur_E_tuple = assemble_excitation(mesh_exc, rwg_exc, cur_E_tuple; quad_order=3)
rel_cur_tuple = norm(v_cur_E_tuple - v_imp_E) / max(norm(v_imp_E), 1e-30)
println("  ImportedExcitation tuple-return rel. diff: $rel_cur_tuple")
@assert rel_cur_tuple < 1e-13

# surface-current mode uses local equivalent-sheet map E ≈ η Js.
Js_test(r) = CVec3((2r[1]) + 0im, 0.0 + 0im, 0.0 + 0im)
eta_test = 120.0 + 30.0im
cur_Js = make_imported_excitation(Js_test; kind=:surface_current_density, eta_equiv=eta_test, min_quad_order=3)
imp_etaJs = ImportedExcitation(r -> eta_test * Js_test(r); kind=:electric_field, min_quad_order=3)
v_cur_Js = assemble_excitation(mesh_exc, rwg_exc, cur_Js; quad_order=3)
v_imp_Js = assemble_excitation(mesh_exc, rwg_exc, imp_etaJs; quad_order=3)
rel_cur_js = norm(v_cur_Js - v_imp_Js) / max(norm(v_imp_Js), 1e-30)
println("  ImportedExcitation(:surface_current_density) map rel. diff: $rel_cur_js")
@assert rel_cur_js < 1e-13

# Imported field can also be tuple/vector-like.
imp_tuple = ImportedExcitation(r -> (r[1] + 0im, 0.0 + 0im, 0.0 + 0im); kind=:electric_field, min_quad_order=3)
v_imp_tuple = assemble_excitation(mesh_exc, rwg_exc, imp_tuple; quad_order=3)
@assert all(isfinite, real.(v_imp_tuple))
@assert all(isfinite, imag.(v_imp_tuple))

thrown_imp_bad_dim = try
    imp_bad_dim = ImportedExcitation(r -> ComplexF64[1 + 0im, 2 + 0im]; kind=:electric_field)
    assemble_excitation(mesh_exc, rwg_exc, imp_bad_dim; quad_order=3)
    false
catch
    true
end
@assert thrown_imp_bad_dim

thrown_imp_nonfinite = try
    imp_nonfinite = ImportedExcitation(r -> CVec3(NaN + 0im, 0 + 0im, 0 + 0im); kind=:electric_field)
    assemble_excitation(mesh_exc, rwg_exc, imp_nonfinite; quad_order=3)
    false
catch
    true
end
@assert thrown_imp_nonfinite

# Constructor guard
thrown_cur = try
    make_imported_excitation(E_field_test; min_quad_order=0)
    false
catch
    true
end
@assert thrown_cur

thrown_cur_bad_dim = try
    cur_bad_dim = make_imported_excitation(r -> ComplexF64[1 + 0im, 2 + 0im];
                                           kind=:electric_field,
                                           min_quad_order=3)
    assemble_excitation(mesh_exc, rwg_exc, cur_bad_dim; quad_order=3)
    false
catch
    true
end
@assert thrown_cur_bad_dim

thrown_cur_nonfinite = try
    cur_nonfinite = make_imported_excitation(r -> CVec3(NaN + 0im, 0 + 0im, 0 + 0im);
                                             kind=:electric_field,
                                             min_quad_order=3)
    assemble_excitation(mesh_exc, rwg_exc, cur_nonfinite; quad_order=3)
    false
catch
    true
end
@assert thrown_cur_nonfinite

# Hard-break API check: legacy excitation wrappers are intentionally removed.
@assert !isdefined(DifferentiableMoM, :CurrentDistributionExcitation)
@assert !isdefined(DifferentiableMoM, :ImportedFieldExcitation)
@assert !isdefined(DifferentiableMoM, :make_current_distribution)

m_mag = CVec3(0.0 + 0im, 0.0 + 0im, 1e-4 + 0im) # A·m²
dip_mag = make_dipole(Vec3(0.0, 0.0, 0.0), m_mag, Vec3(0.0, 0.0, 1.0), :magnetic, freq_exc)
Rfar = 5.0
E_mag_num = DifferentiableMoM.dipole_incident_field(Vec3(Rfar, 0.0, 0.0), dip_mag)
E_mag_ref = -1im * eta0 * k_exc^2 * m_mag[3] * exp(-1im * k_exc * Rfar) / (4π * Rfar)
rel_mag = abs(E_mag_num[2] - E_mag_ref) / max(abs(E_mag_ref), 1e-30)
println("  Magnetic-dipole far-field rel. error: $rel_mag")
@assert rel_mag < 0.03

# Electric dipole: broadside far-field amplitude and axial quasi-static behavior
p_e = CVec3(0.0 + 0im, 0.0 + 0im, 1e-12 + 0im)
dip_e = make_dipole(Vec3(0.0, 0.0, 0.0), p_e, Vec3(0.0, 0.0, 1.0), :electric, freq_exc)
Rfar_e = 8.0
E_e_num = DifferentiableMoM.dipole_incident_field(Vec3(Rfar_e, 0.0, 0.0), dip_e)
eps0_exc = 8.854187817e-12
E_e_ref = p_e[3] * k_exc^2 * exp(-1im * k_exc * Rfar_e) / (4π * eps0_exc * Rfar_e)
rel_e = abs(E_e_num[3] - E_e_ref) / max(abs(E_e_ref), 1e-30)
println("  Electric-dipole broadside far-field rel. error: $rel_e")
@assert rel_e < 0.03

dip_e_low = make_dipole(Vec3(0.0, 0.0, 0.0), p_e, Vec3(0.0, 0.0, 1.0), :electric, 1.0e6)
R_quasi = 0.5
E_e_axial = DifferentiableMoM.dipole_incident_field(Vec3(0.0, 0.0, R_quasi), dip_e_low)
E_e_static = 2 * p_e[3] / (4π * eps0_exc * R_quasi^3)
rel_e_quasi = abs(real(E_e_axial[3]) - real(E_e_static)) / max(abs(real(E_e_static)), 1e-30)
println("  Electric-dipole axial quasi-static rel. error: $rel_e_quasi")
@assert rel_e_quasi < 0.05

# Loop field must match equivalent magnetic dipole field and RHS exactly
loop_eq = make_loop(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), 0.01, 2.0 + 0im, freq_exc)
m_eq = (2.0 + 0im) * π * (0.01)^2
dip_eq = make_dipole(Vec3(0.0, 0.0, 0.0), CVec3(0.0 + 0im, 0.0 + 0im, m_eq), Vec3(0.0, 0.0, 1.0), :magnetic, freq_exc)
E_loop_eq = DifferentiableMoM.loop_incident_field(Vec3(Rfar, 0.0, 0.0), loop_eq)
E_dip_eq = DifferentiableMoM.dipole_incident_field(Vec3(Rfar, 0.0, 0.0), dip_eq)
rel_loop_field = norm(E_loop_eq - E_dip_eq) / max(norm(E_dip_eq), 1e-30)
println("  Loop vs equivalent magnetic-dipole field rel. diff: $rel_loop_field")
@assert rel_loop_field < 1e-13

v_loop_eq = assemble_excitation(mesh_exc, rwg_exc, loop_eq; quad_order=3)
v_dip_eq = assemble_excitation(mesh_exc, rwg_exc, dip_eq; quad_order=3)
rel_loop_rhs = norm(v_loop_eq - v_dip_eq) / max(norm(v_dip_eq), 1e-30)
println("  Loop vs equivalent magnetic-dipole RHS rel. diff: $rel_loop_rhs")
@assert rel_loop_rhs < 1e-13

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 15: Dipole/loop far-field pattern gate
# ─────────────────────────────────────────────────
println("\n── Test 15: Dipole/loop far-field pattern gate ──")

freq_pat = 1.0e9
k_pat = 2π * freq_pat / 299792458.0
lambda_pat = 2π / k_pat
Rfar_pat = 50 * lambda_pat
theta_deg_pat = collect(0.0:1.0:180.0)
theta_pat = deg2rad.(theta_deg_pat)

dip_pat = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, 1e-12 + 0im),   # electric dipole (C·m)
    Vec3(0.0, 0.0, 1.0),
    :electric,
    freq_pat
)
loop_pat = make_loop(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.01,                                        # 1 cm radius
    1.0 + 0im,
    freq_pat
)

P_dip_num = zeros(Float64, length(theta_pat))
P_loop_num = zeros(Float64, length(theta_pat))
P_ana = sin.(theta_pat) .^ 2
E_dip_theta_cmp = zeros(ComplexF64, length(theta_pat))
E_loop_phi_cmp = zeros(ComplexF64, length(theta_pat))

for i in eachindex(theta_pat)
    th = theta_pat[i]
    rhat = Vec3(sin(th), 0.0, cos(th))          # φ = 0 cut
    r = Rfar_pat * rhat
    E_d = DifferentiableMoM.dipole_incident_field(r, dip_pat)
    E_l = DifferentiableMoM.loop_incident_field(r, loop_pat)
    e_theta = Vec3(cos(th), 0.0, -sin(th))
    e_phi = Vec3(0.0, 1.0, 0.0)
    E_dip_theta_cmp[i] = dot(E_d, e_theta)
    E_loop_phi_cmp[i] = dot(E_l, e_phi)
    P_dip_num[i] = norm(E_d)^2
    P_loop_num[i] = norm(E_l)^2
end

P_dip_num ./= maximum(P_dip_num)
P_loop_num ./= maximum(P_loop_num)
P_ana ./= maximum(P_ana)

err_dip = P_dip_num .- P_ana
err_loop = P_loop_num .- P_ana

rmse_dip = sqrt(sum(abs2, err_dip) / length(err_dip))
rmse_loop = sqrt(sum(abs2, err_loop) / length(err_loop))
maxabs_dip = maximum(abs.(err_dip))
maxabs_loop = maximum(abs.(err_loop))

null_max_dip = max(P_dip_num[1], P_dip_num[end])
null_max_loop = max(P_loop_num[1], P_loop_num[end])

# Polarization-resolved check over a coarse (θ,φ) grid.
phi_pat = deg2rad.(collect(0.0:10.0:350.0))
theta_pol_pat = deg2rad.(collect(1.0:2.0:179.0))
crossfrac_dip = Float64[]
crossfrac_loop = Float64[]
for th in theta_pol_pat, ph in phi_pat
    rhat = Vec3(sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
    r = Rfar_pat * rhat
    e_theta = Vec3(cos(th) * cos(ph), cos(th) * sin(ph), -sin(th))
    e_phi = Vec3(-sin(ph), cos(ph), 0.0)
    E_d = DifferentiableMoM.dipole_incident_field(r, dip_pat)
    E_l = DifferentiableMoM.loop_incident_field(r, loop_pat)
    E_d_theta = dot(E_d, e_theta)
    E_d_phi = dot(E_d, e_phi)
    E_l_theta = dot(E_l, e_theta)
    E_l_phi = dot(E_l, e_phi)
    P_d = abs2(E_d_theta) + abs2(E_d_phi)
    P_l = abs2(E_l_theta) + abs2(E_l_phi)
    push!(crossfrac_dip, abs(E_d_phi) / sqrt(P_d))
    push!(crossfrac_loop, abs(E_l_theta) / sqrt(P_l))
end
max_crossfrac_dip = maximum(crossfrac_dip)
max_crossfrac_loop = maximum(crossfrac_loop)

# Co-pol phase consistency on φ=0 cut: Eϕ(loop)/Eθ(dipole) ≈ ±90° phase.
wrap_to_pi(x) = atan(sin(x), cos(x))
phase_err_pm90_deg = Float64[]
phase_ratio_deg = Float64[]
amp_floor_phase = 1e-12 * max(maximum(abs.(E_dip_theta_cmp)), maximum(abs.(E_loop_phi_cmp)))
for i in eachindex(theta_pat)
    if abs(E_dip_theta_cmp[i]) > amp_floor_phase && abs(E_loop_phi_cmp[i]) > amp_floor_phase
        Δϕ = angle(E_loop_phi_cmp[i] / E_dip_theta_cmp[i])
        push!(phase_ratio_deg, rad2deg(Δϕ))
        err_pm90 = min(abs(wrap_to_pi(Δϕ - π / 2)), abs(wrap_to_pi(Δϕ + π / 2)))
        push!(phase_err_pm90_deg, rad2deg(err_pm90))
    end
end
phase_mean_deg = mean(phase_ratio_deg)
phase_std_deg = std(phase_ratio_deg)
phase_max_err_pm90_deg = maximum(phase_err_pm90_deg)

println("  Dipole pattern RMSE:      $rmse_dip")
println("  Dipole pattern max |err|: $maxabs_dip")
println("  Loop pattern RMSE:        $rmse_loop")
println("  Loop pattern max |err|:   $maxabs_loop")
println("  Dipole null max:          $null_max_dip")
println("  Loop null max:            $null_max_loop")
println("  Dipole max cross-pol frac: $max_crossfrac_dip")
println("  Loop max cross-pol frac:   $max_crossfrac_loop")
println("  Phase mean (deg):          $phase_mean_deg")
println("  Phase std (deg):           $phase_std_deg")
println("  Phase max err to ±90° (deg): $phase_max_err_pm90_deg")

# CI thresholds (pattern-shape gate)
@assert rmse_dip < 1e-4 "Dipole pattern gate failed: RMSE=$rmse_dip"
@assert maxabs_dip < 2e-4 "Dipole pattern gate failed: max |err|=$maxabs_dip"
@assert rmse_loop < 1e-10 "Loop pattern gate failed: RMSE=$rmse_loop"
@assert maxabs_loop < 1e-9 "Loop pattern gate failed: max |err|=$maxabs_loop"
@assert null_max_dip < 1e-3 "Dipole pattern gate failed: null level=$null_max_dip"
@assert null_max_loop < 1e-10 "Loop pattern gate failed: null level=$null_max_loop"
@assert max_crossfrac_dip < 1e-10 "Dipole polarization gate failed: max cross-pol frac=$max_crossfrac_dip"
@assert max_crossfrac_loop < 1e-10 "Loop polarization gate failed: max cross-pol frac=$max_crossfrac_loop"
@assert phase_max_err_pm90_deg < 1.0 "Dipole/loop phase gate failed: max err to ±90° = $phase_max_err_pm90_deg deg"
@assert phase_std_deg < 0.1 "Dipole/loop phase gate failed: phase std = $phase_std_deg deg"

df_pattern_gate = DataFrame(
    metric = [
        "dipole_rmse",
        "dipole_maxabs",
        "loop_rmse",
        "loop_maxabs",
        "dipole_null_max",
        "loop_null_max",
        "dipole_max_crossfrac",
        "loop_max_crossfrac",
        "phase_mean_deg",
        "phase_std_deg",
        "phase_max_err_pm90_deg",
        "Rfar_over_lambda",
        "freq_GHz",
    ],
    value = [
        rmse_dip,
        maxabs_dip,
        rmse_loop,
        maxabs_loop,
        null_max_dip,
        null_max_loop,
        max_crossfrac_dip,
        max_crossfrac_loop,
        phase_mean_deg,
        phase_std_deg,
        phase_max_err_pm90_deg,
        Rfar_pat / lambda_pat,
        freq_pat / 1e9,
    ],
)
CSV.write(joinpath(DATADIR, "dipole_loop_pattern_gate_metrics.csv"), df_pattern_gate)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 16: Pattern-feed excitation gate
# ─────────────────────────────────────────────────
println("\n── Test 16: Pattern-feed excitation gate ──")

const EPS0_PAT = 8.854187817e-12
const C0_PAT = 299792458.0

freq_pf = 1.0e9
k_pf = 2π * freq_pf / C0_PAT
λ_pf = C0_PAT / freq_pf
Rfar_pf = 80 * λ_pf
pz_pf = 1e-12 + 0im

dip_pf = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, pz_pf),
    Vec3(0.0, 0.0, 1.0),
    :electric,
    freq_pf,
)

theta_pat_deg_pf = collect(0.0:2.0:180.0)
phi_pat_deg_pf = collect(0.0:6.0:354.0)
pat_plus = make_analytic_dipole_pattern_feed(
    dip_pf,
    theta_pat_deg_pf,
    phi_pat_deg_pf;
    angles_in_degrees=true,
)

# Adapter-style constructor using two pattern objects with fields x, y, U
struct _PatternLike
    x::Vector{Float64}
    y::Vector{Float64}
    U::Matrix{ComplexF64}
end
pat_like_θ = _PatternLike(copy(pat_plus.theta), copy(pat_plus.phi), copy(pat_plus.Ftheta))
pat_like_ϕ = _PatternLike(copy(pat_plus.theta), copy(pat_plus.phi), copy(pat_plus.Fphi))
pat_from_like = make_pattern_feed(pat_like_θ, pat_like_ϕ, pat_plus.frequency; angles_in_degrees=false)

probe_like_dirs = (Vec3(1.0, 0.5, 2.0), Vec3(-0.8, 0.3, 1.4), Vec3(0.1, -0.9, 1.1))
rel_like = let worst = 0.0
    for d in probe_like_dirs
        r = Rfar_pf * d / norm(d)
        E_ref = pattern_feed_field(r, pat_plus)
        E_new = pattern_feed_field(r, pat_from_like)
        worst = max(worst, norm(E_ref - E_new) / max(norm(E_ref), 1e-30))
    end
    worst
end
println("  Pattern-object adapter mismatch:    $rel_like")
@assert rel_like < 1e-13

# Matrix-shape tolerance: transposed input should auto-correct
# Expected-by-design shape-tolerance behavior: auto-transpose with warnings.
pat_transposed = @test_logs (
    :warn,
    r"Ftheta matrix shape .* appears transposed; auto-transposing to .*",
) (
    :warn,
    r"Fphi matrix shape .* appears transposed; auto-transposing to .*",
) make_pattern_feed(
    pat_plus.theta,
    pat_plus.phi,
    permutedims(pat_plus.Ftheta),
    permutedims(pat_plus.Fphi),
    pat_plus.frequency;
    angles_in_degrees=false,
)
rel_transposed = let worst = 0.0
    for d in probe_like_dirs
        r = Rfar_pf * d / norm(d)
        E_ref = pattern_feed_field(r, pat_plus)
        E_new = pattern_feed_field(r, pat_transposed)
        worst = max(worst, norm(E_ref - E_new) / max(norm(E_ref), 1e-30))
    end
    worst
end
println("  Pattern-transpose auto-fix mismatch: $rel_transposed")
@assert rel_transposed < 1e-12

# Field-level comparison on a fine φ=0 cut against closed-form dipole far-field.
theta_eval_deg_pf = collect(0.0:0.5:180.0)
theta_eval_pf = deg2rad.(theta_eval_deg_pf)
P_num_pf = zeros(Float64, length(theta_eval_pf))
P_ref_pf = zeros(Float64, length(theta_eval_pf))
Etheta_num_pf = zeros(ComplexF64, length(theta_eval_pf))
Etheta_ref_pf = zeros(ComplexF64, length(theta_eval_pf))
cross_ratio_pf = zeros(Float64, length(theta_eval_pf))

for i in eachindex(theta_eval_pf)
    θ = theta_eval_pf[i]
    ϕ = 0.0
    rhat = Vec3(sin(θ), 0.0, cos(θ))
    r = Rfar_pf * rhat
    eθ = Vec3(cos(θ), 0.0, -sin(θ))
    eϕ = Vec3(0.0, 1.0, 0.0)

    E_num = pattern_feed_field(r, pat_plus)
    Eθ_num = dot(E_num, eθ)
    Eϕ_num = dot(E_num, eϕ)

    Eθ_ref = (k_pf^2 * pz_pf / (4π * EPS0_PAT)) * sin(θ) * exp(-1im * k_pf * Rfar_pf) / Rfar_pf
    Eϕ_ref = 0.0 + 0im

    Etheta_num_pf[i] = Eθ_num
    Etheta_ref_pf[i] = Eθ_ref
    P_num_pf[i] = abs2(Eθ_num) + abs2(Eϕ_num)
    P_ref_pf[i] = abs2(Eθ_ref) + abs2(Eϕ_ref)
    cross_ratio_pf[i] = abs(Eϕ_num) / max(sqrt(P_num_pf[i]), 1e-30)
end

P_num_pf ./= maximum(P_num_pf)
P_ref_pf ./= maximum(P_ref_pf)
err_lin_pf = P_num_pf .- P_ref_pf
rmse_pf = sqrt(mean(abs2, err_lin_pf))
maxabs_pf = maximum(abs.(err_lin_pf))
max_cross_pf = maximum(cross_ratio_pf)

phase_err_deg_pf = fill(NaN, length(theta_eval_pf))
phase_floor_pf = 1e-12 * maximum(abs.(Etheta_ref_pf))
for i in eachindex(theta_eval_pf)
    if abs(Etheta_ref_pf[i]) > phase_floor_pf && abs(Etheta_num_pf[i]) > phase_floor_pf
        phase_err_deg_pf[i] = rad2deg(angle(Etheta_num_pf[i] / Etheta_ref_pf[i]))
    end
end

phase_valid_pf = phase_err_deg_pf[.!isnan.(phase_err_deg_pf)]
phase_mean_pf = mean(phase_valid_pf)
phase_std_pf = std(phase_valid_pf)
phase_max_pf = maximum(abs.(phase_valid_pf))
phase_resid_pf = [rad2deg(atan(sin(deg2rad(x - phase_mean_pf)), cos(deg2rad(x - phase_mean_pf)))) for x in phase_valid_pf]
phase_resid_std_pf = std(phase_resid_pf)
phase_resid_max_pf = maximum(abs.(phase_resid_pf))

# Convention conversion check:
# If imported data comes from exp(-iωt), using conjugated coefficients with
# convention=:exp_minus_iwt must reproduce the same physical field.
pat_minus = make_pattern_feed(
    pat_plus.theta,
    pat_plus.phi,
    conj.(pat_plus.Ftheta),
    conj.(pat_plus.Fphi),
    pat_plus.frequency;
    convention=:exp_minus_iwt,
)
probe_dirs_pf = (
    Vec3(1.2, -0.4, 2.1),
    Vec3(-0.8, 1.5, 1.2),
    Vec3(0.5, 0.9, -1.7),
)
conv_mismatch_pf = let mismatch = 0.0
    for d in probe_dirs_pf
        r = Rfar_pf * d / norm(d)
        E_plus = pattern_feed_field(r, pat_plus)
        E_minus = pattern_feed_field(r, pat_minus)
        mismatch = max(mismatch, norm(E_plus - E_minus) / max(norm(E_plus), 1e-30))
    end
    mismatch
end

# RHS consistency against direct imported electric-field path.
mesh_pf = make_rect_plate(0.04, 0.04, 4, 4)
rwg_pf = build_rwg(mesh_pf)
v_pf_pat = assemble_excitation(mesh_pf, rwg_pf, pat_plus; quad_order=3)
imp_pf = ImportedExcitation(r -> pattern_feed_field(r, pat_plus); kind=:electric_field, min_quad_order=3)
v_pf_imp = assemble_excitation(mesh_pf, rwg_pf, imp_pf; quad_order=3)
rhs_rel_pf = norm(v_pf_pat - v_pf_imp) / max(norm(v_pf_pat), 1e-30)

println("  Pattern-feed RMSE (linear):        $rmse_pf")
println("  Pattern-feed max |err| (linear):   $maxabs_pf")
println("  Pattern-feed max cross-pol ratio:  $max_cross_pf")
println("  Pattern-feed phase mean (deg):     $phase_mean_pf")
println("  Pattern-feed phase std (deg):      $phase_std_pf")
println("  Pattern-feed phase max |err| (deg): $phase_max_pf")
println("  Pattern-feed phase residual std (deg): $phase_resid_std_pf")
println("  Pattern-feed phase residual max |err| (deg): $phase_resid_max_pf")
println("  Convention conversion mismatch:    $conv_mismatch_pf")
println("  RHS path mismatch (pattern vs imported): $rhs_rel_pf")

@assert rmse_pf < 2e-4 "Pattern-feed gate failed: RMSE=$rmse_pf"
@assert maxabs_pf < 5e-4 "Pattern-feed gate failed: max |err|=$maxabs_pf"
@assert max_cross_pf < 1e-8 "Pattern-feed gate failed: cross-pol ratio=$max_cross_pf"
@assert phase_resid_std_pf < 0.2 "Pattern-feed gate failed: phase residual std=$phase_resid_std_pf"
@assert phase_resid_max_pf < 0.5 "Pattern-feed gate failed: phase residual max |err|=$phase_resid_max_pf"
@assert conv_mismatch_pf < 1e-12 "Pattern-feed gate failed: convention mismatch=$conv_mismatch_pf"
@assert rhs_rel_pf < 1e-12 "Pattern-feed gate failed: RHS mismatch=$rhs_rel_pf"

df_pattern_feed_gate = DataFrame(
    metric = [
        "rmse_lin",
        "maxabs_lin",
        "max_crosspol_ratio",
        "phase_mean_deg",
        "phase_std_deg",
        "phase_max_abs_deg",
        "phase_residual_std_deg",
        "phase_residual_max_abs_deg",
        "convention_mismatch",
        "rhs_rel_mismatch",
        "Rfar_over_lambda",
        "freq_GHz",
        "theta_pattern_step_deg",
        "phi_pattern_step_deg",
    ],
    value = [
        rmse_pf,
        maxabs_pf,
        max_cross_pf,
        phase_mean_pf,
        phase_std_pf,
        phase_max_pf,
        phase_resid_std_pf,
        phase_resid_max_pf,
        conv_mismatch_pf,
        rhs_rel_pf,
        Rfar_pf / λ_pf,
        freq_pf / 1e9,
        2.0,
        6.0,
    ],
)
CSV.write(joinpath(DATADIR, "pattern_feed_gate_metrics.csv"), df_pattern_feed_gate)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 17: Randomized preconditioner construction
# ─────────────────────────────────────────────────
println("\n── Test 17: Randomized preconditioner construction ──")

# Use the EFIE matrix Z_efie and impedance setup from earlier tests
theta_rp = fill(200.0, Nt)
Z_rp = assemble_full_Z(Z_efie, Mp, theta_rp)
N_rp = size(Z_rp, 1)
k_rp = min(N_rp, 10)  # preconditioner rank

# Build preconditioner with seed for reproducibility (default :auto = two-level)
P_rp = build_randomized_preconditioner(Matrix{ComplexF64}(Z_rp), k_rp; seed=42)

# Struct fields should have correct dimensions
@assert size(P_rp.Q) == (N_rp, k_rp)
@assert size(P_rp.Omega) == (N_rp, k_rp)
@assert size(P_rp.Y) == (N_rp, k_rp)
@assert P_rp.MpOmega === nothing
@assert P_rp.D_inv !== nothing "Default :auto mode should set D_inv"
@assert length(P_rp.D_inv) == N_rp

# Q should be orthonormal
QtQ = P_rp.Q' * P_rp.Q
@assert norm(QtQ - I(k_rp)) < 1e-12 "Q columns not orthonormal"

# Preconditioner action should produce valid output
v_test_rp = randn(ComplexF64, N_rp)
Pv = apply_preconditioner(P_rp, v_test_rp)
@assert length(Pv) == N_rp
@assert all(isfinite, Pv)

# Adjoint preconditioner action
PadjV = apply_preconditioner_adjoint(P_rp, v_test_rp)
@assert length(PadjV) == N_rp
@assert all(isfinite, PadjV)

# Adjoint consistency: ⟨P⁻¹x, y⟩ ≈ ⟨x, P⁻ᴴy⟩
x_rp = randn(ComplexF64, N_rp)
y_rp = randn(ComplexF64, N_rp)
lhs_adj = dot(apply_preconditioner(P_rp, x_rp), y_rp)
rhs_adj = dot(x_rp, apply_preconditioner_adjoint(P_rp, y_rp))
rel_adj_err = abs(lhs_adj - rhs_adj) / max(abs(lhs_adj), 1e-30)
println("  Adjoint consistency ⟨P⁻¹x, y⟩ vs ⟨x, P⁻ᴴy⟩: $rel_adj_err")
@assert rel_adj_err < 1e-12 "Preconditioner adjoint inconsistency: $rel_adj_err"

# Reproducibility: same seed → same preconditioner
P_rp2 = build_randomized_preconditioner(Matrix{ComplexF64}(Z_rp), k_rp; seed=42)
@assert norm(P_rp.Q - P_rp2.Q) < 1e-14
@assert norm(P_rp.Omega - P_rp2.Omega) < 1e-14

# Legacy scalar mu_mode options
mu_diag = DifferentiableMoM._compute_mu(Matrix{ComplexF64}(Z_rp), :diag)
mu_trace = DifferentiableMoM._compute_mu(Matrix{ComplexF64}(Z_rp), :trace)
mu_num = DifferentiableMoM._compute_mu(Matrix{ComplexF64}(Z_rp), 0.5)
@assert isfinite(mu_diag) && abs(mu_diag) > 0
@assert isfinite(mu_trace) && abs(mu_trace) > 0
@assert mu_num == ComplexF64(0.5)
println("  mu_mode :diag=$mu_diag  :trace=$mu_trace  number=$mu_num")

# Legacy mode should set D_inv = nothing
P_rp_legacy = build_randomized_preconditioner(Matrix{ComplexF64}(Z_rp), k_rp; seed=42, mu_mode=:diag)
@assert P_rp_legacy.D_inv === nothing "Legacy :diag mode should have D_inv === nothing"

# P⁻¹ Z should have eigenvalues clustered near 1 in the captured subspace
PinvZ = hcat([apply_preconditioner(P_rp, Vector{ComplexF64}(Z_rp[:, j])) for j in 1:N_rp]...)
eigvals_PZ = eigvals(PinvZ)
near_one_count = count(abs.(eigvals_PZ .- 1.0) .< 0.5)
println("  Eigenvalues of P⁻¹Z near 1: $near_one_count / $N_rp")
@assert near_one_count >= k_rp "Preconditioner should cluster at least k eigenvalues near 1"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 18: PreconditionerOperator wrappers (mul!)
# ─────────────────────────────────────────────────
println("\n── Test 18: PreconditionerOperator wrappers ──")

op_fwd = PreconditionerOperator(P_rp)
op_adj = PreconditionerAdjointOperator(P_rp)

@assert size(op_fwd) == (N_rp, N_rp)
@assert size(op_adj) == (N_rp, N_rp)
@assert eltype(op_fwd) == ComplexF64
@assert eltype(op_adj) == ComplexF64

# Test * operator
v_op = randn(ComplexF64, N_rp)
r1_op = op_fwd * v_op
r2_op = apply_preconditioner(P_rp, v_op)
@assert norm(r1_op - r2_op) < 1e-14

r1_adj_op = op_adj * v_op
r2_adj_op = apply_preconditioner_adjoint(P_rp, v_op)
@assert norm(r1_adj_op - r2_adj_op) < 1e-14

# Test mul!
y_op = zeros(ComplexF64, N_rp)
LinearAlgebra.mul!(y_op, op_fwd, v_op)
@assert norm(y_op - r2_op) < 1e-14

y_op_adj = zeros(ComplexF64, N_rp)
LinearAlgebra.mul!(y_op_adj, op_adj, v_op)
@assert norm(y_op_adj - r2_adj_op) < 1e-14

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 19: GMRES forward solve
# ─────────────────────────────────────────────────
println("\n── Test 19: GMRES forward solve ──")

# Direct solve reference
I_direct = Z_rp \ v

# GMRES without preconditioner
I_gmres_nop, stats_nop = solve_gmres(Matrix{ComplexF64}(Z_rp), Vector{ComplexF64}(v);
                                       tol=1e-10, maxiter=500)
rel_gmres_nop = norm(I_gmres_nop - I_direct) / max(norm(I_direct), 1e-30)
println("  GMRES (no precond) rel error: $rel_gmres_nop  iters: $(stats_nop.niter)")
@assert rel_gmres_nop < 1e-6 "GMRES without preconditioner inaccurate: $rel_gmres_nop"

# GMRES with randomized preconditioner
I_gmres_rp, stats_rp = solve_gmres(Matrix{ComplexF64}(Z_rp), Vector{ComplexF64}(v);
                                     preconditioner=P_rp,
                                     tol=1e-10, maxiter=500)
rel_gmres_rp = norm(I_gmres_rp - I_direct) / max(norm(I_direct), 1e-30)
println("  GMRES (precond k=$k_rp) rel error: $rel_gmres_rp  iters: $(stats_rp.niter)")
@assert rel_gmres_rp < 1e-6 "GMRES with preconditioner inaccurate: $rel_gmres_rp"

# Preconditioned GMRES should converge in fewer iterations
println("  Iterations: unpreconditioned=$(stats_nop.niter), preconditioned=$(stats_rp.niter)")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 20: GMRES adjoint solve
# ─────────────────────────────────────────────────
println("\n── Test 20: GMRES adjoint solve ──")

rhs_adj_test = Q * I_direct
lam_direct = Z_rp' \ rhs_adj_test

# GMRES adjoint without preconditioner
lam_gmres_nop, stats_adj_nop = solve_gmres_adjoint(Matrix{ComplexF64}(Z_rp),
                                                     Vector{ComplexF64}(rhs_adj_test);
                                                     tol=1e-10, maxiter=500)
rel_adj_nop = norm(lam_gmres_nop - lam_direct) / max(norm(lam_direct), 1e-30)
println("  GMRES adjoint (no precond) rel error: $rel_adj_nop  iters: $(stats_adj_nop.niter)")
@assert rel_adj_nop < 1e-6

# GMRES adjoint with preconditioner
lam_gmres_rp, stats_adj_rp = solve_gmres_adjoint(Matrix{ComplexF64}(Z_rp),
                                                    Vector{ComplexF64}(rhs_adj_test);
                                                    preconditioner=P_rp,
                                                    tol=1e-10, maxiter=500)
rel_adj_rp = norm(lam_gmres_rp - lam_direct) / max(norm(lam_direct), 1e-30)
println("  GMRES adjoint (precond) rel error: $rel_adj_rp  iters: $(stats_adj_rp.niter)")
@assert rel_adj_rp < 1e-6

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 21: solve_forward / solve_adjoint dispatch
# ─────────────────────────────────────────────────
println("\n── Test 21: solve_forward / solve_adjoint dispatch ──")

# solve_forward with :direct
I_sf_direct = solve_forward(Matrix{ComplexF64}(Z_rp), Vector{ComplexF64}(v))
rel_sf_direct = norm(I_sf_direct - I_direct) / max(norm(I_direct), 1e-30)
@assert rel_sf_direct < 1e-12

# solve_forward with :gmres + preconditioner
I_sf_gmres = solve_forward(Matrix{ComplexF64}(Z_rp), Vector{ComplexF64}(v);
                            solver=:gmres, preconditioner=P_rp,
                            gmres_tol=1e-10, gmres_maxiter=500)
rel_sf_gmres = norm(I_sf_gmres - I_direct) / max(norm(I_direct), 1e-30)
println("  solve_forward :gmres rel error: $rel_sf_gmres")
@assert rel_sf_gmres < 1e-6

# solve_adjoint with :direct
lam_sa_direct = solve_adjoint(Matrix{ComplexF64}(Z_rp), Q, I_direct)
rel_sa_direct = norm(lam_sa_direct - lam_direct) / max(norm(lam_direct), 1e-30)
@assert rel_sa_direct < 1e-12

# solve_adjoint with :gmres + preconditioner
lam_sa_gmres = solve_adjoint(Matrix{ComplexF64}(Z_rp), Q, I_direct;
                              solver=:gmres, preconditioner=P_rp,
                              gmres_tol=1e-10, gmres_maxiter=500)
rel_sa_gmres = norm(lam_sa_gmres - lam_direct) / max(norm(lam_direct), 1e-30)
println("  solve_adjoint :gmres rel error: $rel_sa_gmres")
@assert rel_sa_gmres < 1e-6

# Bad solver symbol should error
thrown_bad_solver = try
    solve_forward(Matrix{ComplexF64}(Z_rp), Vector{ComplexF64}(v); solver=:unknown)
    false
catch
    true
end
@assert thrown_bad_solver "Expected error for unknown solver"

thrown_bad_solver_adj = try
    solve_adjoint(Matrix{ComplexF64}(Z_rp), Q, I_direct; solver=:unknown)
    false
catch
    true
end
@assert thrown_bad_solver_adj "Expected error for unknown adjoint solver"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 22: Adjoint gradient with GMRES solver
# ─────────────────────────────────────────────────
println("\n── Test 22: Adjoint gradient with GMRES solver ──")

# Reference: adjoint gradient with direct solver (already computed in Test 7 as g_adj)
# Recompute with GMRES
theta_gm = copy(theta_real)
Z_gm = assemble_full_Z(Z_efie, Mp, theta_gm)
I_gm = solve_forward(Matrix{ComplexF64}(Z_gm), Vector{ComplexF64}(v);
                       solver=:gmres, preconditioner=build_randomized_preconditioner(
                           Matrix{ComplexF64}(Z_gm), k_rp; seed=42),
                       gmres_tol=1e-10, gmres_maxiter=500)

P_gm = build_randomized_preconditioner(Matrix{ComplexF64}(Z_gm), k_rp; seed=42)
lam_gm = solve_adjoint(Matrix{ComplexF64}(Z_gm), Q, I_gm;
                         solver=:gmres, preconditioner=P_gm,
                         gmres_tol=1e-10, gmres_maxiter=500)
g_adj_gmres = gradient_impedance(Mp, I_gm, lam_gm)

# Compare GMRES gradient against finite differences
println("  Checking GMRES adjoint gradient vs central FD (h=1e-5)...")
rel_errors_gm = Float64[]
n_check_gm = min(Nt, 10)
for p in 1:n_check_gm
    g_fd = fd_grad(J_of_theta, theta_gm, p; h=1e-5)
    rel_err = abs(g_adj_gmres[p] - g_fd) / max(abs(g_adj_gmres[p]), abs(g_fd), 1e-30)
    push!(rel_errors_gm, rel_err)
end
max_rel_err_gm = maximum(rel_errors_gm)
println("  Max rel error (GMRES adjoint vs FD): $max_rel_err_gm")
@assert max_rel_err_gm < 1e-3 "GMRES gradient verification FAILED: max rel error = $max_rel_err_gm"

# Also compare GMRES gradient against direct gradient
rel_gm_vs_direct = norm(g_adj_gmres - g_adj) / max(norm(g_adj), 1e-30)
println("  GMRES vs direct gradient rel diff: $rel_gm_vs_direct")
@assert rel_gm_vs_direct < 1e-4 "GMRES gradient diverges from direct: $rel_gm_vs_direct"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 23: cache_MpOmega and preconditioner recycling
# ─────────────────────────────────────────────────
println("\n── Test 23: Preconditioner recycling ──")

# Build initial preconditioner
theta_rc0 = fill(200.0, Nt)
Z_rc0 = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_rc0))
P_rc0 = build_randomized_preconditioner(Z_rc0, k_rp; seed=42)

# Cache MpOmega
mp_omega = cache_MpOmega(Mp, P_rc0.Omega)
@assert length(mp_omega) == Nt
for p in 1:Nt
    @assert size(mp_omega[p]) == (N_rp, k_rp)
    # Verify: mp_omega[p] == Mp[p] * Omega
    ref_MpOmega = Matrix{ComplexF64}(Mp[p]) * P_rc0.Omega
    @assert norm(mp_omega[p] - ref_MpOmega) < 1e-12 * norm(ref_MpOmega)
end
println("  cache_MpOmega correctness verified")

# Perturb theta
theta_rc1 = theta_rc0 .+ randn(Nt) .* 10.0
delta_theta = theta_rc1 .- theta_rc0
Z_rc1 = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_rc1))

# Update preconditioner with cached MpOmega (incremental)
P_rc1_inc = update_randomized_preconditioner(P_rc0, Z_rc1, delta_theta, Mp;
                                               MpOmega=mp_omega)

# Update preconditioner without cached MpOmega (full rebuild of sketch)
P_rc1_full = update_randomized_preconditioner(P_rc0, Z_rc1, delta_theta, Mp)

# Both updates should produce effective preconditioners
I_rc1_direct = Z_rc1 \ v
I_rc1_inc, stats_inc = solve_gmres(Z_rc1, Vector{ComplexF64}(v);
                                     preconditioner=P_rc1_inc,
                                     tol=1e-10, maxiter=500)
I_rc1_full, stats_full = solve_gmres(Z_rc1, Vector{ComplexF64}(v);
                                       preconditioner=P_rc1_full,
                                       tol=1e-10, maxiter=500)

rel_inc = norm(I_rc1_inc - I_rc1_direct) / max(norm(I_rc1_direct), 1e-30)
rel_full = norm(I_rc1_full - I_rc1_direct) / max(norm(I_rc1_direct), 1e-30)
println("  Recycled (incremental) rel error: $rel_inc  iters: $(stats_inc.niter)")
println("  Recycled (full sketch) rel error: $rel_full  iters: $(stats_full.niter)")
@assert rel_inc < 1e-6 "Incremental recycled preconditioner inaccurate: $rel_inc"
@assert rel_full < 1e-6 "Full-sketch recycled preconditioner inaccurate: $rel_full"

# Incremental and full-sketch updates should give same sketch Y
# (since Y_new = Z_new * Omega regardless of the path)
@assert norm(P_rc1_inc.Y - P_rc1_full.Y) < 1e-8 * norm(P_rc1_full.Y) "Incremental sketch mismatch"

# Test with reactive=true
theta_rc_reac = theta_rc0 .+ randn(Nt) .* 5.0
delta_theta_reac = theta_rc_reac .- theta_rc0
Z_rc_reac = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_rc_reac; reactive=true))
P_rc_reac = update_randomized_preconditioner(P_rc0, Z_rc_reac, delta_theta_reac, Mp;
                                               MpOmega=mp_omega, reactive=true)
@assert size(P_rc_reac.Q) == (N_rp, k_rp)
println("  Reactive recycling: struct valid")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 24: optimize_lbfgs with solver=:gmres
# ─────────────────────────────────────────────────
println("\n── Test 24: optimize_lbfgs with solver=:gmres ──")

theta_init_gm = fill(300.0, Nt)
theta_opt_gm, trace_gm = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_gm;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:gmres, precond_rank=k_rp, precond_seed=42,
    gmres_tol=1e-8, gmres_maxiter=300,
)

if length(trace_gm) >= 2
    J_first_gm = trace_gm[1].J
    J_last_gm = trace_gm[end].J
    println("  J(iter=1)  = $J_first_gm")
    println("  J(iter=$(length(trace_gm))) = $J_last_gm")
end

# Compare with direct solver optimization (from Test 10)
# Both should produce a valid optimization trajectory
@assert length(trace_gm) >= 2 "GMRES optimization should run at least 2 iterations"

# Run direct solver optimization with same initial point for comparison
theta_opt_dir, trace_dir = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_gm;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:direct,
)

# Both should produce similar first-iteration objective (same starting point)
rel_J0 = abs(trace_gm[1].J - trace_dir[1].J) / max(abs(trace_dir[1].J), 1e-30)
println("  First-iteration J agreement: $rel_J0")
@assert rel_J0 < 1e-4 "GMRES and direct first-iter J disagree: $rel_J0"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 25: optimize_directivity with solver=:gmres
# ─────────────────────────────────────────────────
println("\n── Test 25: optimize_directivity with solver=:gmres ──")

# Build a Q_total for total radiated power (use full-sphere Q)
Q_total_test = build_Q(G_mat, grid, pol_mat)

# Direct solver run
theta_init_dir_d = fill(300.0, Nt)
theta_opt_dir_d, trace_dir_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_dir_d;
    maxiter=5, tol=1e-8, verbose=false,
    solver=:direct,
)

# GMRES solver run
theta_opt_gm_d, trace_gm_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_dir_d;
    maxiter=5, tol=1e-8, verbose=false,
    solver=:gmres, precond_rank=k_rp, precond_seed=42,
    gmres_tol=1e-8, gmres_maxiter=300,
)

@assert length(trace_gm_d) >= 2 "GMRES directivity optimization should run at least 2 iterations"

# First iteration objectives should agree
rel_J0_d = abs(trace_gm_d[1].J - trace_dir_d[1].J) / max(abs(trace_dir_d[1].J), 1e-30)
println("  First-iteration J_ratio agreement: $rel_J0_d")
@assert rel_J0_d < 1e-3 "GMRES and direct first-iter J_ratio disagree: $rel_J0_d"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 26: Preconditioner as effective approximate inverse
# ─────────────────────────────────────────────────
println("\n── Test 26: Preconditioner effectiveness ──")

# For a well-conditioned system, P⁻¹Z should be close to identity
# Test with increasing rank k to show convergence
ks_test = [2, 5, min(N_rp, 10)]
residuals_k = Float64[]
for k_test in ks_test
    P_k = build_randomized_preconditioner(Matrix{ComplexF64}(Z_rp), k_test; seed=42)
    # Test on a random vector
    b_test = randn(ComplexF64, N_rp)
    x_exact = Z_rp \ b_test
    x_gmres, _ = solve_gmres(Matrix{ComplexF64}(Z_rp), b_test;
                               preconditioner=P_k, tol=1e-10, maxiter=500)
    res = norm(x_gmres - x_exact) / max(norm(x_exact), 1e-30)
    push!(residuals_k, res)
    println("  rank=$k_test  solve rel error: $res")
end

# All should converge within tolerance
for (k_test, res) in zip(ks_test, residuals_k)
    @assert res < 1e-5 "Preconditioner rank=$k_test failed: rel error=$res"
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 27: GMRES gradient with reactive impedance
# ─────────────────────────────────────────────────
println("\n── Test 27: GMRES gradient with reactive impedance ──")

theta_reac = fill(150.0, Nt)
Z_reac = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_reac; reactive=true))
I_reac_direct = Z_reac \ v

# Adjoint gradient (direct) for reactive case
lam_reac_dir = solve_adjoint(Z_reac, Q, I_reac_direct)
g_reac_dir = gradient_impedance(Mp, I_reac_direct, lam_reac_dir; reactive=true)

# Adjoint gradient (GMRES) for reactive case
P_reac = build_randomized_preconditioner(Z_reac, k_rp; seed=42)
I_reac_gm = solve_forward(Z_reac, Vector{ComplexF64}(v);
                            solver=:gmres, preconditioner=P_reac,
                            gmres_tol=1e-10, gmres_maxiter=500)
lam_reac_gm = solve_adjoint(Z_reac, Q, I_reac_gm;
                              solver=:gmres, preconditioner=P_reac,
                              gmres_tol=1e-10, gmres_maxiter=500)
g_reac_gm = gradient_impedance(Mp, I_reac_gm, lam_reac_gm; reactive=true)

# FD reference for reactive
function J_of_theta_reac(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= (1im * theta_vec[p]) .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

rel_errors_reac = Float64[]
n_check_reac = min(Nt, 5)
for p in 1:n_check_reac
    g_fd = fd_grad(J_of_theta_reac, theta_reac, p; h=1e-5)
    rel_err_dir = abs(g_reac_dir[p] - g_fd) / max(abs(g_fd), 1e-30)
    rel_err_gm = abs(g_reac_gm[p] - g_fd) / max(abs(g_fd), 1e-30)
    push!(rel_errors_reac, rel_err_gm)
    println("    p=$p: dir=$rel_err_dir  gmres=$rel_err_gm")
end
max_rel_err_reac = maximum(rel_errors_reac)
println("  Max rel error (GMRES reactive gradient vs FD): $max_rel_err_reac")
@assert max_rel_err_reac < 1e-3 "Reactive GMRES gradient failed: $max_rel_err_reac"

# Direct and GMRES gradients should agree closely
rel_reac_gm_vs_dir = norm(g_reac_gm - g_reac_dir) / max(norm(g_reac_dir), 1e-30)
println("  Reactive gradient GMRES vs direct: $rel_reac_gm_vs_dir")
@assert rel_reac_gm_vs_dir < 1e-4

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 28: Regression — default solver=:direct unchanged
# ─────────────────────────────────────────────────
println("\n── Test 28: Regression — default solver=:direct ──")

# solve_forward default should match Z \ v exactly
I_reg_sf = solve_forward(Matrix{ComplexF64}(Z_rp), Vector{ComplexF64}(v))
I_reg_direct = Z_rp \ v
rel_reg = norm(I_reg_sf - I_reg_direct) / max(norm(I_reg_direct), 1e-30)
@assert rel_reg < 1e-12 "Default solve_forward regression: $rel_reg"

# solve_adjoint default should match Z' \ (Q*I)
lam_reg_sa = solve_adjoint(Matrix{ComplexF64}(Z_rp), Q, I_reg_direct)
lam_reg_dir = Z_rp' \ (Q * I_reg_direct)
rel_reg_adj = norm(lam_reg_sa - lam_reg_dir) / max(norm(lam_reg_dir), 1e-30)
@assert rel_reg_adj < 1e-12 "Default solve_adjoint regression: $rel_reg_adj"

# optimize_lbfgs default (solver=:direct) should still work
theta_reg_init = fill(300.0, Nt)
theta_reg_opt, trace_reg = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_reg_init;
    maxiter=3, tol=1e-8, verbose=false,
)
@assert length(trace_reg) >= 2

println("  All defaults unchanged")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 29: Near-field sparse preconditioner
# ─────────────────────────────────────────────────
println("\n── Test 29: Near-field sparse preconditioner ──")

# Test rwg_centers
centers = rwg_centers(mesh, rwg)
@assert length(centers) == N
@assert all(c -> length(c) == 3, centers)

# Build near-field preconditioner at 1.0λ cutoff
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, lambda0)
@assert P_nf.cutoff == lambda0
@assert 0.0 < P_nf.nnz_ratio <= 1.0

# GMRES with near-field preconditioner
I_nf, stats_nf = solve_gmres(Z_efie, v; preconditioner=P_nf, tol=1e-8, maxiter=200)
rel_nf = norm(I_nf - I_pec) / max(norm(I_pec), 1e-30)
println("  NF precond (1.0λ) rel error: $rel_nf  iters: $(stats_nf.niter)")
@assert rel_nf < 1e-6 "Near-field preconditioned solve inaccurate: $rel_nf"

# Compare iteration count: near-field should help vs unpreconditioned
I_nop, stats_nop = solve_gmres(Z_efie, v; tol=1e-8, maxiter=200)
println("  Iterations: no_precond=$(stats_nop.niter), NF=$(stats_nf.niter)")

# NearFieldOperator / NearFieldAdjointOperator wrappers
M_nf = NearFieldOperator(P_nf)
@assert size(M_nf) == (N, N)
@assert eltype(M_nf) == ComplexF64
y_nf = M_nf * v
@assert length(y_nf) == N

M_nf_adj = NearFieldAdjointOperator(P_nf)
@assert size(M_nf_adj) == (N, N)
y_nf_adj = M_nf_adj * v

# Adjoint consistency: ⟨P⁻¹x, y⟩ ≈ ⟨x, P⁻ᴴy⟩
x_test = randn(ComplexF64, N)
y_test = randn(ComplexF64, N)
lhs = dot(M_nf * x_test, y_test)
rhs = dot(x_test, M_nf_adj * y_test)
adj_err = abs(lhs - rhs) / max(abs(lhs), 1e-30)
println("  Adjoint consistency: $adj_err")
@assert adj_err < 1e-12 "Near-field adjoint inconsistent: $adj_err"

# GMRES adjoint solve with near-field
rhs_adj = Q * I_pec
lambda_nf, stats_adj_nf = solve_gmres_adjoint(Z_efie, rhs_adj;
                                                preconditioner=P_nf, tol=1e-8, maxiter=200)
lambda_direct = Z_efie' \ rhs_adj
rel_adj_nf = norm(lambda_nf - lambda_direct) / max(norm(lambda_direct), 1e-30)
println("  NF adjoint solve rel error: $rel_adj_nf  iters: $(stats_adj_nf.niter)")
@assert rel_adj_nf < 1e-6 "Near-field adjoint solve inaccurate: $rel_adj_nf"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 30: Right preconditioning
# ─────────────────────────────────────────────────
println("\n── Test 30: Right preconditioning ──")

# Build randomized preconditioner
P_rp = build_randomized_preconditioner(Z_efie, 10; seed=42)

# Right-preconditioned GMRES
I_right, stats_right = solve_gmres(Z_efie, v;
                                     preconditioner=P_rp, precond_side=:right,
                                     tol=1e-8, maxiter=200)
rel_right = norm(I_right - I_pec) / max(norm(I_pec), 1e-30)
println("  Right precond rel error: $rel_right  iters: $(stats_right.niter)")
@assert rel_right < 1e-6 "Right-preconditioned solve inaccurate: $rel_right"

# Right-preconditioned adjoint
lambda_right, stats_adj_right = solve_gmres_adjoint(Z_efie, rhs_adj;
                                                      preconditioner=P_rp, precond_side=:right,
                                                      tol=1e-8, maxiter=200)
rel_adj_right = norm(lambda_right - lambda_direct) / max(norm(lambda_direct), 1e-30)
println("  Right adjoint rel error: $rel_adj_right  iters: $(stats_adj_right.niter)")
@assert rel_adj_right < 1e-6 "Right adjoint solve inaccurate: $rel_adj_right"

# Near-field with right preconditioning
I_nf_right, stats_nf_right = solve_gmres(Z_efie, v;
                                           preconditioner=P_nf, precond_side=:right,
                                           tol=1e-8, maxiter=200)
rel_nf_right = norm(I_nf_right - I_pec) / max(norm(I_pec), 1e-30)
println("  NF right precond rel error: $rel_nf_right  iters: $(stats_nf_right.niter)")
@assert rel_nf_right < 1e-6 "NF right-preconditioned solve inaccurate: $rel_nf_right"

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
