# ex_beam_steer.jl — Beam-steering metasurface inverse design
#
# Canonical benchmark: optimize a reactive impedance sheet to steer
# a normally-incident plane wave to θ_s = 30° off broadside.
#
# Uses purely reactive (imaginary) impedance Z_s = iθ for lossless
# phase control — the physical mechanism for metasurface beam steering.
#
# Run: julia --project=. examples/ex_beam_steer.jl

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
println("Beam-Steering Metasurface Optimization")
println("="^60)

# ─────────────────────────────────────────────────
# 1. Problem Setup
# ─────────────────────────────────────────────────
freq = 3e9            # 3 GHz
c0   = 299792458.0
lambda0 = c0 / freq   # ≈ 10 cm
k    = 2π / lambda0
eta0 = 376.730313668

# Plate geometry: 4λ × 4λ aperture (sufficient directivity for 30° steering)
Lx = 4 * lambda0
Ly = 4 * lambda0
Nx, Ny = 12, 12       # ~λ/3 element size for good spatial resolution

mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges
Nt   = ntriangles(mesh)

println("\n── Setup ──")
println("  Frequency:  $(freq/1e9) GHz,  λ = $(round(lambda0*100, digits=2)) cm")
println("  Plate:      $(round(Lx*100, digits=1)) cm × $(round(Ly*100, digits=1)) cm")
println("  Mesh:       $(nvertices(mesh)) vertices, $Nt triangles, $N RWG basis functions")

# ─────────────────────────────────────────────────
# 2. EFIE Assembly
# ─────────────────────────────────────────────────
println("\n── Assembling EFIE matrix ──")
Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
cond_info = condition_diagnostics(Z_efie)
println("  Z_efie condition number: $(round(cond_info.cond, sigdigits=3))")

# ─────────────────────────────────────────────────
# 3. Impedance Partition (one patch per triangle)
# ─────────────────────────────────────────────────
partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
println("  Impedance patches: $Nt")

# ─────────────────────────────────────────────────
# 4. Excitation: Normal-incidence plane wave, x-polarized
# ─────────────────────────────────────────────────
k_vec = Vec3(0.0, 0.0, -k)    # propagating -z
E0    = 1.0
pol_inc = Vec3(1.0, 0.0, 0.0) # x-polarized

v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3)

# ─────────────────────────────────────────────────
# 5. Far-field grid & Q matrix for beam steering
# ─────────────────────────────────────────────────
println("\n── Building far-field objective ──")

theta_steer = 30.0 * π / 180  # steer to 30°
phi_steer   = 0.0              # in the xz-plane

grid = make_sph_grid(180, 72)
NΩ = length(grid.w)
println("  Far-field grid: $NΩ directions (1° θ × 5° φ)")

G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)

# Polarization: θ-hat for x-polarized radiation
pol_mat = pol_linear_x(grid)

# Target mask: narrow cone around steered direction
steer_rhat = Vec3(sin(theta_steer) * cos(phi_steer),
                  sin(theta_steer) * sin(phi_steer),
                  cos(theta_steer))

mask = BitVector([begin
    rh = Vec3(grid.rhat[:, q])
    angle = acos(clamp(dot(rh, steer_rhat), -1.0, 1.0))
    angle <= 5.0 * π / 180
end for q in 1:NΩ])
n_target = count(mask)
println("  Target directions (θ_s=$(round(rad2deg(theta_steer)))°, Δθ=5°): $n_target points")

Q_target = build_Q(G_mat, grid, pol_mat; mask=mask)
println("  Q is Hermitian PSD: $(norm(Q_target - Q_target') < 1e-12 * norm(Q_target))")

# Build total-power Q for normalization reference
Q_total = build_Q(G_mat, grid, pol_mat)

# Directivity objective: maximize J = (I†Q_target I)/(I†Q_total I)
# This is the fraction of radiated power landing in the target cone.
# Unlike absolute power, this naturally steers the beam rather than broadening it.
println("  Objective: max P_target / P_total  (directivity ratio)")

# ─────────────────────────────────────────────────
# 6. Reference: PEC solution (no impedance sheet)
# ─────────────────────────────────────────────────
println("\n── Reference: PEC solution ──")
I_pec = Z_efie \ v
E_ff_pec = compute_farfield(G_mat, I_pec, NΩ)
P_target_pec = real(dot(I_pec, Q_target * I_pec))
P_total_pec  = real(dot(I_pec, Q_total * I_pec))
J_pec = P_target_pec / P_total_pec
P_rad_pec = radiated_power(E_ff_pec, grid)
P_in_pec  = input_power(I_pec, v)
println("  J_pec (directivity fraction) = $(round(J_pec*100, digits=2))%")
println("  P_rad = $(P_rad_pec),  P_in = $(P_in_pec)")
println("  Energy ratio P_rad/P_in = $(round(P_rad_pec / P_in_pec, sigdigits=4))")

# ─────────────────────────────────────────────────
# 7. Optimization: steer beam to θ_s
#    maximize projected power ratio J = (I†Q_target I)/(I†Q_total I)
#    with reactive impedance Z_s = iθ
# ─────────────────────────────────────────────────
println("\n── Optimization (L-BFGS, reactive impedance, directivity) ──")
println("  Objective: max projected target-power fraction (two-adjoint quotient rule)")

# Initialize with phase gradient along x (physical starting point for beam steering)
# A linear impedance ramp creates the phase tilt needed to steer to θ_s
tri_centers_init = [triangle_center(mesh, t) for t in 1:Nt]
cx_init = [tc[1] for tc in tri_centers_init]
x_center = (minimum(cx_init) + maximum(cx_init)) / 2
x_halfspan = (maximum(cx_init) - minimum(cx_init)) / 2
theta_init = [300.0 * (cx_t - x_center) / x_halfspan for cx_t in cx_init]
println("  Phase-gradient init: θ ∈ [$(round(minimum(theta_init), digits=1)), $(round(maximum(theta_init), digits=1))] Ω")

# Box constraints: |θ| ≤ 500 Ω (fabrication limits)
theta_bound = 500.0

theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta_init;
    maxiter=300, tol=1e-12,
    alpha0=1e8,
    verbose=true,
    reactive=true,
    lb=-theta_bound, ub=theta_bound
)

# ─────────────────────────────────────────────────
# 8. Post-optimization analysis
# ─────────────────────────────────────────────────
println("\n── Post-optimization ──")

Z_opt  = assemble_full_Z(Z_efie, Mp, theta_opt; reactive=true)
I_opt  = Z_opt \ v
E_ff_opt = compute_farfield(G_mat, I_opt, NΩ)
P_target_opt = real(dot(I_opt, Q_target * I_opt))
P_total_opt  = real(dot(I_opt, Q_total * I_opt))
J_opt  = P_target_opt / P_total_opt
P_rad_opt = radiated_power(E_ff_opt, grid)
P_in_opt  = input_power(I_opt, v)

improvement = J_opt / J_pec
println("  J_pec = $(round(J_pec*100, digits=2))%")
println("  J_opt = $(round(J_opt*100, digits=2))%  ($(round(improvement, sigdigits=3))x over PEC)")
println("  P_rad = $(P_rad_opt),  P_in = $(P_in_opt)")
println("  Energy ratio P_rad/P_in = $(round(P_rad_opt / P_in_opt, sigdigits=4))")

cond_opt = condition_diagnostics(Z_opt)
println("  cond(Z_opt) = $(round(cond_opt.cond, sigdigits=3))")

# Find peak direction in phi=0 cut
ff_power = [real(dot(E_ff_opt[:, q], E_ff_opt[:, q])) for q in 1:NΩ]
dphi_check = 2π / 72
phi0_check = [q for q in 1:NΩ if min(grid.phi[q], 2π - grid.phi[q]) <= dphi_check/2 + 1e-10]
if !isempty(phi0_check)
    _, peak_local = findmax(ff_power[phi0_check])
    peak_q = phi0_check[peak_local]
    println("  Peak direction (phi=0 cut): θ = $(round(rad2deg(grid.theta[peak_q]), digits=1))°")
    target_power = sum(ff_power[q] * grid.w[q] for q in 1:NΩ if mask[q])
    total_power  = sum(ff_power[q] * grid.w[q] for q in 1:NΩ)
    println("  Directivity fraction (target/total): $(round(target_power/total_power*100, digits=1))%")
end

# Gradient check at optimum (two-solve ratio gradient)
lam_t_opt = Z_opt' \ (Q_target * I_opt)
lam_a_opt = Z_opt' \ (Q_total * I_opt)
g_f_opt = gradient_impedance(Mp, I_opt, lam_t_opt; reactive=true)
g_g_opt = gradient_impedance(Mp, I_opt, lam_a_opt; reactive=true)
g_opt = (P_total_opt .* g_f_opt .- P_target_opt .* g_g_opt) ./ (P_total_opt^2)
println("  |g| at optimum = $(norm(g_opt))")

# ─────────────────────────────────────────────────
# 9. Gradient verification at optimized point
# ─────────────────────────────────────────────────
println("\n── Gradient verification (FD at optimum) ──")

function J_of_theta_reactive(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= (1im * theta_vec[p]) .* Mp[p]
    end
    I_t = Z_t \ v
    f_t = real(dot(I_t, Q_target * I_t))
    g_t = real(dot(I_t, Q_total * I_t))
    return f_t / g_t
end

n_check = min(5, Nt)
for p in 1:n_check
    g_fd = fd_grad(J_of_theta_reactive, theta_opt, p; h=1e-5)
    rel_err = abs(g_opt[p] - g_fd) / max(abs(g_opt[p]), 1e-30)
    println("  p=$p: adj=$(g_opt[p])  fd=$g_fd  rel_err=$rel_err")
end

# ─────────────────────────────────────────────────
# 10. Save results as CSVs
# ─────────────────────────────────────────────────
println("\n── Saving data ──")

# Optimization trace
df_trace = DataFrame(
    iter  = [t.iter for t in trace],
    J     = [t.J for t in trace],
    gnorm = [t.gnorm for t in trace]
)
CSV.write(joinpath(DATADIR, "beam_steer_trace.csv"), df_trace)

# Optimized impedance distribution
tri_centers = [triangle_center(mesh, t) for t in 1:Nt]
df_imp = DataFrame(
    patch     = 1:Nt,
    cx        = [tc[1] for tc in tri_centers],
    cy        = [tc[2] for tc in tri_centers],
    theta_init = theta_init,
    theta_opt  = theta_opt
)
CSV.write(joinpath(DATADIR, "beam_steer_impedance.csv"), df_imp)

# Far-field: PEC vs optimized — compute directivity D = 4π|E|²/∫|E|²dΩ
ff_power_pec = [real(dot(E_ff_pec[:, q], E_ff_pec[:, q])) for q in 1:NΩ]
ff_power_opt = [real(dot(E_ff_opt[:, q], E_ff_opt[:, q])) for q in 1:NΩ]

P_sphere_pec = sum(ff_power_pec[q] * grid.w[q] for q in 1:NΩ)
P_sphere_opt = sum(ff_power_opt[q] * grid.w[q] for q in 1:NΩ)
D_pec = [4π * ff_power_pec[q] / P_sphere_pec for q in 1:NΩ]
D_opt = [4π * ff_power_opt[q] / P_sphere_opt for q in 1:NΩ]

println("  Peak directivity PEC: $(round(10*log10(maximum(D_pec)), digits=2)) dBi")
println("  Peak directivity Opt: $(round(10*log10(maximum(D_opt)), digits=2)) dBi")

df_ff = DataFrame(
    theta_deg  = rad2deg.(grid.theta),
    phi_deg    = rad2deg.(grid.phi),
    dir_pec_dBi = 10 .* log10.(max.(D_pec, 1e-30)),
    dir_opt_dBi = 10 .* log10.(max.(D_opt, 1e-30)),
    in_target  = mask
)
CSV.write(joinpath(DATADIR, "beam_steer_farfield.csv"), df_ff)

# Phi=0 cut — select the phi bin closest to 0°
dphi = 2π / 72
phi0_idx = [q for q in 1:NΩ if min(grid.phi[q], 2π - grid.phi[q]) <= dphi/2 + 1e-10]
if !isempty(phi0_idx)
    # Sort by theta for clean plotting
    perm = sortperm(grid.theta[phi0_idx])
    phi0_sorted = phi0_idx[perm]
    df_cut = DataFrame(
        theta_deg    = rad2deg.(grid.theta[phi0_sorted]),
        dir_pec_dBi  = 10 .* log10.(max.(D_pec[phi0_sorted], 1e-30)),
        dir_opt_dBi  = 10 .* log10.(max.(D_opt[phi0_sorted], 1e-30))
    )
    CSV.write(joinpath(DATADIR, "beam_steer_cut_phi0.csv"), df_cut)
end

println("\n" * "="^60)
println("BEAM-STEERING OPTIMIZATION COMPLETE")
println("="^60)
println("CSV files saved to: $DATADIR/")
for f in readdir(DATADIR)
    if startswith(f, "beam_steer")
        println("  $f")
    end
end
