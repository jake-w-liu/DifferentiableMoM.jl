# ex_dipole_loop_farfield_pattern.jl — dipole/loop far-field pattern validation
#
# Produces analytical-vs-numerical pattern comparisons (linear + dB),
# error curves, and CSV outputs for:
#   1) z-directed electric dipole
#   2) z-directed small loop (equivalent magnetic dipole)
#
# Run:
#   julia --project=. examples/ex_dipole_loop_farfield_pattern.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using Statistics
using CSV
using DataFrames
using Plots

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

const c0 = 299792458.0
const ϵ0 = 8.854187817e-12
const μ0 = 4π * 1e-7
const η0 = sqrt(μ0 / ϵ0)

function to_dB(x; floor=1e-12)
    return 10log10(max(x, floor))
end

println("="^60)
println("Dipole/Loop Far-field Pattern Validation")
println("="^60)

# Problem setup
freq = 1.0e9
λ0 = c0 / freq
k = 2π / λ0
Rfar = 50 * λ0

# Sources
dipole = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, 1e-12 + 0im),  # z-directed electric dipole moment (C·m)
    Vec3(0.0, 0.0, 1.0),
    :electric,
    freq
)
loop = make_loop(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.01,                # 1 cm loop radius
    1.0 + 0im,           # 1 A loop current
    freq
)

# Angular sampling (phi=0 cut)
theta_deg = collect(0.0:1.0:180.0)
theta = deg2rad.(theta_deg)
phi = 0.0

P_dip_num = zeros(Float64, length(theta))
P_loop_num = zeros(Float64, length(theta))
P_dip_ana = sin.(theta).^2
P_loop_ana = copy(P_dip_ana)  # small loop has same sin^2 shape
P_dip_theta = zeros(Float64, length(theta))
P_dip_phi = zeros(Float64, length(theta))
P_loop_theta = zeros(Float64, length(theta))
P_loop_phi = zeros(Float64, length(theta))
E_dip_theta_cmp = zeros(ComplexF64, length(theta))
E_loop_phi_cmp = zeros(ComplexF64, length(theta))

for (i, th) in enumerate(theta)
    rhat = Vec3(sin(th) * cos(phi), sin(th) * sin(phi), cos(th))
    r = Rfar * rhat
    e_theta = Vec3(cos(th) * cos(phi), cos(th) * sin(phi), -sin(th))
    e_phi = Vec3(-sin(phi), cos(phi), 0.0)

    E_dip = DifferentiableMoM.dipole_incident_field(r, dipole)
    E_loop = DifferentiableMoM.loop_incident_field(r, loop)
    E_dip_theta = dot(E_dip, e_theta)
    E_dip_phi = dot(E_dip, e_phi)
    E_loop_theta = dot(E_loop, e_theta)
    E_loop_phi = dot(E_loop, e_phi)

    P_dip_num[i] = norm(E_dip)^2
    P_loop_num[i] = norm(E_loop)^2
    P_dip_theta[i] = abs2(E_dip_theta)
    P_dip_phi[i] = abs2(E_dip_phi)
    P_loop_theta[i] = abs2(E_loop_theta)
    P_loop_phi[i] = abs2(E_loop_phi)
    E_dip_theta_cmp[i] = E_dip_theta
    E_loop_phi_cmp[i] = E_loop_phi
end

# Normalize for pattern-shape comparison
scale_dip = maximum(P_dip_num)
scale_loop = maximum(P_loop_num)
P_dip_num ./= scale_dip
P_loop_num ./= scale_loop
P_dip_ana ./= maximum(P_dip_ana)
P_loop_ana ./= maximum(P_loop_ana)
P_dip_theta ./= scale_dip
P_dip_phi ./= scale_dip
P_loop_theta ./= scale_loop
P_loop_phi ./= scale_loop

err_dip = P_dip_num .- P_dip_ana
err_loop = P_loop_num .- P_loop_ana

rmse_dip = sqrt(mean(abs2, err_dip))
rmse_loop = sqrt(mean(abs2, err_loop))
maxabs_dip = maximum(abs.(err_dip))
maxabs_loop = maximum(abs.(err_loop))

dB_dip_num = [to_dB(x) for x in P_dip_num]
dB_dip_ana = [to_dB(x) for x in P_dip_ana]
dB_loop_num = [to_dB(x) for x in P_loop_num]
dB_loop_ana = [to_dB(x) for x in P_loop_ana]
err_dB_dip = dB_dip_num .- dB_dip_ana
err_dB_loop = dB_loop_num .- dB_loop_ana
xpd_dip_db = [10log10(max(P_dip_phi[i], 1e-30) / max(P_dip_theta[i], 1e-30)) for i in eachindex(theta)]
xpd_loop_db = [10log10(max(P_loop_theta[i], 1e-30) / max(P_loop_phi[i], 1e-30)) for i in eachindex(theta)]

# Full-angle polarization check on a coarse (θ,φ) grid
theta_pol_deg = collect(1.0:2.0:179.0)
phi_pol_deg = collect(0.0:10.0:350.0)
theta_pol = deg2rad.(theta_pol_deg)
phi_pol = deg2rad.(phi_pol_deg)
crossfrac_dip = Float64[]
crossfrac_loop = Float64[]
for th in theta_pol, ph in phi_pol
    rhat = Vec3(sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
    r = Rfar * rhat
    e_theta = Vec3(cos(th) * cos(ph), cos(th) * sin(ph), -sin(th))
    e_phi = Vec3(-sin(ph), cos(ph), 0.0)
    E_d = DifferentiableMoM.dipole_incident_field(r, dipole)
    E_l = DifferentiableMoM.loop_incident_field(r, loop)
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

# Phase-consistency check on the φ=0 cut:
# For z-directed electric dipole vs z-directed small loop, the co-pol fields
# differ by approximately ±90° in the far zone.
wrap_to_pi(x) = atan(sin(x), cos(x))
phase_ratio_deg = fill(NaN, length(theta))
phase_err_pm90_deg = fill(NaN, length(theta))
phase_valid = falses(length(theta))
amp_floor = 1e-12 * max(maximum(abs.(E_dip_theta_cmp)), maximum(abs.(E_loop_phi_cmp)))
for i in eachindex(theta)
    if abs(E_dip_theta_cmp[i]) > amp_floor && abs(E_loop_phi_cmp[i]) > amp_floor
        Δϕ = angle(E_loop_phi_cmp[i] / E_dip_theta_cmp[i])
        err_pm90 = min(abs(wrap_to_pi(Δϕ - π / 2)), abs(wrap_to_pi(Δϕ + π / 2)))
        phase_ratio_deg[i] = rad2deg(Δϕ)
        phase_err_pm90_deg[i] = rad2deg(err_pm90)
        phase_valid[i] = true
    end
end
phase_ratio_valid = phase_ratio_deg[phase_valid]
phase_err_valid = phase_err_pm90_deg[phase_valid]
phase_mean_deg = mean(phase_ratio_valid)
phase_std_deg = std(phase_ratio_valid)
phase_max_err_pm90_deg = maximum(phase_err_valid)

println("\nPattern metrics (normalized linear scale):")
println("  Dipole RMSE:      $rmse_dip")
println("  Dipole max |err|: $maxabs_dip")
println("  Loop RMSE:        $rmse_loop")
println("  Loop max |err|:   $maxabs_loop")
println("  Max dipole cross-pol fraction: $max_crossfrac_dip")
println("  Max loop cross-pol fraction:   $max_crossfrac_loop")
println("  Phase mean (deg): $phase_mean_deg")
println("  Phase std (deg):  $phase_std_deg")
println("  Phase max err to ±90° (deg): $phase_max_err_pm90_deg")

# Save CSV
df = DataFrame(
    theta_deg = theta_deg,
    P_dip_num = P_dip_num,
    P_dip_ana = P_dip_ana,
    P_loop_num = P_loop_num,
    P_loop_ana = P_loop_ana,
    err_dip = err_dip,
    err_loop = err_loop,
    dB_dip_num = dB_dip_num,
    dB_dip_ana = dB_dip_ana,
    dB_loop_num = dB_loop_num,
    dB_loop_ana = dB_loop_ana,
    err_dB_dip = err_dB_dip,
    err_dB_loop = err_dB_loop,
    P_dip_theta = P_dip_theta,
    P_dip_phi = P_dip_phi,
    P_loop_theta = P_loop_theta,
    P_loop_phi = P_loop_phi,
    xpd_dip_db = xpd_dip_db,
    xpd_loop_db = xpd_loop_db,
    phase_ratio_deg = phase_ratio_deg,
    phase_err_pm90_deg = phase_err_pm90_deg,
)
csv_path = joinpath(DATADIR, "dipole_loop_pattern_phi0.csv")
CSV.write(csv_path, df)

df_summary = DataFrame(
    metric = [
        "rmse_dip",
        "maxabs_dip",
        "rmse_loop",
        "maxabs_loop",
        "max_crossfrac_dip",
        "max_crossfrac_loop",
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
        max_crossfrac_dip,
        max_crossfrac_loop,
        phase_mean_deg,
        phase_std_deg,
        phase_max_err_pm90_deg,
        Rfar / λ0,
        freq / 1e9,
    ],
)
summary_path = joinpath(DATADIR, "dipole_loop_pattern_summary.csv")
CSV.write(summary_path, df_summary)

# Plot: linear
p_lin = plot(theta_deg, P_dip_num;
    lw=2, color=:blue, label="Dipole numerical",
    xlabel="θ (deg)", ylabel="Normalized power", title="Dipole/Loop pattern (linear)")
plot!(p_lin, theta_deg, P_dip_ana; lw=2, ls=:dash, color=:blue, label="Dipole analytical (sin²θ)")
plot!(p_lin, theta_deg, P_loop_num; lw=2, color=:red, label="Loop numerical")
plot!(p_lin, theta_deg, P_loop_ana; lw=2, ls=:dash, color=:red, label="Loop analytical (sin²θ)")

# Plot: dB
p_db = plot(theta_deg, dB_dip_num;
    lw=2, color=:blue, label="Dipole numerical",
    xlabel="θ (deg)", ylabel="Normalized power (dB)", title="Dipole/Loop pattern (dB)")
plot!(p_db, theta_deg, dB_dip_ana; lw=2, ls=:dash, color=:blue, label="Dipole analytical")
plot!(p_db, theta_deg, dB_loop_num; lw=2, color=:red, label="Loop numerical")
plot!(p_db, theta_deg, dB_loop_ana; lw=2, ls=:dash, color=:red, label="Loop analytical")

# Plot: error curves
p_err = plot(theta_deg, err_dip;
    lw=2, color=:blue, label="Dipole error",
    xlabel="θ (deg)", ylabel="Numerical - analytical", title="Pattern error curves")
plot!(p_err, theta_deg, err_loop; lw=2, color=:red, label="Loop error")
hline!(p_err, [0.0]; ls=:dot, color=:black, label=nothing)

# Plot: polarization components and cross-pol ratio
p_pol = plot(theta_deg, [to_dB(x) for x in P_dip_theta];
    lw=2, color=:blue, label="Dipole |Eθ|²",
    xlabel="θ (deg)", ylabel="Component power (dB, normalized)",
    title="Polarization components (φ=0 cut)")
plot!(p_pol, theta_deg, [to_dB(x) for x in P_dip_phi]; lw=2, ls=:dash, color=:blue, label="Dipole |Eϕ|²")
plot!(p_pol, theta_deg, [to_dB(x) for x in P_loop_phi]; lw=2, color=:red, label="Loop |Eϕ|²")
plot!(p_pol, theta_deg, [to_dB(x) for x in P_loop_theta]; lw=2, ls=:dash, color=:red, label="Loop |Eθ|²")

p_xpd = plot(theta_deg, xpd_dip_db;
    lw=2, color=:blue, label="Dipole cross/co (Eϕ/Eθ)",
    xlabel="θ (deg)", ylabel="Cross-pol ratio (dB)", title="Cross-polarization ratio (φ=0 cut)")
plot!(p_xpd, theta_deg, xpd_loop_db; lw=2, color=:red, label="Loop cross/co (Eθ/Eϕ)")

p_phase = plot(theta_deg[phase_valid], phase_ratio_valid;
    lw=2, color=:purple, label="arg(Eϕ(loop)/Eθ(dipole))",
    xlabel="θ (deg)", ylabel="Phase difference (deg)",
    title="Phase consistency on co-pol components (φ=0 cut)")
hline!(p_phase, [90.0, -90.0]; lw=1.5, ls=:dash, color=:black, label=["+90°" "-90°"])

plot_lin_path = joinpath(DATADIR, "dipole_loop_pattern_linear.png")
plot_db_path = joinpath(DATADIR, "dipole_loop_pattern_db.png")
plot_err_path = joinpath(DATADIR, "dipole_loop_pattern_error.png")
plot_pol_path = joinpath(DATADIR, "dipole_loop_pattern_polarization.png")
plot_xpd_path = joinpath(DATADIR, "dipole_loop_pattern_crosspol.png")
plot_phase_path = joinpath(DATADIR, "dipole_loop_pattern_phase.png")
plot_combined_path = joinpath(DATADIR, "dipole_loop_pattern_validation.png")

savefig(p_lin, plot_lin_path)
savefig(p_db, plot_db_path)
savefig(p_err, plot_err_path)
savefig(p_pol, plot_pol_path)
savefig(p_xpd, plot_xpd_path)
savefig(p_phase, plot_phase_path)
savefig(plot(p_lin, p_db, p_err, p_pol, p_xpd, p_phase; layout=(6, 1), size=(900, 2100)), plot_combined_path)

println("\nOutputs:")
println("  Pattern CSV:  $csv_path")
println("  Summary CSV:  $summary_path")
println("  Linear plot:  $plot_lin_path")
println("  dB plot:      $plot_db_path")
println("  Error plot:   $plot_err_path")
println("  Pol plot:     $plot_pol_path")
println("  X-pol plot:   $plot_xpd_path")
println("  Phase plot:   $plot_phase_path")
println("  Combined:     $plot_combined_path")
println("\nDone.")
