# ex_pattern_feed_dipole_validation.jl
#
# Validate PatternFeedExcitation using analytically generated dipole
# far-field coefficients.
#
# Outputs:
#   - data/pattern_feed_dipole_phi0.csv
#   - data/pattern_feed_dipole_summary.csv
#   - data/pattern_feed_dipole_validation.png
#
# Run:
#   julia --project=. examples/ex_pattern_feed_dipole_validation.jl

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

const C0 = 299792458.0
const EPS0 = 8.854187817e-12

to_dB(x; floor=1e-30) = 10 * log10(max(x, floor))

println("="^64)
println("Pattern-feed validation using analytical dipole coefficients")
println("="^64)

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
freq = 1.0e9
k = 2π * freq / C0
λ0 = C0 / freq
Rfar = 80 * λ0

pz = 1e-12 + 0im
dip = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, pz),
    Vec3(0.0, 0.0, 1.0),
    :electric,
    freq,
)

# Pattern grid used as "imported" data
theta_pat_deg = collect(0.0:2.0:180.0)
phi_pat_deg = collect(0.0:5.0:355.0)
pat = make_analytic_dipole_pattern_feed(
    dip,
    theta_pat_deg,
    phi_pat_deg;
    angles_in_degrees=true,
)

# Evaluation cut (finer than pattern grid to exercise interpolation)
theta_eval_deg = collect(0.0:0.5:180.0)
theta_eval = deg2rad.(theta_eval_deg)
phi_eval = 0.0

Eθ_num = zeros(ComplexF64, length(theta_eval))
Eφ_num = zeros(ComplexF64, length(theta_eval))
Eθ_ana = zeros(ComplexF64, length(theta_eval))
Eφ_ana = zeros(ComplexF64, length(theta_eval))

for (i, θ) in enumerate(theta_eval)
    rhat = Vec3(sin(θ) * cos(phi_eval), sin(θ) * sin(phi_eval), cos(θ))
    r = Rfar * rhat
    eθ = Vec3(cos(θ) * cos(phi_eval), cos(θ) * sin(phi_eval), -sin(θ))
    eϕ = Vec3(-sin(phi_eval), cos(phi_eval), 0.0)

    E_num = pattern_feed_field(r, pat)

    # Analytical far-field of z-directed electric dipole (exp(+iωt))
    Eθ_ref = (k^2 * pz / (4π * EPS0)) * sin(θ) * exp(-1im * k * Rfar) / Rfar
    Eϕ_ref = 0.0 + 0im
    E_ref = Eθ_ref * eθ + Eϕ_ref * eϕ

    Eθ_num[i] = dot(E_num, eθ)
    Eφ_num[i] = dot(E_num, eϕ)
    Eθ_ana[i] = dot(E_ref, eθ)
    Eφ_ana[i] = dot(E_ref, eϕ)
end

P_num = abs2.(Eθ_num) .+ abs2.(Eφ_num)
P_ana = abs2.(Eθ_ana) .+ abs2.(Eφ_ana)
P_num ./= maximum(P_num)
P_ana ./= maximum(P_ana)

err_lin = P_num .- P_ana
rmse = sqrt(mean(abs2, err_lin))
maxabs = maximum(abs.(err_lin))

db_num = [to_dB(x) for x in P_num]
db_ana = [to_dB(x) for x in P_ana]
err_db = db_num .- db_ana

# Co-polar phase error, excluding near-null samples.
phase_err_deg = fill(NaN, length(theta_eval))
amp_floor = 1e-10 * maximum(abs.(Eθ_ana))
for i in eachindex(theta_eval)
    if abs(Eθ_ana[i]) > amp_floor && abs(Eθ_num[i]) > amp_floor
        phase_err_deg[i] = rad2deg(angle(Eθ_num[i] / Eθ_ana[i]))
    end
end
phase_valid = phase_err_deg[.!isnan.(phase_err_deg)]
phase_offset = rad2deg(angle(sum(exp.(1im .* deg2rad.(phase_valid)))))
phase_residual = [rad2deg(atan(sin(deg2rad(x - phase_offset)), cos(deg2rad(x - phase_offset)))) for x in phase_valid]
phase_mean = mean(phase_valid)
phase_std = std(phase_valid)
phase_max = maximum(abs.(phase_valid))
phase_resid_std = std(phase_residual)
phase_resid_max = maximum(abs.(phase_residual))

max_crossfrac = maximum(abs.(Eφ_num) ./ max.(sqrt.(abs2.(Eθ_num) .+ abs2.(Eφ_num)), 1e-30))

println("RMSE (linear normalized power): $rmse")
println("Max |error| (linear):           $maxabs")
println("Max cross-pol fraction:         $max_crossfrac")
println("Phase mean (raw, deg):          $phase_mean")
println("Phase std (raw, deg):           $phase_std")
println("Phase max |raw| (deg):          $phase_max")
println("Phase offset (circular mean):   $phase_offset")
println("Phase residual std (deg):       $phase_resid_std")
println("Phase residual max |err| (deg): $phase_resid_max")

df = DataFrame(
    theta_deg = theta_eval_deg,
    P_num = P_num,
    P_ana = P_ana,
    err_lin = err_lin,
    dB_num = db_num,
    dB_ana = db_ana,
    err_db = err_db,
    Etheta_num_real = real.(Eθ_num),
    Etheta_num_imag = imag.(Eθ_num),
    Etheta_ana_real = real.(Eθ_ana),
    Etheta_ana_imag = imag.(Eθ_ana),
    Ephi_num_abs = abs.(Eφ_num),
    phase_err_deg = phase_err_deg,
)
CSV.write(joinpath(DATADIR, "pattern_feed_dipole_phi0.csv"), df)

df_summary = DataFrame(
    metric = [
        "rmse_lin",
        "maxabs_lin",
        "max_crosspol_frac",
        "phase_mean_raw_deg",
        "phase_std_raw_deg",
        "phase_max_abs_raw_deg",
        "phase_offset_deg",
        "phase_residual_std_deg",
        "phase_residual_max_abs_deg",
        "Rfar_over_lambda",
        "freq_GHz",
        "theta_pattern_step_deg",
        "phi_pattern_step_deg",
    ],
    value = [
        rmse,
        maxabs,
        max_crossfrac,
        phase_mean,
        phase_std,
        phase_max,
        phase_offset,
        phase_resid_std,
        phase_resid_max,
        Rfar / λ0,
        freq / 1e9,
        2.0,
        5.0,
    ],
)
CSV.write(joinpath(DATADIR, "pattern_feed_dipole_summary.csv"), df_summary)

p_lin = plot(theta_eval_deg, P_num;
    lw=2, color=:blue, label="Pattern-feed",
    xlabel="θ (deg)", ylabel="Normalized power",
    title="Dipole cut (φ=0): linear")
plot!(p_lin, theta_eval_deg, P_ana; lw=2, ls=:dash, color=:black, label="Analytical")

p_db = plot(theta_eval_deg, db_num;
    lw=2, color=:blue, label="Pattern-feed",
    xlabel="θ (deg)", ylabel="Normalized power (dB)",
    title="Dipole cut (φ=0): dB")
plot!(p_db, theta_eval_deg, db_ana; lw=2, ls=:dash, color=:black, label="Analytical")

p_err = plot(theta_eval_deg, err_lin;
    lw=2, color=:red, label="Linear error",
    xlabel="θ (deg)", ylabel="Error",
    title="Pattern-feed minus analytical")
hline!(p_err, [0.0], color=:gray, ls=:dot, label=nothing)

p_phase = plot(theta_eval_deg, phase_err_deg;
    lw=2, color=:purple, label="Phase error",
    xlabel="θ (deg)", ylabel="deg",
    title="Co-pol phase error")
hline!(p_phase, [0.0], color=:gray, ls=:dot, label=nothing)

fig = plot(p_lin, p_db, p_err, p_phase; layout=(4, 1), size=(900, 1200))
fig_path = joinpath(DATADIR, "pattern_feed_dipole_validation.png")
savefig(fig, fig_path)

println("Saved:")
println("  $(joinpath(DATADIR, "pattern_feed_dipole_phi0.csv"))")
println("  $(joinpath(DATADIR, "pattern_feed_dipole_summary.csv"))")
println("  $fig_path")
