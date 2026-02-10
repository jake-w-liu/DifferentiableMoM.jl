# ex_radiationpatterns_adapter.jl
#
# Direct adapter example:
#   CSV (Eθ/Eϕ on θ/ϕ grid) -> RadiationPatterns.Pattern objects -> PatternFeedExcitation
#
# Demonstrates both constructor paths:
#   1) make_pattern_feed(Etheta_pattern, Ephi_pattern, ...)
#   2) make_pattern_feed(theta, phi, Ftheta, Fphi, ...)
#
# Run:
#   julia --project=. examples/ex_radiationpatterns_adapter.jl
#   julia --project=. examples/ex_radiationpatterns_adapter.jl examples/antenna_pattern.csv 3.0

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using Statistics
using CSV
using DataFrames
using Plots

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

include(joinpath(@__DIR__, "pattern_import_utils.jl"))
using .PatternImportUtils

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

to_dB(x; floor=1e-30) = 10 * log10(max(x, floor))

println("="^68)
println("RadiationPatterns adapter example: CSV -> Pattern -> PatternFeed")
println("="^68)

csv_path = isempty(ARGS) ? joinpath(@__DIR__, "antenna_pattern.csv") : ARGS[1]
freq = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) * 1e9 : 3.0e9

raw = load_pattern_csv(csv_path; drop_phi_endpoint=true)
println("Loaded CSV: $csv_path")
println("  Nθ = $(length(raw.theta_deg)), Nϕ = $(length(raw.phi_deg))")
println("  Dropped φ endpoint: $(raw.dropped_phi_endpoint)")

pat_objs = maybe_make_radiationpatterns_patterns(raw.theta_deg, raw.phi_deg, raw.Ftheta, raw.Fphi)
println("Pattern backend: $(pat_objs.backend)")
println("  $(pat_objs.message)")

phase_center = Vec3(0.0, 0.0, 0.0)
pat_obj = make_pattern_feed(
    pat_objs.Etheta_pattern,
    pat_objs.Ephi_pattern,
    freq;
    angles_in_degrees=true,
    phase_center=phase_center,
    convention=:exp_plus_iwt,
)
pat_raw = make_pattern_feed(
    raw.theta_deg,
    raw.phi_deg,
    raw.Ftheta,
    raw.Fphi,
    freq;
    angles_in_degrees=true,
    phase_center=phase_center,
    convention=:exp_plus_iwt,
)

λ0 = 299792458.0 / freq
R = 80 * λ0
θ_eval_deg = collect(0.0:0.5:180.0)
θ_eval = deg2rad.(θ_eval_deg)
ϕ_eval = 0.0

Eθ_obj = zeros(ComplexF64, length(θ_eval))
Eϕ_obj = zeros(ComplexF64, length(θ_eval))
Eθ_raw = zeros(ComplexF64, length(θ_eval))
Eϕ_raw = zeros(ComplexF64, length(θ_eval))

for (i, θ) in enumerate(θ_eval)
    rhat = Vec3(sin(θ) * cos(ϕ_eval), sin(θ) * sin(ϕ_eval), cos(θ))
    r = phase_center + R * rhat
    eθ = Vec3(cos(θ) * cos(ϕ_eval), cos(θ) * sin(ϕ_eval), -sin(θ))
    eϕ = Vec3(-sin(ϕ_eval), cos(ϕ_eval), 0.0)

    E_obj = pattern_feed_field(r, pat_obj)
    E_raw = pattern_feed_field(r, pat_raw)
    Eθ_obj[i] = dot(E_obj, eθ)
    Eϕ_obj[i] = dot(E_obj, eϕ)
    Eθ_raw[i] = dot(E_raw, eθ)
    Eϕ_raw[i] = dot(E_raw, eϕ)
end

P_obj = abs2.(Eθ_obj) .+ abs2.(Eϕ_obj)
P_raw = abs2.(Eθ_raw) .+ abs2.(Eϕ_raw)
P_obj_norm = P_obj ./ max(maximum(P_obj), 1e-30)
P_raw_norm = P_raw ./ max(maximum(P_raw), 1e-30)

abs_err_power = abs.(P_obj_norm .- P_raw_norm)
rmse_power = sqrt(mean(abs2, P_obj_norm .- P_raw_norm))
max_err_power = maximum(abs_err_power)

vec_num = vcat(Eθ_obj, Eϕ_obj)
vec_ref = vcat(Eθ_raw, Eϕ_raw)
rel_field_l2 = norm(vec_num - vec_ref) / max(norm(vec_ref), 1e-30)

println("Adapter metrics:")
println("  Relative field L2 (pattern-obj vs raw-array): $rel_field_l2")
println("  Power RMSE (normalized):                      $rmse_power")
println("  Power max |err|:                              $max_err_power")

df_cut = DataFrame(
    theta_deg = θ_eval_deg,
    P_pattern_obj = P_obj_norm,
    P_pattern_raw = P_raw_norm,
    abs_err_power = abs_err_power,
    dB_pattern_obj = [to_dB(x) for x in P_obj_norm],
    dB_pattern_raw = [to_dB(x) for x in P_raw_norm],
)
csv_cut = joinpath(DATADIR, "radiationpatterns_adapter_cut_phi0.csv")
CSV.write(csv_cut, df_cut)

df_summary = DataFrame(
    metric = [
        "backend",
        "frequency_GHz",
        "Ntheta",
        "Nphi",
        "relative_field_l2",
        "rmse_power_norm",
        "maxabs_power_norm",
    ],
    value = [
        String(pat_objs.backend),
        freq / 1e9,
        length(raw.theta_deg),
        length(raw.phi_deg),
        rel_field_l2,
        rmse_power,
        max_err_power,
    ],
)
csv_summary = joinpath(DATADIR, "radiationpatterns_adapter_summary.csv")
CSV.write(csv_summary, df_summary)

p_db = plot(
    θ_eval_deg,
    df_cut.dB_pattern_obj;
    lw=2,
    color=:blue,
    label="Pattern-object path",
    xlabel="θ (deg)",
    ylabel="Normalized power (dB)",
    title="Adapter check: φ=0° cut",
)
plot!(p_db, θ_eval_deg, df_cut.dB_pattern_raw; lw=2, ls=:dash, color=:black, label="Raw-array path")

p_err = plot(
    θ_eval_deg,
    df_cut.abs_err_power;
    lw=2,
    color=:red,
    label="|Δ power|",
    xlabel="θ (deg)",
    ylabel="Absolute error",
    title="Normalized power error",
)
hline!(p_err, [0.0], color=:gray, ls=:dot, label=nothing)

p_phase = plot(
    θ_eval_deg,
    rad2deg.(angle.(Eθ_obj ./ max.(abs.(Eθ_obj), 1e-30)));
    lw=2,
    color=:purple,
    label="∠Eθ (pattern-object)",
    xlabel="θ (deg)",
    ylabel="deg",
    title="Co-pol phase (pattern-object path)",
)

fig = plot(p_db, p_err, p_phase; layout=(3, 1), size=(900, 1000))
png_path = joinpath(DATADIR, "radiationpatterns_adapter_check.png")
savefig(fig, png_path)

println("Saved:")
println("  $csv_cut")
println("  $csv_summary")
println("  $png_path")
