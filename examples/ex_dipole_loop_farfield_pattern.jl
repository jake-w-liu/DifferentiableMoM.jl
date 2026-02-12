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
using PlotlySupply

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
const FIGDIR = joinpath(@__DIR__, "..", "figs")
mkpath(DATADIR)
mkpath(FIGDIR)

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

const PANEL_WIDTH = 900
const PANEL_HEIGHT = 500
const COMBINED_WIDTH = 1020
const COMBINED_HEIGHT = 4200

function _save_pair(fig, basepath::AbstractString; width::Int, height::Int)
    png = basepath * ".png"
    pdf = basepath * ".pdf"
    savefig(fig, png; width = width, height = height)
    savefig(fig, pdf; width = width, height = height)
    return (png_path = png, pdf_path = pdf)
end

function _single_panel(
    title::AbstractString,
    x,
    ys::Vector{<:AbstractVector},
    legends::Vector{<:AbstractString},
    colors::Vector{<:AbstractString};
    dashes::Vector{<:AbstractString}=fill("", length(ys)),
    xlabel::AbstractString="",
    ylabel::AbstractString="",
    width::Int=PANEL_WIDTH,
    height::Int=PANEL_HEIGHT,
)
    sf = subplots(1, 1; sync = false, width = width, height = height, subplot_titles = reshape([title], 1, 1))
    plot_scatter!(
        sf,
        x,
        ys;
        color = colors,
        dash = dashes,
        legend = legends,
        xlabel = xlabel,
        ylabel = ylabel,
    )
    subplot_legends!(sf; position = :topright)
    return sf.plot
end

# Individual panels
p_lin = _single_panel(
    "Dipole/Loop pattern (linear)",
    theta_deg,
    [P_dip_num, P_dip_ana, P_loop_num, P_loop_ana],
    ["Dipole numerical", "Dipole analytical (sin²θ)", "Loop numerical", "Loop analytical (sin²θ)"],
    ["blue", "blue", "red", "red"];
    dashes = ["", "dash", "", "dash"],
    xlabel = "θ (deg)",
    ylabel = "Normalized power",
)

p_db = _single_panel(
    "Dipole/Loop pattern (dB)",
    theta_deg,
    [dB_dip_num, dB_dip_ana, dB_loop_num, dB_loop_ana],
    ["Dipole numerical", "Dipole analytical", "Loop numerical", "Loop analytical"],
    ["blue", "blue", "red", "red"];
    dashes = ["", "dash", "", "dash"],
    xlabel = "θ (deg)",
    ylabel = "Normalized power (dB)",
)

p_err = _single_panel(
    "Pattern error curves",
    theta_deg,
    [err_dip, err_loop],
    ["Dipole error", "Loop error"],
    ["blue", "red"];
    xlabel = "θ (deg)",
    ylabel = "Numerical - analytical",
)
add_hline!(p_err, 0.0; line_color = "black", line_dash = "dot", line_width = 1.2)

p_pol = _single_panel(
    "Polarization components (φ=0 cut)",
    theta_deg,
    [
        [to_dB(x) for x in P_dip_theta],
        [to_dB(x) for x in P_dip_phi],
        [to_dB(x) for x in P_loop_phi],
        [to_dB(x) for x in P_loop_theta],
    ],
    ["Dipole |Eθ|²", "Dipole |Eϕ|²", "Loop |Eϕ|²", "Loop |Eθ|²"],
    ["blue", "blue", "red", "red"];
    dashes = ["", "dash", "", "dash"],
    xlabel = "θ (deg)",
    ylabel = "Component power (dB, normalized)",
)

p_xpd = _single_panel(
    "Cross-polarization ratio (φ=0 cut)",
    theta_deg,
    [xpd_dip_db, xpd_loop_db],
    ["Dipole cross/co (Eϕ/Eθ)", "Loop cross/co (Eθ/Eϕ)"],
    ["blue", "red"];
    xlabel = "θ (deg)",
    ylabel = "Cross-pol ratio (dB)",
)

p_phase = _single_panel(
    "Phase consistency on co-pol components (φ=0 cut)",
    theta_deg[phase_valid],
    [phase_ratio_valid],
    ["arg(Eϕ(loop)/Eθ(dipole))"],
    ["purple"];
    xlabel = "θ (deg)",
    ylabel = "Phase difference (deg)",
)
addtraces!(
    p_phase,
    scatter(
        x = theta_deg[phase_valid],
        y = fill(90.0, length(theta_deg[phase_valid])),
        mode = "lines",
        name = "+90°",
        line = attr(color = "black", width = 2, dash = "dash"),
    ),
)
addtraces!(
    p_phase,
    scatter(
        x = theta_deg[phase_valid],
        y = fill(-90.0, length(theta_deg[phase_valid])),
        mode = "lines",
        name = "-90°",
        line = attr(color = "black", width = 2, dash = "dash"),
    ),
)
set_legend!(p_phase; position = :topright)

lin_out = _save_pair(p_lin, joinpath(FIGDIR, "dipole_loop_pattern_linear"); width = PANEL_WIDTH, height = PANEL_HEIGHT)
db_out = _save_pair(p_db, joinpath(FIGDIR, "dipole_loop_pattern_db"); width = PANEL_WIDTH, height = PANEL_HEIGHT)
err_out = _save_pair(p_err, joinpath(FIGDIR, "dipole_loop_pattern_error"); width = PANEL_WIDTH, height = PANEL_HEIGHT)
pol_out = _save_pair(p_pol, joinpath(FIGDIR, "dipole_loop_pattern_polarization"); width = PANEL_WIDTH, height = PANEL_HEIGHT)
xpd_out = _save_pair(p_xpd, joinpath(FIGDIR, "dipole_loop_pattern_crosspol"); width = PANEL_WIDTH, height = PANEL_HEIGHT)
phase_out = _save_pair(p_phase, joinpath(FIGDIR, "dipole_loop_pattern_phase"); width = PANEL_WIDTH, height = PANEL_HEIGHT)

# Combined validation panel
combined_titles = reshape(
    [
        "Dipole/Loop pattern (linear)",
        "Dipole/Loop pattern (dB)",
        "Pattern error curves",
        "Polarization components (φ=0 cut)",
        "Cross-polarization ratio (φ=0 cut)",
        "Phase consistency on co-pol components (φ=0 cut)",
    ],
    6,
    1,
)
sf_all = subplots(6, 1; sync = false, width = COMBINED_WIDTH, height = COMBINED_HEIGHT, subplot_titles = combined_titles)

plot_scatter!(
    sf_all,
    theta_deg,
    [P_dip_num, P_dip_ana, P_loop_num, P_loop_ana];
    color = ["blue", "blue", "red", "red"],
    dash = ["", "dash", "", "dash"],
    legend = ["Dipole numerical", "Dipole analytical (sin²θ)", "Loop numerical", "Loop analytical (sin²θ)"],
    xlabel = "θ (deg)",
    ylabel = "Normalized power",
)

subplot!(sf_all, 2, 1)
plot_scatter!(
    sf_all,
    theta_deg,
    [dB_dip_num, dB_dip_ana, dB_loop_num, dB_loop_ana];
    color = ["blue", "blue", "red", "red"],
    dash = ["", "dash", "", "dash"],
    legend = ["Dipole numerical", "Dipole analytical", "Loop numerical", "Loop analytical"],
    xlabel = "θ (deg)",
    ylabel = "Normalized power (dB)",
)

subplot!(sf_all, 3, 1)
plot_scatter!(
    sf_all,
    theta_deg,
    [err_dip, err_loop];
    color = ["blue", "red"],
    legend = ["Dipole error", "Loop error"],
    xlabel = "θ (deg)",
    ylabel = "Numerical - analytical",
)
add_hline!(sf_all.plot, 0.0; row = 3, col = 1, line_color = "black", line_dash = "dot", line_width = 1.2)

subplot!(sf_all, 4, 1)
plot_scatter!(
    sf_all,
    theta_deg,
    [
        [to_dB(x) for x in P_dip_theta],
        [to_dB(x) for x in P_dip_phi],
        [to_dB(x) for x in P_loop_phi],
        [to_dB(x) for x in P_loop_theta],
    ];
    color = ["blue", "blue", "red", "red"],
    dash = ["", "dash", "", "dash"],
    legend = ["Dipole |Eθ|²", "Dipole |Eϕ|²", "Loop |Eϕ|²", "Loop |Eθ|²"],
    xlabel = "θ (deg)",
    ylabel = "Component power (dB, normalized)",
)

subplot!(sf_all, 5, 1)
plot_scatter!(
    sf_all,
    theta_deg,
    [xpd_dip_db, xpd_loop_db];
    color = ["blue", "red"],
    legend = ["Dipole cross/co (Eϕ/Eθ)", "Loop cross/co (Eθ/Eϕ)"],
    xlabel = "θ (deg)",
    ylabel = "Cross-pol ratio (dB)",
)

subplot!(sf_all, 6, 1)
plot_scatter!(
    sf_all,
    theta_deg[phase_valid],
    phase_ratio_valid;
    color = "purple",
    legend = "arg(Eϕ(loop)/Eθ(dipole))",
    xlabel = "θ (deg)",
    ylabel = "Phase difference (deg)",
)
addtraces!(
    sf_all,
    scatter(
        x = theta_deg[phase_valid],
        y = fill(90.0, length(theta_deg[phase_valid])),
        mode = "lines",
        name = "+90°",
        line = attr(color = "black", width = 2, dash = "dash"),
    ),
    scatter(
        x = theta_deg[phase_valid],
        y = fill(-90.0, length(theta_deg[phase_valid])),
        mode = "lines",
        name = "-90°",
        line = attr(color = "black", width = 2, dash = "dash"),
    );
    row = 6,
    col = 1,
)

subplot_legends!(sf_all; position = :topright)
combined_out = _save_pair(sf_all.plot, joinpath(FIGDIR, "dipole_loop_pattern_validation"); width = COMBINED_WIDTH, height = COMBINED_HEIGHT)

println("\nOutputs:")
println("  Pattern CSV:  $csv_path")
println("  Summary CSV:  $summary_path")
println("  Linear plot:  $(lin_out.png_path)")
println("  dB plot:      $(db_out.png_path)")
println("  Error plot:   $(err_out.png_path)")
println("  Pol plot:     $(pol_out.png_path)")
println("  X-pol plot:   $(xpd_out.png_path)")
println("  Phase plot:   $(phase_out.png_path)")
println("  Combined:     $(combined_out.png_path)")
println("\nDone.")
