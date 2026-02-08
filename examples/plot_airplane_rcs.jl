# plot_airplane_rcs.jl — heuristic plots for airplane RCS demo outputs
#
# Usage:
#   julia --project=. examples/plot_airplane_rcs.jl [data_dir] [out_dir]
#
# Defaults:
#   data_dir = ../data
#   out_dir  = ../figs

using CSV
using DataFrames
using Plots

data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data")
out_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "figs")
mkpath(out_dir)

cut_csv = joinpath(data_dir, "airplane_bistatic_rcs_phi0.csv")
mono_csv = joinpath(data_dir, "airplane_monostatic_rcs.csv")

isfile(cut_csv) || error("Missing file: $cut_csv")
isfile(mono_csv) || error("Missing file: $mono_csv")

cut = CSV.read(cut_csv, DataFrame)
mono = CSV.read(mono_csv, DataFrame)
sort!(cut, :theta_deg)

default(
    linewidth = 2,
    framestyle = :box,
    grid = true,
    legendfontsize = 9,
    guidefontsize = 11,
    tickfontsize = 9,
)

p1 = plot(
    cut.theta_deg,
    cut.sigma_dBsm,
    color = :royalblue,
    xlabel = "θ (deg)",
    ylabel = "RCS (dBsm)",
    label = "Bistatic cut (φ ≈ $(round(cut.phi_cut_deg[1], digits=1))°)",
    title = "Airplane PEC RCS — dB scale",
)
scatter!(
    p1,
    [mono.theta_obs_deg[1]],
    [mono.sigma_dBsm[1]],
    marker = (:star5, 8, :crimson),
    label = "Monostatic sample",
)
annotate!(
    p1,
    mono.theta_obs_deg[1] + 5,
    mono.sigma_dBsm[1] + 1.2,
    text("σ = $(round(mono.sigma_dBsm[1], digits=2)) dBsm", 8, :crimson),
)

p2 = plot(
    cut.theta_deg,
    cut.sigma_m2,
    color = :darkorange,
    xlabel = "θ (deg)",
    ylabel = "RCS (m²)",
    label = "Bistatic cut (linear)",
    title = "Airplane PEC RCS — linear scale",
)
scatter!(
    p2,
    [mono.theta_obs_deg[1]],
    [mono.sigma_m2[1]],
    marker = (:star5, 8, :crimson),
    label = "Monostatic sample",
)

p = plot(p1, p2, layout = (2, 1), size = (800, 700))

png_path = joinpath(out_dir, "airplane_rcs_heuristic.png")
pdf_path = joinpath(out_dir, "airplane_rcs_heuristic.pdf")
savefig(p, png_path)
savefig(p, pdf_path)

println("Saved:")
println("  $png_path")
println("  $pdf_path")
