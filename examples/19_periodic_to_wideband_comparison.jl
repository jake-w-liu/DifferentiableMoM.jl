# 19_periodic_to_wideband_comparison.jl — Fair baseline + wideband robustness + optimized-case cross-check
#
# Evaluates three matched designs on the same λ/2 periodic unit cell:
#   1) PEC reference,
#   2) deterministic checkerboard baseline,
#   3) optimized density from examples/15.
#
# Reports frequency robustness and a direct-vs-GMRES solver-path cross-check
# for the optimized design at the design frequency.
#
# Run: julia --project=. examples/19_periodic_to_wideband_comparison.jl

using DifferentiableMoM
using LinearAlgebra
using StaticArrays
using Statistics
using CSV, DataFrames
using PlotlySupply
using PlotlyKaleido
PlotlyKaleido.start(mathjax=false)

const PKG_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "..", "data")
const FIG_DIR = joinpath(PKG_DIR, "..", "figures")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

println("=" ^ 70)
println("  Wideband + Comparative Evaluation (Matched Periodic Cell)")
println("=" ^ 70)

# Matched geometry (same as examples/15)
f_design = 10e9
c0 = 3e8
lambda_design = c0 / f_design

dx_cell = 0.5 * lambda_design
dy_cell = 0.5 * lambda_design
Nx = 10
Ny = 10

mesh = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
k_design = 2π / lambda_design
lattice_design = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k_design)
rwg = build_rwg_periodic(mesh, lattice_design; precheck=false)
Nt = ntriangles(mesh)
println("  Unit cell: λ/2 × λ/2 at 10 GHz, mesh $(Nx)×$(Ny), Nt=$(Nt), N=$(rwg.nedges)")

Mt = precompute_triangle_mass(mesh, rwg)
config = DensityConfig(; p=3.0, Z_max_factor=10.0, vf_target=0.5)

# Frequency sweep controls (GHz)
fmin_ghz = parse(Float64, get(ENV, "DMOM_WB_FMIN_GHZ", "8.0"))
fmax_ghz = parse(Float64, get(ENV, "DMOM_WB_FMAX_GHZ", "12.0"))
fstep_ghz = parse(Float64, get(ENV, "DMOM_WB_FSTEP_GHZ", "0.5"))
freqs_ghz = collect(fmin_ghz:fstep_ghz:fmax_ghz)
length(freqs_ghz) >= 2 || error("Need at least two frequency points")
println("  Sweep: $(first(freqs_ghz))–$(last(freqs_ghz)) GHz (step $(fstep_ghz) GHz)")

# Load optimized density field from examples/15 (projected rho_bar)
rho_file = joinpath(DATA_DIR, "results_rho_final.csv")
isfile(rho_file) || error("Missing data/results_rho_final.csv. Run examples/15_periodic_to_demo.jl first.")
rho_df = CSV.read(rho_file, DataFrame)
length(rho_df.rho_bar) == Nt || error("results_rho_final.csv triangle count mismatch (expected $(Nt))")
rho_opt = Float64.(rho_df.rho_bar)

# Deterministic checkerboard baseline (matched cell/mesh and vf ≈ 0.5)
centroids = [triangle_center(mesh, t) for t in 1:Nt]
xs = [c[1] for c in centroids]
ys = [c[2] for c in centroids]
x0, y0 = minimum(xs), minimum(ys)

rho_checker = zeros(Float64, Nt)
for t in 1:Nt
    ix = clamp(floor(Int, (centroids[t][1] - x0) / (dx_cell / Nx)) + 1, 1, Nx)
    iy = clamp(floor(Int, (centroids[t][2] - y0) / (dy_cell / Ny)) + 1, 1, Ny)
    rho_checker[t] = isodd(ix + iy) ? 1.0 : 0.0
end

# Ensure exact matched fill fraction with the optimized design (fairness control)
opt_vf = mean(rho_opt)
checker_rank = sortperm(rho_checker; rev=true)
fill_count = clamp(round(Int, opt_vf * Nt), 0, Nt)
rho_checker .= 0.0
for i in 1:fill_count
    rho_checker[checker_rank[i]] = 1.0
end

rho_pec = ones(Float64, Nt)

cases = [
    (name="PEC", rho=rho_pec),
    (name="Checkerboard", rho=rho_checker),
    (name="Optimized", rho=rho_opt),
]

println("  Volume fractions: PEC=$(round(mean(rho_pec), digits=3)), checkerboard=$(round(mean(rho_checker), digits=3)), optimized=$(round(mean(rho_opt), digits=3))")

function analyze_case(freq_hz::Float64, case_name::String, rho_bar::Vector{Float64},
                      mesh, rwg, Mt, config, dx_cell, dy_cell)
    k = 2π * freq_hz / 3e8
    lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)

    Z_per = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice))
    pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
    v = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw))

    grid_ff = make_sph_grid(20, 40)
    Q_spec = Matrix{ComplexF64}(specular_rcs_objective(
        mesh, rwg, grid_ff, k, lattice; half_angle=10 * π / 180, polarization=:x))

    Z_pen = assemble_Z_penalty(Mt, rho_bar, config)
    Z_tot = Z_per + Z_pen
    I = Z_tot \ v

    J_spec = real(dot(I, Q_spec * I))
    modes, R = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I), k, lattice;
                                       pol=SVector(1.0, 0.0, 0.0), E0=1.0)
    idx00 = findfirst(m -> m.m == 0 && m.n == 0, modes)
    idx00 === nothing && error("(0,0) Floquet mode missing at $(freq_hz/1e9) GHz")

    pb = power_balance(Vector{ComplexF64}(I), Z_pen, dx_cell * dy_cell, k, modes, R;
                       transmission=:floquet)

    return (
        J_spec=J_spec,
        R00_abs=abs(R[idx00]),
        R00_db=20 * log10(max(abs(R[idx00]), 1e-30)),
        refl_frac=pb.refl_frac,
        abs_frac=pb.abs_frac,
        trans_frac=pb.trans_frac,
        resid_frac=pb.resid_frac,
        vf=mean(rho_bar),
    )
end

rows = DataFrame(
    freq_ghz=Float64[],
    case=String[],
    J_spec=Float64[],
    R00_abs=Float64[],
    R00_db=Float64[],
    refl_frac=Float64[],
    abs_frac=Float64[],
    trans_frac=Float64[],
    resid_frac=Float64[],
    vf=Float64[],
)

for fghz in freqs_ghz
    freq_hz = fghz * 1e9
    println("\n  Evaluating $(round(fghz, digits=3)) GHz")
    for c in cases
        r = analyze_case(freq_hz, c.name, c.rho, mesh, rwg, Mt, config, dx_cell, dy_cell)
        push!(rows, (fghz, c.name, r.J_spec, r.R00_abs, r.R00_db,
                     r.refl_frac, r.abs_frac, r.trans_frac, r.resid_frac, r.vf))
        println("    $(rpad(c.name, 12)) |R00|=$(round(r.R00_abs, sigdigits=4))  J=$(round(r.J_spec, sigdigits=4))")
    end
end

# Add relative metrics versus PEC and checkerboard at each frequency
rows.J_vs_pec_db = zeros(Float64, nrow(rows))
rows.R00_vs_pec_db = zeros(Float64, nrow(rows))
rows.J_vs_checker_db = fill(NaN, nrow(rows))
rows.R00_vs_checker_db = fill(NaN, nrow(rows))

for fghz in freqs_ghz
    idx_f = findall(rows.freq_ghz .== fghz)
    sub = rows[idx_f, :]

    i_pec = findfirst(sub.case .== "PEC")
    i_chk = findfirst(sub.case .== "Checkerboard")
    (i_pec === nothing || i_chk === nothing) && error("Missing PEC/checkerboard row at $(fghz) GHz")

    J_pec = sub.J_spec[i_pec]
    R_pec = sub.R00_abs[i_pec]
    J_chk = sub.J_spec[i_chk]
    R_chk = sub.R00_abs[i_chk]

    for (local_i, global_i) in enumerate(idx_f)
        rows.J_vs_pec_db[global_i] = 10 * log10(max(sub.J_spec[local_i], 1e-30) / max(J_pec, 1e-30))
        rows.R00_vs_pec_db[global_i] = 20 * log10(max(sub.R00_abs[local_i], 1e-30) / max(R_pec, 1e-30))
        rows.J_vs_checker_db[global_i] = 10 * log10(max(sub.J_spec[local_i], 1e-30) / max(J_chk, 1e-30))
        rows.R00_vs_checker_db[global_i] = 20 * log10(max(sub.R00_abs[local_i], 1e-30) / max(R_chk, 1e-30))
    end
end

CSV.write(joinpath(DATA_DIR, "results_wideband_comparison.csv"), rows)
println("\n  ✓ Saved: data/results_wideband_comparison.csv")

# Design-frequency summary table
f0 = f_design / 1e9
idx_f0 = findall(rows.freq_ghz .== f0)
if isempty(idx_f0)
    idx_f0 = [argmin(abs.(rows.freq_ghz .- f0))]
end
summary_df = rows[idx_f0, [:freq_ghz, :case, :J_spec, :R00_abs, :R00_db,
                            :J_vs_pec_db, :R00_vs_pec_db,
                            :J_vs_checker_db, :R00_vs_checker_db,
                            :refl_frac, :abs_frac, :trans_frac, :resid_frac, :vf]]
CSV.write(joinpath(DATA_DIR, "results_comparative_summary.csv"), summary_df)
println("  ✓ Saved: data/results_comparative_summary.csv")

# Optimized-case solver-path cross-check (direct vs GMRES)
println("\n▸ Optimized-case cross-check (direct vs GMRES) at 10 GHz")
k0 = 2π * f_design / c0
lattice0 = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k0)
Z_per0 = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k0, lattice0))
Z_pen_opt = assemble_Z_penalty(Mt, rho_opt, config)
Z_opt = Z_per0 + Z_pen_opt
pw0 = make_plane_wave(Vec3(0.0, 0.0, -k0), 1.0, Vec3(1.0, 0.0, 0.0))
v0 = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw0))

I_direct = Z_opt \ v0
I_gmres = solve_forward(Z_opt, v0; solver=:gmres, gmres_tol=1e-10, gmres_maxiter=800)
rel_I = norm(I_gmres - I_direct) / max(norm(I_direct), 1e-30)
rel_resid = norm(Z_opt * I_gmres - v0) / max(norm(v0), 1e-30)

grid0 = make_sph_grid(20, 40)
Q0 = Matrix{ComplexF64}(specular_rcs_objective(mesh, rwg, grid0, k0, lattice0;
                                                half_angle=10 * π / 180, polarization=:x))
J_direct = real(dot(I_direct, Q0 * I_direct))
J_gmres = real(dot(I_gmres, Q0 * I_gmres))

modes_d, R_d = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_direct), k0, lattice0;
                                       pol=SVector(1.0, 0.0, 0.0), E0=1.0)
modes_g, R_g = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_gmres), k0, lattice0;
                                       pol=SVector(1.0, 0.0, 0.0), E0=1.0)
idx00_d = findfirst(m -> m.m == 0 && m.n == 0, modes_d)
idx00_g = findfirst(m -> m.m == 0 && m.n == 0, modes_g)
(idx00_d === nothing || idx00_g === nothing) && error("Missing (0,0) Floquet mode in cross-check")

R00_d = abs(R_d[idx00_d])
R00_g = abs(R_g[idx00_g])

cross_df = DataFrame(
    freq_ghz=[f0],
    rel_current_error=[rel_I],
    gmres_rel_residual=[rel_resid],
    J_direct=[J_direct],
    J_gmres=[J_gmres],
    J_rel_error=[abs(J_gmres - J_direct) / max(abs(J_direct), 1e-30)],
    R00_direct_abs=[R00_d],
    R00_gmres_abs=[R00_g],
    R00_abs_delta=[abs(R00_g - R00_d)],
    R00_delta_db=[20 * log10(max(R00_g, 1e-30) / max(R00_d, 1e-30))],
)
CSV.write(joinpath(DATA_DIR, "results_solver_crosscheck_optimized.csv"), cross_df)
println("  ✓ Saved: data/results_solver_crosscheck_optimized.csv")
println("    rel current error = $(round(rel_I, sigdigits=4))")
println("    rel GMRES residual = $(round(rel_resid, sigdigits=4))")
println("    |R00| direct=$(round(R00_d, sigdigits=5)), gmres=$(round(R00_g, sigdigits=5))")

# Figures
freq_axis = sort(unique(rows.freq_ghz))

function series_for(case_name::String, col::Symbol)
    sub = rows[rows.case .== case_name, :]
    p = sortperm(sub.freq_ghz)
    return collect(sub.freq_ghz[p]), collect(sub[!, col][p])
end

x_pec, y_pec_r = series_for("PEC", :R00_abs)
x_chk, y_chk_r = series_for("Checkerboard", :R00_abs)
x_opt, y_opt_r = series_for("Optimized", :R00_abs)

fig_r = plot_scatter(
    [x_pec, x_chk, x_opt],
    [y_pec_r, y_chk_r, y_opt_r];
    mode=["lines+markers", "lines+markers", "lines+markers"],
    legend=["PEC", "Checkerboard baseline", "Optimized"],
    color=["#0072B2", "#D55E00", "#009E73"],
    dash=["solid", "dash", "dashdot"],
    marker_size=[6, 6, 6],
    xlabel="Frequency [GHz]",
    ylabel="Specular Floquet amplitude |R₀₀|",
    title="Wideband specular reflection comparison",
    width=560, height=400, fontsize=14)
set_legend!(fig_r; position=:topright)
PlotlySupply.savefig(fig_r, joinpath(FIG_DIR, "fig_results_wideband_r00_comparison.pdf"))

x_chk2, y_chk2 = series_for("Checkerboard", :J_vs_pec_db)
x_opt2, y_opt2 = series_for("Optimized", :J_vs_pec_db)
zero_line = zeros(length(freq_axis))

fig_j = plot_scatter(
    [collect(freq_axis), x_chk2, x_opt2],
    [zero_line, y_chk2, y_opt2];
    mode=["lines", "lines+markers", "lines+markers"],
    legend=["PEC reference (0 dB)", "Checkerboard baseline", "Optimized"],
    color=["#808080", "#D55E00", "#009E73"],
    dash=["dot", "dash", "dashdot"],
    marker_size=[0, 6, 6],
    xlabel="Frequency [GHz]",
    ylabel="Cone objective vs PEC [dB]",
    title="Wideband objective suppression under matched conditions",
    width=560, height=400, fontsize=14)
set_legend!(fig_j; position=:topright)
PlotlySupply.savefig(fig_j, joinpath(FIG_DIR, "fig_results_wideband_objective_comparison.pdf"))

println("\n  ✓ Saved: figures/fig_results_wideband_r00_comparison.pdf")
println("  ✓ Saved: figures/fig_results_wideband_objective_comparison.pdf")

println("\n" * "=" ^ 70)
println("  Comparative/Wideband Summary @ $(round(f0, digits=2)) GHz")
println("=" ^ 70)
for row in eachrow(summary_df)
    println("  $(rpad(row.case, 12)) |R00|=$(round(row.R00_abs, sigdigits=4)), " *
            "J_vs_PEC=$(round(row.J_vs_pec_db, digits=2)) dB, " *
            "J_vs_checker=$(round(row.J_vs_checker_db, digits=2)) dB")
end
println("=" ^ 70)
