# ex_horn_pattern_import_demo.jl
#
# Horn-pattern import demo:
#   1) load external (Eθ, Eϕ) pattern CSV
#   2) convert to Pattern objects (RadiationPatterns.jl when available)
#   3) build PatternFeedExcitation
#   4) run a PEC scattering solve on a reflector (or plate)
#   5) plot bistatic cut and incident-pattern cut
#
# Run:
#   julia --project=. examples/ex_horn_pattern_import_demo.jl
#   julia --project=. examples/ex_horn_pattern_import_demo.jl examples/antenna_pattern.csv 28.0 reflector
#   julia --project=. examples/ex_horn_pattern_import_demo.jl examples/antenna_pattern.csv 28.0 plate

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
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
println("Horn-pattern import demo: external Eθ/Eϕ pattern as incident field")
println("="^68)

csv_path = isempty(ARGS) ? joinpath(@__DIR__, "antenna_pattern.csv") : ARGS[1]
freq = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) * 1e9 : 28.0e9
geometry = length(ARGS) >= 3 ? lowercase(strip(ARGS[3])) : "reflector"
(geometry == "reflector" || geometry == "plate") ||
    error("Geometry must be \"reflector\" or \"plate\".")

c0 = 299792458.0
λ0 = c0 / freq
k = 2π / λ0
η0 = 376.730313668

raw = load_pattern_csv(csv_path; drop_phi_endpoint=true)
pat_objs = maybe_make_radiationpatterns_patterns(raw.theta_deg, raw.phi_deg, raw.Ftheta, raw.Fphi)

println("Pattern input:")
println("  file: $csv_path")
println("  backend: $(pat_objs.backend)")
println("  Nθ×Nϕ = $(length(raw.theta_deg)) × $(length(raw.phi_deg))")

# Geometry + feed setup.
mesh = nothing
geom_desc = ""
if geometry == "reflector"
    # Simple front-fed parabolic reflector:
    # z = (x² + y²)/(4f), aperture diameter D.
    D = 0.30
    f_over_D = 0.35
    f = f_over_D * D
    Nr, Nphi = 8, 28
    mesh = make_parabolic_reflector(D, f, Nr, Nphi)
    phase_center = Vec3(0.0, 0.0, f) # horn phase center near focus
    geom_desc = "parabolic reflector (D=$(round(D, digits=3)) m, f=$(round(f, digits=3)) m, f/D=$(round(f_over_D, digits=3)))"
else
    # Legacy plate fallback.
    Lx = 1.5 * λ0
    Ly = 1.5 * λ0
    Nx, Ny = 8, 8
    mesh = make_rect_plate(Lx, Ly, Nx, Ny)
    phase_center = Vec3(0.0, 0.0, -1.0)
    geom_desc = "flat plate ($(round(Lx, digits=4)) m × $(round(Ly, digits=4)) m)"
end

pat_feed = make_pattern_feed(
    pat_objs.Etheta_pattern,
    pat_objs.Ephi_pattern,
    freq;
    angles_in_degrees=true,
    phase_center=phase_center,
    convention=:exp_plus_iwt,
)

rwg = build_rwg(mesh)

println("Scattering setup:")
println("  freq = $(round(freq / 1e9, digits=3)) GHz (λ=$(round(λ0, digits=4)) m)")
println("  geometry = $geom_desc")
println("  mesh: Nv=$(nvertices(mesh)), Nt=$(ntriangles(mesh)), Nrwg=$(rwg.nedges)")
println("  phase center = $(Tuple(phase_center)) m")

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)
v_horn = assemble_excitation(mesh, rwg, pat_feed; quad_order=3)
I_horn = solve_forward(Z_efie, v_horn)
residual = norm(Z_efie * I_horn - v_horn) / max(norm(v_horn), 1e-30)

grid = make_sph_grid(181, 72)
NΩ = length(grid.w)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
E_ff = compute_farfield(G, I_horn, NΩ)
σ = bistatic_rcs(E_ff; E0=1.0)

ϕ_target = 0.0
ϕ_idx = argmin(abs.(grid.phi .- ϕ_target))
ϕ_sel = grid.phi[ϕ_idx]
cut_idx = [q for q in 1:NΩ if abs(grid.phi[q] - ϕ_sel) < 1e-12]
perm = sortperm(grid.theta[cut_idx])
cut_idx = cut_idx[perm]

df_rcs = DataFrame(
    theta_deg = rad2deg.(grid.theta[cut_idx]),
    phi_deg = fill(rad2deg(ϕ_sel), length(cut_idx)),
    sigma_m2 = σ[cut_idx],
    sigma_dBsm = 10 .* log10.(max.(σ[cut_idx], 1e-30)),
)
prefix = geometry == "reflector" ? "horn_reflector_import" : "horn_pattern_import"
csv_rcs = joinpath(DATADIR, "$(prefix)_scatter_phi0.csv")
CSV.write(csv_rcs, df_rcs)

inc_cut = phi_cut_power(raw.theta_deg, raw.phi_deg, raw.Ftheta, raw.Fphi; phi_target_deg=0.0)
df_inc = DataFrame(
    theta_deg = inc_cut.theta_deg,
    phi_deg = fill(inc_cut.phi_used_deg, length(inc_cut.theta_deg)),
    power_norm = inc_cut.power,
    power_dB = [to_dB(x) for x in inc_cut.power],
)
csv_inc = joinpath(DATADIR, "$(prefix)_incident_phi0.csv")
CSV.write(csv_inc, df_inc)

khat_inc = Vec3(0.0, 0.0, 1.0) # nominal forward direction from feed centerline
bs = backscatter_rcs(E_ff, grid, khat_inc; E0=1.0)
eratio = energy_ratio(I_horn, v_horn, E_ff, grid; eta0=η0)

df_summary = DataFrame(
    metric = [
        "backend",
        "geometry",
        "frequency_GHz",
        "lambda_m",
        "phase_center_x_m",
        "phase_center_y_m",
        "phase_center_z_m",
        "mesh_vertices",
        "mesh_triangles",
        "rwg_unknowns",
        "solve_relative_residual",
        "energy_ratio_prad_pin",
        "monostatic_sigma_m2",
        "monostatic_sigma_dBsm",
        "monostatic_theta_deg",
        "monostatic_phi_deg",
        "monostatic_sample_error_deg",
    ],
    value = Any[
        String(pat_objs.backend),
        geometry,
        freq / 1e9,
        λ0,
        phase_center[1],
        phase_center[2],
        phase_center[3],
        nvertices(mesh),
        ntriangles(mesh),
        rwg.nedges,
        residual,
        eratio,
        bs.sigma,
        10 * log10(max(bs.sigma, 1e-30)),
        rad2deg(bs.theta),
        rad2deg(bs.phi),
        bs.angular_error_deg,
    ],
)
csv_summary = joinpath(DATADIR, "$(prefix)_summary.csv")
CSV.write(csv_summary, df_summary)

p_inc = plot(
    df_inc.theta_deg,
    df_inc.power_dB;
    lw=2,
    color=:blue,
    label="Imported pattern (φ=$(round(df_inc.phi_deg[1], digits=2))°)",
    xlabel="θ (deg)",
    ylabel="Normalized |E|² (dB)",
    title="Incident horn pattern cut",
)

p_rcs = plot(
    df_rcs.theta_deg,
    df_rcs.sigma_dBsm;
    lw=2,
    color=:red,
    label="PEC scatter (φ=$(round(df_rcs.phi_deg[1], digits=2))°)",
    xlabel="θ (deg)",
    ylabel="σ (dBsm)",
    title="Bistatic cut under imported horn pattern ($geometry)",
)
scatter!(
    p_rcs,
    [rad2deg(bs.theta)],
    [10 * log10(max(bs.sigma, 1e-30))];
    marker=:star5,
    color=:black,
    ms=8,
    label="Monostatic sample",
)

p_err = plot(
    df_rcs.theta_deg,
    df_rcs.sigma_m2;
    lw=2,
    color=:darkgreen,
    label="σ (linear)",
    xlabel="θ (deg)",
    ylabel="σ (m²)",
    title="Linear-scale RCS cut",
)

fig = plot(p_inc, p_rcs, p_err; layout=(3, 1), size=(960, 1100))
png_path = joinpath(DATADIR, "$(prefix)_demo.png")
savefig(fig, png_path)

println("Results:")
println("  relative residual = $residual")
println("  energy ratio Prad/Pin = $eratio")
println("  monostatic σ = $(bs.sigma) m² ($(round(10*log10(max(bs.sigma, 1e-30)), digits=3)) dBsm)")
println("Saved:")
println("  $csv_inc")
println("  $csv_rcs")
println("  $csv_summary")
println("  $png_path")
