# 09_mom_vs_po.jl — MoM vs Physical Optics RCS comparison
#
# Demonstrates:
#   Part A: Flat plate — PO vs MoM vs analytical PO formula
#   Part B: Aircraft at 0.3 GHz (14λ wingspan) — PO, direct, iterative solvers
#           at three mesh refinement levels, with 1° RCS pattern plots
#
# Run: julia -t 4 --project=. examples/09_mom_vs_po.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM
using LinearAlgebra
using PlotlySupply
import PlotlyKaleido
PlotlyKaleido.start()

println("="^60)
println("Example 09: MoM vs Physical Optics")
println("="^60)

c0 = 299792458.0
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

"""
    make_cut_grid(Ntheta, phi_values)

Create a SphGrid with 1° θ resolution at specific φ planes only.
Much smaller than a full sphere grid → feasible for large N far-field.
"""
function make_cut_grid(Ntheta::Int, phi_values::Vector{Float64})
    dtheta = π / Ntheta
    Nphi = length(phi_values)
    NΩ = Ntheta * Nphi
    rhat  = zeros(3, NΩ)
    theta = zeros(NΩ)
    phi   = zeros(NΩ)
    w     = zeros(NΩ)
    idx = 0
    for it in 1:Ntheta
        θ = (it - 0.5) * dtheta
        for φ in phi_values
            idx += 1
            theta[idx] = θ
            phi[idx]   = φ
            rhat[1, idx] = sin(θ) * cos(φ)
            rhat[2, idx] = sin(θ) * sin(φ)
            rhat[3, idx] = cos(θ)
            w[idx] = sin(θ) * dtheta * (2π / Nphi)
        end
    end
    return SphGrid(rhat, theta, phi, w)
end

# ════════════════════════════════════════════════════
# Part A: Flat plate validation
# ════════════════════════════════════════════════════
println("\n" * "─"^60)
println("Part A: Flat plate — PO vs MoM vs analytical")
println("─"^60)

freq = 3e9
λ0   = c0 / freq
k    = 2π / λ0

Lx, Ly = 2λ0, 2λ0
Nx, Ny = 10, 10
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges

println("\nFrequency: $(freq/1e9) GHz, λ = $(round(λ0*100, digits=2)) cm")
println("Plate: $(round(Lx/λ0, digits=1))λ × $(round(Ly/λ0, digits=1))λ, $N RWG unknowns")

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

# 1° cut grid for plate (φ=0 only, plate is symmetric)
grid_plate = make_cut_grid(180, [0.0])
NΩ_plate = length(grid_plate.w)

# PO
t_po = @elapsed po = solve_po(mesh, freq, pw; grid=grid_plate)
println("\nPO: $(ntriangles(mesh)) tri, $(count(po.illuminated)) illum, $(round(t_po, digits=3)) s")

# MoM
t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k)
v = assemble_excitation(mesh, rwg, pw)
t_sol = @elapsed I_mom = Z \ v
G_mat = radiation_vectors(mesh, rwg, grid_plate, k)
E_ff_mom = compute_farfield(G_mat, Vector{ComplexF64}(I_mom), NΩ_plate)
println("MoM: $N unknowns, asm $(round(t_asm, digits=3)) s, solve $(round(t_sol, digits=3)) s")

σ_po_plate  = 10 .* log10.(max.(bistatic_rcs(po.E_ff; E0=1.0), 1e-30))
σ_mom_plate = 10 .* log10.(max.(bistatic_rcs(E_ff_mom; E0=1.0), 1e-30))

A_plate = Lx * Ly
σ_analytical_dB = 10 * log10(4π * A_plate^2 / λ0^2)
best_idx = argmax([grid_plate.rhat[3, q] for q in 1:NΩ_plate])

println("\n── Specular (θ ≈ 0°) ──")
println("  Analytical: $(round(σ_analytical_dB, digits=2)) dBsm")
println("  PO:         $(round(σ_po_plate[best_idx], digits=2)) dBsm")
println("  MoM:        $(round(σ_mom_plate[best_idx], digits=2)) dBsm")

# Flat plate plot — all points are φ=0, sorted by θ
let
    idx = sortperm(grid_plate.theta)
    θ_deg = rad2deg.(grid_plate.theta[idx])
    sf = subplots(1, 1; sync=false, width=800, height=500,
                  subplot_titles=reshape(["Flat Plate 2λ×2λ — φ=0° Cut (3 GHz, 1° resolution)"], 1, 1))
    addtraces!(sf, scatter(x=θ_deg, y=σ_po_plate[idx], mode="lines",
               name="PO", line=attr(color="blue", width=2)); row=1, col=1)
    addtraces!(sf, scatter(x=θ_deg, y=σ_mom_plate[idx], mode="lines",
               name="MoM (N=$N)", line=attr(color="red", width=2, dash="dash")); row=1, col=1)
    p = sf.plot
    relayout!(p, xaxis=attr(title="θ (deg)", range=[0, 90]),
              yaxis=attr(title="Bistatic RCS (dBsm)", range=[-40, 10]),
              legend=attr(x=0.65, y=0.95), margin=attr(l=60, r=30, t=60, b=50))
    PlotlyKaleido.savefig(p, joinpath(figdir, "09_flat_plate_rcs.png"); width=800, height=500)
    println("\nPlot saved: 09_flat_plate_rcs.png")
end

# ════════════════════════════════════════════════════
# Part B: Aircraft at 0.3 GHz — direct vs iterative
# ════════════════════════════════════════════════════
println("\n" * "─"^60)
println("Part B: Aircraft at 0.3 GHz — PO / Direct / Iterative")
println("─"^60)

obj_path = joinpath(@__DIR__, "demo_aircraft.obj")
if !isfile(obj_path)
    println("SKIP: demo_aircraft.obj not found")
else
    mesh_raw = read_obj_mesh(obj_path)
    rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
    mesh_air = rep.mesh

    freq_air = 0.3e9
    λ_air = c0 / freq_air
    k_air = 2π / λ_air

    res = mesh_resolution_report(mesh_air, freq_air)
    println("\nAircraft: $(nvertices(mesh_air)) verts, $(ntriangles(mesh_air)) tri")
    println("Frequency: $(freq_air/1e9) GHz, λ = $(round(λ_air, digits=2)) m")
    println("Wingspan ~14m ≈ $(round(14.0/λ_air, digits=1))λ")
    println("Original max edge/λ: $(round(res.edge_max_over_lambda, digits=2))")

    pw_air = make_plane_wave(Vec3(0.0, 0.0, -k_air), 1.0, Vec3(1.0, 0.0, 0.0))

    # 1° θ resolution at φ=0° and φ=90° (360 observation points)
    grid_cuts = make_cut_grid(180, [0.0, π/2])
    NΩ_cuts = length(grid_cuts.w)
    println("Far-field grid: $NΩ_cuts points (1° θ × φ=0°,90°)")

    # Also keep a coarse sphere for RMSE (small N only)
    grid_coarse = make_sph_grid(36, 72)
    NΩ_coarse = length(grid_coarse.w)

    # ── 1. PO reference ──
    println("\n── PO reference ──")
    po_air = solve_po(mesh_air, freq_air, pw_air; grid=grid_cuts)
    σ_po_cuts = bistatic_rcs(po_air.E_ff; E0=1.0)
    σ_po_cuts_dB = 10 .* log10.(max.(σ_po_cuts, 1e-30))
    bs_po = backscatter_rcs(po_air.E_ff, grid_cuts, Vec3(0.0, 0.0, -k_air); E0=1.0)
    println("  $(ntriangles(mesh_air)) tri, $(count(po_air.illuminated)) illum")
    println("  Backscatter: $(round(10*log10(bs_po.sigma), digits=2)) dBsm")

    # PO on coarse grid for RMSE reference
    po_air_coarse = solve_po(mesh_air, freq_air, pw_air; grid=grid_coarse)
    σ_po_coarse_dB = 10 .* log10.(max.(bistatic_rcs(po_air_coarse.E_ff; E0=1.0), 1e-30))

    # ── 2. Mesh levels ──
    target1 = 4.0 * λ_air   # 4λ → 1 pass
    target2 = 1.0 * λ_air   # 1λ → 3 passes

    println("\n── Mesh refinement ──")
    ref1 = refine_mesh_to_target_edge(mesh_air, target1; max_iters=2, max_triangles=50_000)
    println("  refine1: target $(round(target1, digits=1))m (4λ), $(ntriangles(ref1.mesh)) tri, $(ref1.iterations) pass, max edge $(round(ref1.edge_max_after_m, digits=2))m")
    ref2 = refine_mesh_to_target_edge(mesh_air, target2; max_iters=5, max_triangles=80_000)
    println("  refine2: target $(round(target2, digits=1))m (1λ), $(ntriangles(ref2.mesh)) tri, $(ref2.iterations) passes, max edge $(round(ref2.edge_max_after_m, digits=2))m")

    mesh_levels = [
        (mesh_air, "original"),
        (ref1.mesh, "refine1 (4λ)"),
        (ref2.mesh, "refine2 (1λ)"),
    ]

    # ── 3. Solve at each level ──
    all_curves = Dict{String, Vector{Float64}}()
    all_curves["PO"] = σ_po_cuts_dB
    curve_order = ["PO"]

    MAX_N_DIRECT = 10_000
    # NF cutoff: 5λ for small N, 2λ for large N (ILU handles the fill-in)
    nf_cutoff_for(N_m) = N_m <= 10_000 ? 5.0 * λ_air : 2.0 * λ_air
    # Factorization: full LU for small N, ILU for large N (avoids OOM)
    nf_fac_for(N_m) = N_m <= 10_000 ? :lu : :ilu
    gmres_maxiter_for(N_m) = N_m <= 10_000 ? 1000 : 2000
    ILU_TAU = 1e-3

    println("\n── Solver comparison (NF: 5λ+LU for N≤10k, 2λ+ILU for N>10k) ──")
    println("  Level          N     Solver        Asm(s)  Sol(s)  Iters  Backscatter  RMSE vs PO")
    println("  " * "─"^88)

    for (m, label) in mesh_levels
        rwg_m = build_rwg(m)
        N_m = rwg_m.nedges
        v_m = assemble_excitation(m, rwg_m, pw_air)

        # ── Direct solver ──
        if N_m <= MAX_N_DIRECT
            t_asm_d = @elapsed Z_m = assemble_Z_efie(m, rwg_m, k_air)
            t_sol_d = @elapsed I_d = Z_m \ v_m

            # Far-field on cut grid (1°)
            G_d = radiation_vectors(m, rwg_m, grid_cuts, k_air)
            E_ff_d = compute_farfield(G_d, Vector{ComplexF64}(I_d), NΩ_cuts)
            σ_d_dB = 10 .* log10.(max.(bistatic_rcs(E_ff_d; E0=1.0), 1e-30))
            bs_d = backscatter_rcs(E_ff_d, grid_cuts, Vec3(0.0, 0.0, -k_air); E0=1.0)

            # RMSE on coarse grid
            G_d_c = radiation_vectors(m, rwg_m, grid_coarse, k_air)
            E_ff_d_c = compute_farfield(G_d_c, Vector{ComplexF64}(I_d), NΩ_coarse)
            σ_d_c_dB = 10 .* log10.(max.(bistatic_rcs(E_ff_d_c; E0=1.0), 1e-30))
            rmse_d = sqrt(sum((σ_d_c_dB .- σ_po_coarse_dB).^2) / NΩ_coarse)

            bs_d_dB = round(10 * log10(max(bs_d.sigma, 1e-30)), digits=2)
            println("  $(rpad(label, 14))  $(lpad(N_m, 5))  $(lpad("direct", 12))  " *
                    "$(lpad(round(t_asm_d, digits=1), 6))  $(lpad(round(t_sol_d, digits=1), 6))      -  " *
                    "$(lpad(bs_d_dB, 10)) dBsm  $(round(rmse_d, digits=2)) dB")

            dlabel = "$label direct (N=$N_m)"
            all_curves[dlabel] = σ_d_dB
            push!(curve_order, dlabel)
        else
            println("  $(rpad(label, 14))  $(lpad(N_m, 5))  $(lpad("direct", 12))    SKIP (N > $MAX_N_DIRECT)")
        end

        # ── Iterative solver (ACA + GMRES + NF preconditioner) ──
        t_asm_i = @elapsed A_m = build_aca_operator(m, rwg_m, k_air;
            leaf_size=32, eta=1.5, aca_tol=1e-6, max_rank=80)
        cutoff = nf_cutoff_for(N_m)
        fac = nf_fac_for(N_m)
        println("    Building NF preconditioner (cutoff=$(round(cutoff, digits=1))m, fac=$fac)...")
        t_nf = @elapsed P_nf = build_nearfield_preconditioner(m, rwg_m, k_air, cutoff;
            factorization=fac, ilu_tau=ILU_TAU)
        fac_info = fac == :ilu ? "ilu(τ=$(ILU_TAU))" : "lu"
        println("    NF: nnz=$(round(P_nf.nnz_ratio * 100, digits=1))%, $fac_info, build $(round(t_nf, digits=1))s")
        t_sol_i = @elapsed begin
            I_i, stats = solve_gmres(A_m, v_m; preconditioner=P_nf,
                tol=1e-6, maxiter=gmres_maxiter_for(N_m))
        end

        # Far-field on cut grid (1°)
        G_i = radiation_vectors(m, rwg_m, grid_cuts, k_air)
        E_ff_i = compute_farfield(G_i, Vector{ComplexF64}(I_i), NΩ_cuts)
        σ_i_dB = 10 .* log10.(max.(bistatic_rcs(E_ff_i; E0=1.0), 1e-30))
        bs_i = backscatter_rcs(E_ff_i, grid_cuts, Vec3(0.0, 0.0, -k_air); E0=1.0)

        # RMSE on coarse grid (skip for very large N to save memory)
        rmse_str = ""
        if N_m <= 15_000
            G_i_c = radiation_vectors(m, rwg_m, grid_coarse, k_air)
            E_ff_i_c = compute_farfield(G_i_c, Vector{ComplexF64}(I_i), NΩ_coarse)
            σ_i_c_dB = 10 .* log10.(max.(bistatic_rcs(E_ff_i_c; E0=1.0), 1e-30))
            rmse_i = sqrt(sum((σ_i_c_dB .- σ_po_coarse_dB).^2) / NΩ_coarse)
            rmse_str = "$(round(rmse_i, digits=2)) dB"
        else
            rmse_str = "—"
        end

        bs_i_dB = round(10 * log10(max(bs_i.sigma, 1e-30)), digits=2)
        println("  $(rpad(label, 14))  $(lpad(N_m, 5))  $(lpad("aca+gmres", 12))  " *
                "$(lpad(round(t_asm_i, digits=1), 6))  $(lpad(round(t_sol_i, digits=1), 6))  " *
                "$(lpad(stats.niter, 5))  $(lpad(bs_i_dB, 10)) dBsm  $rmse_str")

        ilabel = "$label iterative (N=$N_m)"
        all_curves[ilabel] = σ_i_dB
        push!(curve_order, ilabel)
    end

    # ════════════════════════════════════════════════════
    # Part C: RCS pattern plots (1° resolution)
    # ════════════════════════════════════════════════════
    println("\n" * "─"^60)
    println("Part C: RCS pattern plots (1° resolution)")
    println("─"^60)

    # Extract φ=0 and φ=90 cut indices from the cut grid
    phi0_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q]) < 0.01]
    phi90_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q] - π/2) < 0.01]
    sort!(phi0_idx; by=q -> grid_cuts.theta[q])
    sort!(phi90_idx; by=q -> grid_cuts.theta[q])
    θ_deg_0  = [rad2deg(grid_cuts.theta[q]) for q in phi0_idx]
    θ_deg_90 = [rad2deg(grid_cuts.theta[q]) for q in phi90_idx]

    println("  φ=0° cut: $(length(phi0_idx)) points")
    println("  φ=90° cut: $(length(phi90_idx)) points")

    # Color/dash scheme: direct = solid, iterative = dashed, same color per mesh level
    mesh_colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red
    style_map = Dict{String, NamedTuple{(:color, :width, :dash), Tuple{String, Int, String}}}()
    style_map["PO"] = (color="black", width=3, dash="solid")
    for (i, (_, label)) in enumerate(mesh_levels)
        N_m = build_rwg(mesh_levels[i][1]).nedges
        dlabel = "$label direct (N=$N_m)"
        ilabel = "$label iterative (N=$N_m)"
        style_map[dlabel] = (color=mesh_colors[i], width=2, dash="solid")
        style_map[ilabel] = (color=mesh_colors[i], width=2, dash="dash")
    end

    function make_rcs_plot(cut_idx, θ_deg, title_str)
        sf = subplots(1, 1; sync=false, width=950, height=600,
                      subplot_titles=reshape([title_str], 1, 1))
        for label in curve_order
            haskey(all_curves, label) || continue
            dB = all_curves[label]
            s = get(style_map, label, (color="gray", width=1, dash="solid"))
            addtraces!(sf,
                scatter(x=θ_deg, y=[dB[q] for q in cut_idx],
                        mode="lines", name=label,
                        line=attr(color=s.color, width=s.width, dash=s.dash));
                row=1, col=1)
        end
        p = sf.plot
        relayout!(p,
            xaxis=attr(title="θ (deg)", range=[0, 180]),
            yaxis=attr(title="Bistatic RCS (dBsm)"),
            legend=attr(x=0.01, y=0.01, xanchor="left", yanchor="bottom",
                        bgcolor="rgba(255,255,255,0.8)", font=attr(size=10)),
            margin=attr(l=60, r=30, t=60, b=50))
        return p
    end

    p0 = make_rcs_plot(phi0_idx, θ_deg_0,
        "Aircraft RCS — φ=0° ($(freq_air/1e9) GHz, λ=$(round(λ_air, digits=2))m, 1° resolution)")
    PlotlyKaleido.savefig(p0, joinpath(figdir, "09_aircraft_rcs_phi0.png"); width=950, height=600)
    println("Plot saved: 09_aircraft_rcs_phi0.png")

    p90 = make_rcs_plot(phi90_idx, θ_deg_90,
        "Aircraft RCS — φ=90° ($(freq_air/1e9) GHz, λ=$(round(λ_air, digits=2))m, 1° resolution)")
    PlotlyKaleido.savefig(p90, joinpath(figdir, "09_aircraft_rcs_phi90.png"); width=950, height=600)
    println("Plot saved: 09_aircraft_rcs_phi90.png")
end

println("\n" * "="^60)
println("Done.")
