# 10_mlfma_scaling.jl — MLFMA vs Dense vs ACA vs PO on aircraft at 0.3 GHz
#
# Demonstrates MLFMA on the 14λ aircraft at 0.3 GHz:
#   - 4λ mesh refinement → ~7.5k unknowns
#   - Compares: Dense (LU), ACA+GMRES, MLFMA+GMRES, PO
#   - RCS pattern plots at φ=0° and φ=90° (1° resolution)
#
# Key results: MLFMA matvec error ~0.008%, solution coefficient error ~0.3%,
# and RCS curves overlap Dense/ACA to within 0.03 dB. GMRES converges in
# ~20 iterations with ILU-preconditioned near-field.
#
# Run: julia -t 4 --project=. examples/10_mlfma_scaling.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM
using LinearAlgebra
using SparseArrays
using PlotlySupply
import PlotlyKaleido
PlotlyKaleido.start()

println("="^72)
println("Example 10: MLFMA vs Dense vs ACA vs PO — Aircraft at 0.3 GHz")
println("="^72)

c0 = 299792458.0
figdir = joinpath(@__DIR__, "figs")
mkpath(figdir)

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
# Setup
# ════════════════════════════════════════════════════
obj_path = joinpath(@__DIR__, "demo_aircraft.obj")
isfile(obj_path) || error("demo_aircraft.obj not found at $obj_path")

mesh_raw = read_obj_mesh(obj_path)
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
mesh_air = rep.mesh

freq = 0.3e9
λ0 = c0 / freq
k  = 2π / λ0

println("\nAircraft: $(nvertices(mesh_air)) verts, $(ntriangles(mesh_air)) tri")
println("Frequency: $(freq/1e9) GHz, λ = $(round(λ0, digits=2)) m")

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
grid_cuts = make_cut_grid(180, [0.0, π/2])
NΩ_cuts = length(grid_cuts.w)

# ── Mesh refinement: 4λ target ──
ref = refine_mesh_to_target_edge(mesh_air, 4.0 * λ0; max_iters=2, max_triangles=50_000)
mesh = ref.mesh
println("Refined: $(ntriangles(mesh)) tri, max edge $(round(ref.edge_max_after_m, digits=2))m ($(round(ref.edge_max_after_m/λ0, digits=2))λ)")

rwg = build_rwg(mesh)
N = rwg.nedges
v = assemble_excitation(mesh, rwg, pw)
println("N = $N RWG unknowns")

# Parameters
ILU_TAU = 1e-3
GMRES_TOL = 1e-6
LEAF_LAMBDA = 3.0

all_curves = Dict{String, Vector{Float64}}()
curve_order = String[]

# ════════════════════════════════════════════════════
# 1. PO
# ════════════════════════════════════════════════════
println("\n── PO (on original mesh, same as ex09) ──")
t_po = @elapsed po = solve_po(mesh_air, freq, pw; grid=grid_cuts)
σ_po = 10 .* log10.(max.(bistatic_rcs(po.E_ff; E0=1.0), 1e-30))
bs_po = backscatter_rcs(po.E_ff, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
bs_po_dB = round(10*log10(max(bs_po.sigma, 1e-30)), digits=2)
println("  $(ntriangles(mesh_air)) tri, $(count(po.illuminated)) illum, $(round(t_po, digits=2))s")
println("  Backscatter: $bs_po_dB dBsm")
all_curves["PO"] = σ_po
push!(curve_order, "PO")

# ════════════════════════════════════════════════════
# 2. Dense direct
# ════════════════════════════════════════════════════
println("\n── Dense direct ──")
t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k; mesh_precheck=false)
t_sol = @elapsed I_dense = Z \ v
G_d = radiation_vectors(mesh, rwg, grid_cuts, k)
E_ff_d = compute_farfield(G_d, Vector{ComplexF64}(I_dense), NΩ_cuts)
σ_d = 10 .* log10.(max.(bistatic_rcs(E_ff_d; E0=1.0), 1e-30))
bs_d = backscatter_rcs(E_ff_d, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
bs_d_dB = round(10*log10(max(bs_d.sigma, 1e-30)), digits=2)
println("  Assembly: $(round(t_asm, digits=1))s, Solve: $(round(t_sol, digits=1))s")
println("  Backscatter: $bs_d_dB dBsm")
all_curves["Dense (LU)"] = σ_d
push!(curve_order, "Dense (LU)")

# ════════════════════════════════════════════════════
# 3. ACA + GMRES
# ════════════════════════════════════════════════════
println("\n── ACA + NF-preconditioned GMRES ──")
t_aca = @elapsed A_aca = build_aca_operator(mesh, rwg, k;
    leaf_size=32, eta=1.5, aca_tol=1e-6, max_rank=80, mesh_precheck=false)
cutoff_a = 5.0 * λ0
t_pre_a = @elapsed P_nf_a = build_nearfield_preconditioner(mesh, rwg, k, cutoff_a;
    factorization=:lu, mesh_precheck=false)
t_sol_a = @elapsed begin
    global I_aca
    I_aca, stats_a = solve_gmres(A_aca, v; preconditioner=P_nf_a,
        tol=GMRES_TOL, maxiter=1000)
end
G_a = radiation_vectors(mesh, rwg, grid_cuts, k)
E_ff_a = compute_farfield(G_a, Vector{ComplexF64}(I_aca), NΩ_cuts)
σ_a = 10 .* log10.(max.(bistatic_rcs(E_ff_a; E0=1.0), 1e-30))
bs_a = backscatter_rcs(E_ff_a, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
bs_a_dB = round(10*log10(max(bs_a.sigma, 1e-30)), digits=2)
err_a = norm(I_aca - I_dense) / norm(I_dense)
println("  Build: $(round(t_aca, digits=1))s, Pre: $(round(t_pre_a, digits=1))s, Solve: $(round(t_sol_a, digits=1))s")
println("  Iters: $(stats_a.niter), nnz=$(round(P_nf_a.nnz_ratio*100, digits=1))%")
println("  Backscatter: $bs_a_dB dBsm, err vs dense: $(round(err_a, sigdigits=2))")
all_curves["ACA+GMRES"] = σ_a
push!(curve_order, "ACA+GMRES")

# ════════════════════════════════════════════════════
# 4. MLFMA + GMRES
# ════════════════════════════════════════════════════
println("\n── MLFMA + NF-preconditioned GMRES ──")
t_mlfma = @elapsed A_mlfma = build_mlfma_operator(mesh, rwg, k;
    leaf_lambda=LEAF_LAMBDA, verbose=true)
t_pre_m = @elapsed P_nf_m = build_nearfield_preconditioner(A_mlfma.Z_near;
    factorization=:ilu, ilu_tau=ILU_TAU)
t_sol_m = @elapsed begin
    global I_mlfma
    I_mlfma, stats_m = solve_gmres(A_mlfma, v;
        preconditioner=P_nf_m, tol=GMRES_TOL, maxiter=2000)
end
G_m = radiation_vectors(mesh, rwg, grid_cuts, k)
E_ff_m = compute_farfield(G_m, Vector{ComplexF64}(I_mlfma), NΩ_cuts)
σ_m = 10 .* log10.(max.(bistatic_rcs(E_ff_m; E0=1.0), 1e-30))
bs_m = backscatter_rcs(E_ff_m, grid_cuts, Vec3(0.0, 0.0, -k); E0=1.0)
bs_m_dB = round(10*log10(max(bs_m.sigma, 1e-30)), digits=2)
err_m = norm(I_mlfma - I_dense) / norm(I_dense)
nL = A_mlfma.octree.nLevels
n_leaf = length(A_mlfma.octree.levels[end].boxes)
nnz_pct = round(nnz(A_mlfma.Z_near) / N^2 * 100, digits=1)
println("  Build: $(round(t_mlfma, digits=1))s, Pre: $(round(t_pre_m, digits=1))s, Solve: $(round(t_sol_m, digits=1))s")
println("  $nL levels, $n_leaf leaf boxes, NF nnz=$(nnz_pct)%")
println("  Iters: $(stats_m.niter)")
println("  Backscatter: $bs_m_dB dBsm, err vs dense: $(round(err_m, sigdigits=2))")
all_curves["MLFMA+GMRES"] = σ_m
push!(curve_order, "MLFMA+GMRES")

# ── MLFMA matvec accuracy ──
println("\n── MLFMA matvec accuracy (random vector) ──")
x_test = randn(ComplexF64, N)
y_dense = Z * x_test
y_mlfma = similar(y_dense)
mul!(y_mlfma, A_mlfma, x_test)
matvec_err = norm(y_mlfma - y_dense) / norm(y_dense)
println("  Matvec rel error: $(round(matvec_err, sigdigits=3))")
println("  (Far-field is only $(round(norm(Z - Matrix(A_mlfma.Z_near))/norm(Z)*100, digits=1))% of |Z|_F)")

# ════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════
println("\n" * "="^72)
println("Summary: Aircraft at $(freq/1e9) GHz, N=$N, $(ntriangles(mesh)) tri")
println("="^72)
println("  Method       Total(s)  Iters  Backscatter  Coeff err  Notes")
println("  " * "─"^72)
println("  PO           $(lpad(round(t_po, digits=1), 7))      —  $(lpad(bs_po_dB, 10)) dBsm      —     high-freq approx")
println("  Dense        $(lpad(round(t_asm+t_sol, digits=1), 7))      —  $(lpad(bs_d_dB, 10)) dBsm    ref     exact")
println("  ACA+GMRES    $(lpad(round(t_aca+t_pre_a+t_sol_a, digits=1), 7))  $(lpad(stats_a.niter, 4))  $(lpad(bs_a_dB, 10)) dBsm    $(lpad(round(err_a, sigdigits=2), 6))")
println("  MLFMA+GMRES  $(lpad(round(t_mlfma+t_pre_m+t_sol_m, digits=1), 7))  $(lpad(stats_m.niter, 4))  $(lpad(bs_m_dB, 10)) dBsm    $(lpad(round(err_m, sigdigits=2), 6))     matvec err=$(round(matvec_err*100, digits=1))%")

# ════════════════════════════════════════════════════
# RCS plots
# ════════════════════════════════════════════════════
println("\n" * "="^72)
println("RCS plots")
println("="^72)

phi0_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q]) < 0.01]
phi90_idx = [q for q in 1:NΩ_cuts if abs(grid_cuts.phi[q] - π/2) < 0.01]
sort!(phi0_idx; by=q -> grid_cuts.theta[q])
sort!(phi90_idx; by=q -> grid_cuts.theta[q])
θ_deg_0  = [rad2deg(grid_cuts.theta[q]) for q in phi0_idx]
θ_deg_90 = [rad2deg(grid_cuts.theta[q]) for q in phi90_idx]

style_map = Dict{String, NamedTuple{(:color, :width, :dash), Tuple{String, Int, String}}}()
style_map["PO"]          = (color="gray",    width=2, dash="dot")
style_map["Dense (LU)"]  = (color="black",   width=3, dash="solid")
style_map["ACA+GMRES"]   = (color="#2ca02c", width=2, dash="dash")
style_map["MLFMA+GMRES"] = (color="#d62728", width=3, dash="solid")

function make_rcs_plot(cut_idx, θ_deg, title_str)
    sf = subplots(1, 1; sync=false, width=950, height=600,
                  subplot_titles=reshape([title_str], 1, 1))
    for clabel in curve_order
        haskey(all_curves, clabel) || continue
        dB = all_curves[clabel]
        s = get(style_map, clabel, (color="gray", width=1, dash="solid"))
        addtraces!(sf,
            scatter(x=θ_deg, y=[dB[q] for q in cut_idx],
                    mode="lines", name=clabel,
                    line=attr(color=s.color, width=s.width, dash=s.dash));
            row=1, col=1)
    end
    p = sf.plot
    relayout!(p,
        xaxis=attr(title="θ (deg)", range=[0, 180]),
        yaxis=attr(title="Bistatic RCS (dBsm)"),
        legend=attr(x=0.01, y=0.01, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.8)", font=attr(size=11)),
        margin=attr(l=60, r=30, t=60, b=50))
    return p
end

p0 = make_rcs_plot(phi0_idx, θ_deg_0,
    "Aircraft RCS φ=0° — Dense vs ACA vs MLFMA vs PO ($(freq/1e9) GHz, N=$N)")
PlotlyKaleido.savefig(p0, joinpath(figdir, "10_aircraft_rcs_phi0.png"); width=950, height=600)
println("Plot saved: 10_aircraft_rcs_phi0.png")

p90 = make_rcs_plot(phi90_idx, θ_deg_90,
    "Aircraft RCS φ=90° — Dense vs ACA vs MLFMA vs PO ($(freq/1e9) GHz, N=$N)")
PlotlyKaleido.savefig(p90, joinpath(figdir, "10_aircraft_rcs_phi90.png"); width=950, height=600)
println("Plot saved: 10_aircraft_rcs_phi90.png")

println("\n" * "="^72)
println("Done.")
