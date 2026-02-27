# 16_periodic_to_multimode_demo.jl — Additional TAP-strengthening demo in a multi-mode periodic regime
#
# Demonstrates the same density-based periodic MoM topology-optimization workflow
# on a larger unit cell (d ≈ 1.2 λ) where multiple Floquet modes propagate.
# The objective remains cone-integrated specular suppression; the results report
# redistribution across propagating Floquet orders.
#
# Run: julia --project=. examples/16_periodic_to_multimode_demo.jl

using DifferentiableMoM
using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using Random
using CSV, DataFrames
using PlotlySupply
using PlotlyKaleido
PlotlyKaleido.start(mathjax=false)

Random.seed!(4242)

const PKG_DIR  = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "..", "data")
const FIG_DIR  = joinpath(PKG_DIR, "..", "figures")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

println("=" ^ 70)
println("  Periodic TO Demo (Multi-Mode Unit Cell, TAP Upgrade)")
println("=" ^ 70)

# ────────────────────────────────────────────────────────────────────────────
# Problem setup (runtime-conscious but physically richer than λ/2 case)
# ────────────────────────────────────────────────────────────────────────────
freq   = 10e9
c0     = 3e8
lambda = c0 / freq
k      = 2π / lambda
eta0   = 376.730313668

# d = 1.2 λ gives 5 propagating modes at normal incidence: (0,0), (±1,0), (0,±1)
dx_cell = 1.2 * lambda
dy_cell = 1.2 * lambda

# Keep runtime manageable; can override for stronger runs.
Nx = parse(Int, get(ENV, "DMOM_MM_NX", "12"))
Ny = parse(Int, get(ENV, "DMOM_MM_NY", string(Nx)))
iters_per_beta = parse(Int, get(ENV, "DMOM_MM_ITERS_PER_BETA", "10"))
betas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

println("  Frequency: $(freq/1e9) GHz, λ = $(round(lambda*1e3, digits=2)) mm")
println("  Unit cell: $(round(dx_cell/lambda, digits=2)) λ × $(round(dy_cell/lambda, digits=2)) λ")
println("  Mesh: $(Nx)×$(Ny) (runtime setting), β schedule=$(Int.(betas)) × $(iters_per_beta)")

mesh = make_rect_plate(dx_cell, dy_cell, Nx, Ny)
rwg  = build_rwg(mesh; precheck=false)
Nt = ntriangles(mesh)
N  = rwg.nedges
println("  Discretization: $(Nt) triangles, $(N) RWG edges")

lattice = PeriodicLattice(dx_cell, dy_cell, 0.0, 0.0, k)

println("  Assembling periodic EFIE matrix (dense)...")
Z_per = Matrix{ComplexF64}(assemble_Z_efie_periodic(mesh, rwg, k, lattice))
println("  Z_per: $(size(Z_per,1)) × $(size(Z_per,2))")

Mt = precompute_triangle_mass(mesh, rwg)

edge_len = dx_cell / Nx
r_min = 2.5 * edge_len
W, w_sum = build_filter_weights(mesh, r_min)

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v  = Vector{ComplexF64}(assemble_excitation(mesh, rwg, pw))

Ntheta_ff, Nphi_ff = 20, 40
spec_half_angle = 10 * π / 180
grid_ff = make_sph_grid(Ntheta_ff, Nphi_ff)
Q_spec = Matrix{ComplexF64}(specular_rcs_objective(
    mesh, rwg, grid_ff, k, lattice; half_angle=spec_half_angle, polarization=:x))

config = DensityConfig(; p=3.0, Z_max_factor=10.0, vf_target=0.5)
alpha_vf = 0.1
println("  Objective: cone-integrated specular Q on $(Ntheta_ff)×$(Nphi_ff), half-angle=10°")
println("  SIMP: p=$(config.p), Z_max=$(round(config.Z_max, sigdigits=4)), α_vf=$(alpha_vf)")

centroids = [triangle_center(mesh, t) for t in 1:Nt]

# ────────────────────────────────────────────────────────────────────────────
# Helper functions (copied from ex15 with minimal changes)
# ────────────────────────────────────────────────────────────────────────────
function run_optimization(Z_per::Matrix{ComplexF64}, Mt, v, Q, config, W, w_sum, rho0;
                          betas=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
                          iters_per_beta=10, alpha_vf=1.0, m_lbfgs=10, verbose=true)
    Nt_loc = length(rho0)
    vf_target = config.vf_target
    rho = copy(rho0)
    project!(x) = (x .= clamp.(x, 0.0, 1.0); x)

    trace = DataFrame(iter=Int[], beta=Float64[], J_scatter=Float64[],
                      J_vf=Float64[], J_total=Float64[], gnorm=Float64[],
                      vf=Float64[], frac_binary=Float64[])
    snapshots = Dict{Float64, Vector{Float64}}()
    global_iter = 0

    for beta in betas
        verbose && println("\n  ── β = $(Int(beta)) ──")
        _, rb_snap = filter_and_project(W, w_sum, rho, beta)
        snapshots[beta] = copy(rb_snap)

        s_list = Vector{Float64}[]
        y_list = Vector{Float64}[]
        rho_old = copy(rho)
        g_old   = zeros(Nt_loc)

        for it in 1:iters_per_beta
            global_iter += 1

            rho_tilde, rho_bar = filter_and_project(W, w_sum, rho, beta)
            Z_total = Z_per + assemble_Z_penalty(Mt, rho_bar, config)
            F = lu(Z_total)
            I_c = F \ v

            QI_c = Q * I_c
            J_scat = real(dot(I_c, QI_c))
            vf_cur = mean(rho_bar)
            J_vf = alpha_vf * (vf_cur - vf_target)^2
            J_tot = J_scat + J_vf

            lam = F' \ QI_c
            g_rb = gradient_density(Mt, Vector{ComplexF64}(I_c), Vector{ComplexF64}(lam), rho_bar, config)
            g_rb .+= alpha_vf * 2.0 * (vf_cur - vf_target) / Nt_loc
            g = gradient_chain_rule(g_rb, rho_tilde, W, w_sum, beta)

            gnorm = norm(g)
            fb = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt_loc
            push!(trace, (global_iter, beta, J_scat, J_vf, J_tot, gnorm, vf_cur, fb))

            if verbose && (it <= 2 || it == iters_per_beta || it % 5 == 0)
                println("    it=$global_iter J=$(round(J_tot, sigdigits=5)) |g|=$(round(gnorm, sigdigits=3)) vf=$(round(vf_cur, digits=3))")
            end

            if it > 1
                s_k = rho .- rho_old
                y_k = g .- g_old
                if dot(s_k, y_k) > 1e-30
                    push!(s_list, copy(s_k)); push!(y_list, copy(y_k))
                    length(s_list) > m_lbfgs && (popfirst!(s_list); popfirst!(y_list))
                end
            end

            q_v = copy(g)
            mc = length(s_list)
            av = zeros(mc)
            for i in mc:-1:1
                ri = 1.0 / dot(y_list[i], s_list[i])
                av[i] = ri * dot(s_list[i], q_v)
                q_v .-= av[i] .* y_list[i]
            end
            gamma = mc > 0 ? dot(s_list[end], y_list[end]) / dot(y_list[end], y_list[end]) : 0.01
            r_v = gamma .* q_v
            for i in 1:mc
                ri = 1.0 / dot(y_list[i], s_list[i])
                bv = ri * dot(y_list[i], r_v)
                r_v .+= (av[i] - bv) .* s_list[i]
            end

            d = -r_v

            alpha_ls = 1.0
            rho_old .= rho
            g_old .= g
            accepted = false
            for ls in 1:20
                rho_trial = project!(rho_old .+ alpha_ls .* d)
                _, rb = filter_and_project(W, w_sum, rho_trial, beta)
                I_t = (Z_per + assemble_Z_penalty(Mt, rb, config)) \ v
                Jt = real(dot(I_t, Q * I_t)) + alpha_vf * (mean(rb) - vf_target)^2
                if Jt <= J_tot + 1e-4 * alpha_ls * dot(g, d)
                    rho .= rho_trial
                    accepted = true
                    break
                end
                alpha_ls *= 0.5
            end
            !accepted && (rho .= project!(rho_old .+ alpha_ls .* d))
        end
    end

    _, rb_final = filter_and_project(W, w_sum, rho, betas[end])
    snapshots[Inf] = copy(rb_final)
    return rho, trace, snapshots
end

function density_to_grid(rho_bar_vec, cents, nx, ny, dxc, dyc)
    grid_rho = zeros(ny, nx)
    grid_cnt = zeros(Int, ny, nx)
    xs = [c[1] for c in cents]; ys = [c[2] for c in cents]
    x0, y0 = minimum(xs), minimum(ys)
    dxg, dyg = dxc / nx, dyc / ny
    for t in eachindex(rho_bar_vec)
        ix = clamp(floor(Int, (cents[t][1] - x0) / dxg) + 1, 1, nx)
        iy = clamp(floor(Int, (cents[t][2] - y0) / dyg) + 1, 1, ny)
        grid_rho[iy, ix] += rho_bar_vec[t]
        grid_cnt[iy, ix] += 1
    end
    grid_rho ./= max.(grid_cnt, 1)
    return grid_rho
end

function mode_power_fraction(mode, R, k)
    mode.propagating || return 0.0
    return abs2(R) * real(mode.kz) / k
end

# ────────────────────────────────────────────────────────────────────────────
# Reference PEC (multi-mode Floquet content)
# ────────────────────────────────────────────────────────────────────────────
println("\n▸ Reference: full PEC plate")
_, rho_bar_pec = filter_and_project(W, w_sum, ones(Nt), betas[end])
Z_pec = Z_per + assemble_Z_penalty(Mt, rho_bar_pec, config)
F_pec = lu(Z_pec)
I_pec = F_pec \ v
J_pec = real(dot(I_pec, Q_spec * I_pec))

pol_inc = SVector(1.0, 0.0, 0.0)
modes_pec, R_pec = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_pec), k, lattice;
                                           pol=pol_inc, E0=1.0)
prop_idx = findall(m -> m.propagating, modes_pec)
println("  Propagating modes at d=1.2λ: $(length(prop_idx))")
length(prop_idx) > 1 || error("Expected multiple propagating modes for d=1.2λ")

spec_pos = findfirst(i -> modes_pec[i].m == 0 && modes_pec[i].n == 0, eachindex(modes_pec))
spec_pos === nothing && error("Specular (0,0) mode not found")

Z_pen_pec = assemble_Z_penalty(Mt, rho_bar_pec, config)
pb_pec = power_balance(Vector{ComplexF64}(I_pec), Z_pen_pec, dx_cell * dy_cell, k, modes_pec, R_pec)
pec_refl_sum = sum(mode_power_fraction(modes_pec[i], R_pec[i], k) for i in prop_idx)
abs(pb_pec.refl_frac - pec_refl_sum) < 5e-10 || error("PEC reflected-power consistency check failed")
println("  PEC power fractions: refl=$(round(100pb_pec.refl_frac,digits=1))%, abs=$(round(100pb_pec.abs_frac,digits=1))%, resid=$(round(100pb_pec.resid_frac,digits=1))%")

# ────────────────────────────────────────────────────────────────────────────
# Optimization
# ────────────────────────────────────────────────────────────────────────────
println("\n▸ Running multi-mode topology optimization (reduced schedule)")
rho0 = fill(0.5, Nt)
t0 = time()
rho_opt, trace, snapshots = run_optimization(
    Z_per, Mt, v, Q_spec, config, W, w_sum, rho0;
    betas=betas, iters_per_beta=iters_per_beta, alpha_vf=alpha_vf, verbose=true)
println("  Optimization runtime: $(round(time()-t0, digits=1)) s")

_, rho_bar_final = filter_and_project(W, w_sum, rho_opt, betas[end])
Z_opt = Z_per + assemble_Z_penalty(Mt, rho_bar_final, config)
F_opt = lu(Z_opt)
I_opt = F_opt \ v
J_opt = real(dot(I_opt, Q_spec * I_opt))
J_reduction_dB = 10 * log10(max(J_opt, 1e-30) / max(J_pec, 1e-30))
println("  J_spec: PEC=$(round(J_pec,sigdigits=5)) → opt=$(round(J_opt,sigdigits=5)) ($(round(J_reduction_dB,digits=2)) dB)")

modes_opt, R_opt = reflection_coefficients(mesh, rwg, Vector{ComplexF64}(I_opt), k, lattice;
                                           pol=pol_inc, E0=1.0)
prop_idx_opt = findall(m -> m.propagating, modes_opt)
prop_idx_opt == prop_idx || error("Propagating-mode set changed unexpectedly")

Z_pen_opt = assemble_Z_penalty(Mt, rho_bar_final, config)
pb_opt = power_balance(Vector{ComplexF64}(I_opt), Z_pen_opt, dx_cell * dy_cell, k, modes_opt, R_opt)
opt_refl_sum = sum(mode_power_fraction(modes_opt[i], R_opt[i], k) for i in prop_idx_opt)
abs(pb_opt.refl_frac - opt_refl_sum) < 5e-10 || error("Optimized reflected-power consistency check failed")

R00_pec = abs(R_pec[spec_pos])
R00_opt = abs(R_opt[spec_pos])
R00_amp_dB = 20 * log10(max(R00_opt, 1e-30) / max(R00_pec, 1e-30))
println("  |R00|: $(round(R00_pec, sigdigits=4)) → $(round(R00_opt, sigdigits=4)) ($(round(R00_amp_dB,digits=2)) dB)")
println("  Optimized power fractions: refl=$(round(100pb_opt.refl_frac,digits=1))%, abs=$(round(100pb_opt.abs_frac,digits=1))%, resid=$(round(100pb_opt.resid_frac,digits=1))%")

# ────────────────────────────────────────────────────────────────────────────
# Spot-check gradient correctness in the multi-mode regime (subset FD test)
# ────────────────────────────────────────────────────────────────────────────
println("\n▸ Spot-check adjoint gradient in multi-mode regime (subset finite differences)")
rho_gv = 0.3 .+ 0.4 * rand(Nt)
beta_gv = 4.0
rt_gv, rb_gv = filter_and_project(W, w_sum, rho_gv, beta_gv)
Z_gv = Z_per + assemble_Z_penalty(Mt, rb_gv, config)
F_gv = lu(Z_gv)
I_gv = F_gv \ v
lam_gv = F_gv' \ (Q_spec * I_gv)
g_adj = gradient_density_full(Mt, Vector{ComplexF64}(I_gv), Vector{ComplexF64}(lam_gv),
                              rt_gv, rb_gv, config, W, w_sum, beta_gv)
subset = sort(randperm(Nt)[1:min(8, Nt)])
h_fd = 1e-5
rel_errs = Float64[]
for t in subset
    rp = copy(rho_gv); rp[t] += h_fd
    rm = copy(rho_gv); rm[t] -= h_fd
    _, rbp = filter_and_project(W, w_sum, rp, beta_gv)
    _, rbm = filter_and_project(W, w_sum, rm, beta_gv)
    Ip = (Z_per + assemble_Z_penalty(Mt, rbp, config)) \ v
    Im = (Z_per + assemble_Z_penalty(Mt, rbm, config)) \ v
    g_fd = (real(dot(Ip, Q_spec * Ip)) - real(dot(Im, Q_spec * Im))) / (2h_fd)
    push!(rel_errs, abs(g_adj[t] - g_fd) / max(abs(g_fd), 1e-20))
end
max_rel_subset = maximum(rel_errs)
println("  subset triangles = $(subset)")
println("  max relative error (subset) = $(round(max_rel_subset, sigdigits=3))")
max_rel_subset < 1e-6 || error("Multi-mode gradient subset check exceeded tolerance")

# ────────────────────────────────────────────────────────────────────────────
# Save data products
# ────────────────────────────────────────────────────────────────────────────
CSV.write(joinpath(DATA_DIR, "results_multimode_optimization_trace.csv"), trace)
CSV.write(joinpath(DATA_DIR, "results_multimode_topology_final.csv"), DataFrame(
    triangle=1:Nt,
    rho_bar=rho_bar_final,
    x=[c[1] for c in centroids],
    y=[c[2] for c in centroids]))

# Propagating Floquet table (sorted: specular first, then by |m|+|n| and lexicographic)
prop_sorted = sort(prop_idx_opt; by=i -> (modes_opt[i].m == 0 && modes_opt[i].n == 0 ? -1 : 0,
                                          abs(modes_opt[i].m) + abs(modes_opt[i].n),
                                          modes_opt[i].m, modes_opt[i].n))
mode_labels = ["($(modes_opt[i].m),$(modes_opt[i].n))" for i in prop_sorted]
mode_df = DataFrame(
    mode_index = 1:length(prop_sorted),
    mode_label = mode_labels,
    m = [modes_opt[i].m for i in prop_sorted],
    n = [modes_opt[i].n for i in prop_sorted],
    kz_over_k = [real(modes_opt[i].kz) / k for i in prop_sorted],
    R_pec_real = [real(R_pec[i]) for i in prop_sorted],
    R_pec_imag = [imag(R_pec[i]) for i in prop_sorted],
    R_pec_abs = [abs(R_pec[i]) for i in prop_sorted],
    R_opt_real = [real(R_opt[i]) for i in prop_sorted],
    R_opt_imag = [imag(R_opt[i]) for i in prop_sorted],
    R_opt_abs = [abs(R_opt[i]) for i in prop_sorted],
    pfrac_pec = [mode_power_fraction(modes_opt[i], R_pec[i], k) for i in prop_sorted],
    pfrac_opt = [mode_power_fraction(modes_opt[i], R_opt[i], k) for i in prop_sorted],
)
mode_df.R_amp_change_dB = 20 .* log10.(max.(mode_df.R_opt_abs, 1e-30) ./ max.(mode_df.R_pec_abs, 1e-30))
CSV.write(joinpath(DATA_DIR, "results_multimode_floquet.csv"), mode_df)

summary_df = DataFrame(
    freq_GHz=[freq/1e9], lambda_mm=[lambda*1e3],
    dx_over_lambda=[dx_cell/lambda], dy_over_lambda=[dy_cell/lambda],
    Nx=[Nx], Ny=[Ny], Nt=[Nt], N_rwg=[N], nprop=[length(prop_idx_opt)],
    Q_grid_theta=[Ntheta_ff], Q_grid_phi=[Nphi_ff], spec_half_angle_deg=[spec_half_angle*180/π],
    J_pec=[J_pec], J_opt=[J_opt], J_reduction_dB=[J_reduction_dB],
    R00_pec=[R00_pec], R00_opt=[R00_opt], R00_amp_dB=[R00_amp_dB],
    refl_frac_pec=[pb_pec.refl_frac], abs_frac_pec=[pb_pec.abs_frac], resid_frac_pec=[pb_pec.resid_frac],
    refl_frac_opt=[pb_opt.refl_frac], abs_frac_opt=[pb_opt.abs_frac], resid_frac_opt=[pb_opt.resid_frac],
    spec_mode_pfrac_opt=[mode_power_fraction(modes_opt[spec_pos], R_opt[spec_pos], k)],
    nonspec_prop_pfrac_opt=[sum(mode_power_fraction(modes_opt[i], R_opt[i], k) for i in prop_idx_opt if i != spec_pos)],
    gradient_subset_max_rel_err=[max_rel_subset],
)
CSV.write(joinpath(DATA_DIR, "results_multimode_summary.csv"), summary_df)

println("  ✓ Saved: data/results_multimode_optimization_trace.csv")
println("  ✓ Saved: data/results_multimode_topology_final.csv")
println("  ✓ Saved: data/results_multimode_floquet.csv")
println("  ✓ Saved: data/results_multimode_summary.csv")

# ────────────────────────────────────────────────────────────────────────────
# Figures (important plots only)
# ────────────────────────────────────────────────────────────────────────────
println("\n▸ Generating multi-mode figures")

# Fig: propagating Floquet amplitudes (main-paper friendly; avoids over-reading coarse power normalization)
fig_modes = plot_scatter(
    [collect(mode_df.mode_index), collect(mode_df.mode_index)],
    [collect(mode_df.R_pec_abs), collect(mode_df.R_opt_abs)];
    mode=["lines+markers", "lines+markers"],
    legend=["PEC plate", "Optimized"],
    color=["#1f77b4", "#d62728"],
    dash=["solid", "dashdot"],
    marker_size=[8, 8],
    xlabel="Propagating mode index (see CSV order)",
    ylabel="Floquet reflection amplitude |R_mn|",
    title="Multi-mode Floquet amplitudes (d = 1.2λ)",
    width=620, height=420, fontsize=14)
set_legend!(fig_modes; position=:topright)
savefig(fig_modes, joinpath(FIG_DIR, "fig_results_multimode_floquet.pdf"))

# Optional topology visualization (can be main or supplementary)
G_top = density_to_grid(rho_bar_final, centroids, Nx, Ny, dx_cell, dy_cell)
fig_top = plot_heatmap(collect(1:Nx), collect(1:Ny), G_top;
    xlabel="Cell x", ylabel="Cell y",
    zrange=[0, 1], colorscale="Greys",
    width=440, height=420, fontsize=16, equalar=true)
savefig(fig_top, joinpath(FIG_DIR, "fig_results_multimode_topology.pdf"))

# Supplementary convergence trace
betas_uniq = unique(trace.beta)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
x_segs = [collect(trace[trace.beta .== b, :iter]) for b in betas_uniq]
y_segs = [collect(trace[trace.beta .== b, :J_scatter]) for b in betas_uniq]
fig_conv = plot_scatter(x_segs, y_segs;
    mode=fill("lines", length(betas_uniq)),
    legend=["β=$(Int(b))" for b in betas_uniq],
    color=colors[1:length(betas_uniq)],
    dash=["solid", "dash", "dashdot", "dot", "longdash", "longdashdot"][1:length(betas_uniq)],
    xlabel="Iteration", ylabel="Scattered power J",
    title="Multi-mode optimization convergence (supplementary)",
    width=620, height=420, fontsize=14)
set_legend!(fig_conv; position=:topleft)
savefig(fig_conv, joinpath(FIG_DIR, "fig_supp_multimode_convergence.pdf"))

println("  ✓ Fig: figures/fig_results_multimode_floquet.pdf")
println("  ✓ Fig: figures/fig_results_multimode_topology.pdf")
println("  ✓ Supp: figures/fig_supp_multimode_convergence.pdf")

println("\n" * "=" ^ 70)
println("  Multi-Mode Demo Summary")
println("=" ^ 70)
println("  Propagating modes: $(length(prop_idx_opt))")
println("  J reduction (specular objective): $(round(J_reduction_dB, digits=2)) dB")
println("  |R00| amplitude change: $(round(R00_amp_dB, digits=2)) dB")
println("  Reflected power: PEC=$(round(100pb_pec.refl_frac,digits=1))%, opt=$(round(100pb_opt.refl_frac,digits=1))%")
println("  Non-specular propagating reflection (opt): $(round(100sum(mode_power_fraction(modes_opt[i], R_opt[i], k) for i in prop_idx_opt if i != spec_pos), digits=2))%")
println("  Gradient subset max rel error: $(round(max_rel_subset, sigdigits=3))")
println("=" ^ 70)
