# plot.jl — Main paper figures for DifferentiableMoM
#
# Produces publication-ready figures following plotting_style.md conventions.
# Run: julia plot.jl

using CSV, DataFrames, Plots, Printf
gr()

# Global defaults (plotting_style.md)
default(
    linewidth    = 2,
    framestyle   = :box,
    grid         = true,
    minorgrid    = false,
    legendfontsize = 10,
    guidefontsize  = 12,
    tickfontsize   = 10,
    titlefontsize  = 12,
    legend_background_color = RGBA(1,1,1,0.85),
    markersize   = 5,
    left_margin  = 3Plots.mm,
    right_margin = 3Plots.mm,
    bottom_margin = 3Plots.mm,
    top_margin   = 2Plots.mm,
)

mkpath("figs")

# ─── Figure 1: Mesh convergence — energy ratio ──────────────────────────────
function fig_convergence_energy(; outpath="figs")
    df = CSV.read("data/convergence_study.csv", DataFrame)
    p = plot(
        df.N_rwg, df.energy_ratio,
        marker     = :circle,
        color      = :blue,
        label      = raw"$P_{\mathrm{rad}}/P_{\mathrm{in}}$",
        xlabel     = raw"$N$ (number of RWG basis functions)",
        ylabel     = raw"Energy ratio $P_{\mathrm{rad}}/P_{\mathrm{in}}$",
        ylims      = (0.95, 1.05),
        size       = (600, 400),
        legend     = :topright,
    )
    hline!([1.0], linestyle=:dash, color=:black, label="Ideal (1.0)")
    savefig(p, joinpath(outpath, "fig_convergence_energy.pdf"))
    println("  Saved fig_convergence_energy.pdf")
    return p
end

# ─── Figure 2: Mesh convergence — condition number ──────────────────────────
function fig_convergence_cond(; outpath="figs")
    df = CSV.read("data/convergence_study.csv", DataFrame)

    # Fit log-log slope
    x = log.(df.N_rwg)
    y = log.(df.cond_Z)
    n = length(x)
    slope = (n * sum(x .* y) - sum(x)*sum(y)) / (n * sum(x.^2) - sum(x)^2)

    p = plot(
        df.N_rwg, df.cond_Z,
        marker     = :square,
        color      = :red,
        label      = "\$\\kappa(\\mathbf{Z})\$, slope \$\\approx $(round(slope, digits=2))\$",
        xlabel     = raw"$N$ (number of RWG basis functions)",
        ylabel     = raw"Condition number $\kappa(\mathbf{Z})$",
        xscale     = :log10,
        yscale     = :log10,
        size       = (600, 400),
        legend     = :topleft,
    )
    savefig(p, joinpath(outpath, "fig_convergence_cond.pdf"))
    println("  Saved fig_convergence_cond.pdf")
    return p
end

# ─── Figure 3: Gradient verification scatter ─────────────────────────────────
function fig_gradient_verification(; outpath="figs")
    df = CSV.read("data/gradient_verification.csv", DataFrame)

    # Scale to units of 1e-7 for clean tick labels
    scale = 1e7
    adj_s = df.adjoint .* scale
    fd_s  = df.fd_central .* scale
    lo_s  = min(minimum(adj_s), minimum(fd_s))
    hi_s  = max(maximum(adj_s), maximum(fd_s))
    margin_s = 0.1 * (hi_s - lo_s)

    p = plot(
        fd_s, adj_s,
        seriestype = :scatter,
        marker     = :circle,
        markersize = 6,
        color      = :blue,
        label      = "Adjoint vs FD",
        xlabel     = raw"FD gradient $\partial J / \partial \theta_p$ ($\times 10^{-7}$)",
        ylabel     = raw"Adjoint gradient $\partial J / \partial \theta_p$ ($\times 10^{-7}$)",
        aspect_ratio = 1,
        size       = (500, 500),
        legend     = :topleft,
    )
    plot!([lo_s - margin_s, hi_s + margin_s], [lo_s - margin_s, hi_s + margin_s],
          linestyle=:dash, color=:black, label="Perfect agreement")
    savefig(p, joinpath(outpath, "fig_gradient_verification.pdf"))
    println("  Saved fig_gradient_verification.pdf")
    return p
end

# ─── Figure 4: Beam-steering optimization convergence ────────────────────────
function fig_beam_steer_trace(; outpath="figs")
    df = CSV.read("data/beam_steer_trace.csv", DataFrame)

    p = plot(
        df.iter, df.J .* 100,
        color      = :blue,
        marker     = :none,
        label      = "",
        xlabel     = "Iteration",
        ylabel     = raw"Directivity fraction $J$ (\%)",
        size       = (600, 400),
    )
    savefig(p, joinpath(outpath, "fig_beam_steer_trace.pdf"))
    println("  Saved fig_beam_steer_trace.pdf")
    return p
end

# ─── Figure 5: Beam-steering far-field cut (phi=0) ──────────────────────────
function fig_beam_steer_farfield(; outpath="figs")
    df = CSV.read("data/beam_steer_cut_phi0.csv", DataFrame)
    df = df[df.theta_deg .<= 90.0, :]

    p = plot(
        df.theta_deg, df.dir_pec_dBi,
        color      = :blue,
        linestyle  = :dash,
        label      = "PEC (no impedance)",
        xlabel     = raw"$\theta$ (degrees)",
        ylabel     = "Directivity (dBi)",
        xlims      = (0, 90),
        size       = (600, 400),
        legend     = :topright,
    )
    plot!(df.theta_deg, df.dir_opt_dBi,
          color=:red, linestyle=:solid, label="Optimized impedance")

    # Mark target angle
    vline!([30.0], linestyle=:dot, color=:green, label=raw"Target $\theta_s = 30°$")
    savefig(p, joinpath(outpath, "fig_beam_steer_farfield.pdf"))
    println("  Saved fig_beam_steer_farfield.pdf")
    return p
end

# ─── Figure 6: Optimized impedance distribution ─────────────────────────────
function fig_impedance_distribution(; outpath="figs")
    df = CSV.read("data/beam_steer_impedance.csv", DataFrame)

    p = scatter(
        df.cx, df.cy,
        zcolor     = df.theta_opt,
        markershape = :square,
        markersize = 8,
        markerstrokewidth = 0,
        colorbar_title = raw"$\mathrm{Im}(Z_s)\,(\Omega)$",
        colorbar_titlefont = font(9),
        xlabel     = raw"$x$ (m)",
        ylabel     = raw"$y$ (m)",
        label      = "",
        aspect_ratio = 1,
        size       = (640, 500),
        right_margin = 10Plots.mm,
        color      = :balance,
    )
    savefig(p, joinpath(outpath, "fig_impedance_distribution.pdf"))
    println("  Saved fig_impedance_distribution.pdf")
    return p
end

# ─── Table generation ─────────────────────────────────────────────────────────
function print_convergence_table()
    df = CSV.read("data/convergence_study.csv", DataFrame)
    println("\n=== Table: Mesh Convergence Study ===")
    println("Nx  | N_v | N_t | N   | κ(Z)    | P_rad/P_in | max |∇err|")
    println("----|-----|-----|-----|---------|------------|----------")
    for row in eachrow(df)
        @printf("%d   | %3d | %3d | %3d | %7.1f | %10.4f | %.1e\n",
                row.Nx, row.Nv, row.Nt, row.N_rwg, row.cond_Z,
                row.energy_ratio, row.max_grad_err)
    end
end

function print_gradient_table()
    df = CSV.read("data/gradient_verification.csv", DataFrame)
    println("\n=== Table: Gradient Verification (Adjoint vs Central FD) ===")
    println("p  | Adjoint           | FD (central)      | Rel. Error")
    println("---|-------------------|-------------------|-----------")
    for row in eachrow(df)
        @printf("%2d | %17.10e | %17.10e | %.2e\n",
                row.param_idx, row.adjoint, row.fd_central, row.rel_error)
    end
end

# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    println("Generating main paper figures...")

    fig_convergence_energy()
    fig_convergence_cond()
    fig_gradient_verification()
    fig_beam_steer_trace()
    fig_beam_steer_farfield()
    fig_impedance_distribution()

    print_convergence_table()
    print_gradient_table()

    println("\nAll figures saved to figs/")
end

main()
