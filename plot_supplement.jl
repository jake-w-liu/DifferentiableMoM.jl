# plot_supplement.jl — Supplementary figures for DifferentiableMoM
#
# Produces supplementary figures for additional verification and analysis.
# Run: julia plot_supplement.jl

using CSV, DataFrames, Plots, Printf
gr()

# Global defaults (plotting_style.md)
default(
    linewidth    = 2,
    framestyle   = :box,
    grid         = true,
    minorgrid    = true,
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

posclip(x; lo=1e-20) = max(x, lo)

# ─── Figure S1: Convergence multi-panel ──────────────────────────────────────
function fig_convergence_multipanel(; outpath="figs")
    df = CSV.read("data/convergence_study.csv", DataFrame)

    p1 = plot(df.N_rwg, df.cond_Z,
              marker=:square, color=:red, label="",
              xlabel=raw"$N$", ylabel=raw"$\kappa(\mathbf{Z})$",
              xscale=:log10, yscale=:log10)

    p2 = plot(df.N_rwg, df.energy_ratio,
              marker=:circle, color=:blue, label="",
              xlabel=raw"$N$", ylabel=raw"$P_{\mathrm{rad}}/P_{\mathrm{in}}$")
    hline!([1.0], linestyle=:dash, color=:black, label="")

    p3 = plot(df.N_rwg, posclip.(df.max_grad_err),
              marker=:diamond, color=:green, label="",
              xlabel=raw"$N$", ylabel=raw"Max $|\nabla \mathrm{err}|$",
              xscale=:log10, yscale=:log10)

    p4 = plot(df.N_rwg, df.J_pec,
              marker=:utriangle, color=:orange, label="",
              xlabel=raw"$N$", ylabel=raw"$J_{\mathrm{PEC}}$")

    p = plot(p1, p2, p3, p4, layout=(2,2), size=(900, 700),
             title=["(a) Condition number" "(b) Energy ratio" "(c) Gradient error" "(d) PEC objective"])
    savefig(p, joinpath(outpath, "fig_convergence_multipanel.pdf"))
    println("  Saved fig_convergence_multipanel.pdf")
    return p
end

# ─── Figure S2: PEC far-field pattern (2D heatmap) ──────────────────────────
function fig_pec_farfield_2d(; outpath="figs")
    df = CSV.read("data/farfield_pec.csv", DataFrame)

    theta_vals = sort(unique(df.theta_deg))
    phi_vals   = sort(unique(df.phi_deg))

    power_grid = fill(NaN, length(theta_vals), length(phi_vals))
    for row in eachrow(df)
        i = findfirst(==(row.theta_deg), theta_vals)
        j = findfirst(==(row.phi_deg), phi_vals)
        if i !== nothing && j !== nothing
            power_grid[i, j] = row.power_dB
        end
    end

    p = heatmap(
        phi_vals, theta_vals, power_grid,
        color      = :viridis,
        xlabel     = raw"$\phi$ (degrees)",
        ylabel     = raw"$\theta$ (degrees)",
        colorbar_title = "Power (dB)",
        size       = (650, 500),
        aspect_ratio = :auto,
    )
    savefig(p, joinpath(outpath, "fig_pec_farfield_2d.pdf"))
    println("  Saved fig_pec_farfield_2d.pdf")
    return p
end

# ─── Figure S3: Beam-steering 2D far-field comparison ────────────────────────
function fig_beam_steer_2d(; outpath="figs")
    df = CSV.read("data/beam_steer_farfield.csv", DataFrame)

    theta_vals = sort(unique(df.theta_deg))
    phi_vals   = sort(unique(df.phi_deg))

    pec_grid = fill(NaN, length(theta_vals), length(phi_vals))
    opt_grid = fill(NaN, length(theta_vals), length(phi_vals))
    for row in eachrow(df)
        i = findfirst(==(row.theta_deg), theta_vals)
        j = findfirst(==(row.phi_deg), phi_vals)
        if i !== nothing && j !== nothing
            pec_grid[i, j] = row.dir_pec_dBi
            opt_grid[i, j] = row.dir_opt_dBi
        end
    end

    clims = (min(minimum(skipmissing(pec_grid)), minimum(skipmissing(opt_grid))),
             max(maximum(skipmissing(pec_grid)), maximum(skipmissing(opt_grid))))

    p1 = heatmap(phi_vals, theta_vals, pec_grid,
                  color=:viridis, clims=clims,
                  xlabel=raw"$\phi$ (°)", ylabel=raw"$\theta$ (°)",
                  colorbar_title="dBi", title="PEC")
    p2 = heatmap(phi_vals, theta_vals, opt_grid,
                  color=:viridis, clims=clims,
                  xlabel=raw"$\phi$ (°)", ylabel=raw"$\theta$ (°)",
                  colorbar_title="dBi", title="Optimized")

    p = plot(p1, p2, layout=(1,2), size=(1100, 450))
    savefig(p, joinpath(outpath, "fig_beam_steer_2d.pdf"))
    println("  Saved fig_beam_steer_2d.pdf")
    return p
end

# ─── Figure S4: Resistive optimization trace ─────────────────────────────────
function fig_resistive_trace(; outpath="figs")
    df = CSV.read("data/optimization_trace.csv", DataFrame)

    p1 = plot(df.iter, df.J,
              color=:blue, marker=:circle, label="",
              xlabel="Iteration", ylabel=raw"Objective $J$")
    p2 = plot(df.iter, df.gnorm,
              color=:red, marker=:circle, yscale=:log10, label="",
              xlabel="Iteration", ylabel=raw"$\|\nabla J\|$")

    p = plot(p1, p2, layout=(1,2), size=(900, 380),
             title=["(a) Objective" "(b) Gradient norm"])
    savefig(p, joinpath(outpath, "fig_resistive_trace.pdf"))
    println("  Saved fig_resistive_trace.pdf")
    return p
end

# ─── Figure S5: Gradient verification detail ─────────────────────────────────
function fig_gradient_detail(; outpath="figs")
    df = CSV.read("data/gradient_verification.csv", DataFrame)

    p = bar(
        df.param_idx, posclip.(df.rel_error),
        color      = :orange,
        label      = "",
        xlabel     = "Parameter index",
        ylabel     = "Relative error",
        yscale     = :log10,
        size       = (600, 400),
        xticks     = 1:nrow(df),
    )
    hline!([1e-6], linestyle=:dash, color=:black, label=raw"$10^{-6}$ threshold")
    savefig(p, joinpath(outpath, "fig_gradient_detail.pdf"))
    println("  Saved fig_gradient_detail.pdf")
    return p
end

# ─── Figure S6: Impedance histogram ─────────────────────────────────────────
function fig_impedance_histogram(; outpath="figs")
    df = CSV.read("data/beam_steer_impedance.csv", DataFrame)

    p = histogram(
        df.theta_opt,
        bins       = 30,
        color      = :blue,
        alpha      = 0.7,
        label      = "",
        xlabel     = raw"$\mathrm{Im}(Z_s)$ ($\Omega$)",
        ylabel     = "Count",
        size       = (600, 400),
    )
    vline!([100.0], linestyle=:dash, color=:red, label="Initial value")
    savefig(p, joinpath(outpath, "fig_impedance_histogram.pdf"))
    println("  Saved fig_impedance_histogram.pdf")
    return p
end

# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    println("Generating supplementary figures...")

    fig_convergence_multipanel()
    fig_pec_farfield_2d()
    fig_beam_steer_2d()
    fig_resistive_trace()
    fig_gradient_detail()
    fig_impedance_histogram()

    println("\nAll supplementary figures saved to figs/")
end

main()
