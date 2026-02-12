# ex_precond_recycling.jl — Preconditioner recycling analysis
#
# Tracks the benefit of preconditioner recycling across optimization iterations:
#   1. Fresh preconditioner every iteration (full O(N²k) rebuild)
#   2. Recycled preconditioner with cached MpΩ (O(Nk²) update)
#   3. No preconditioner update (stale preconditioner)
#
# Shows GMRES iteration counts, solve times, and solution accuracy.
#
# Run: julia --project=. examples/ex_precond_recycling.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

println("="^60)
println("Preconditioner Recycling Analysis")
println("="^60)

# Problem setup
freq = 3e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668

Lx, Ly = 0.2, 0.2   # 2λ × 2λ
Nx, Ny = 6, 6

mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges
Nt   = ntriangles(mesh)

println("\n── Setup ──")
println("  Mesh: $Nx×$Ny  N=$N  Nt=$Nt")

# Assembly
Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

k_vec = Vec3(0.0, 0.0, -k)
v = Vector{ComplexF64}(assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, Vec3(1.0, 0.0, 0.0); quad_order=3))

precond_rank = 15
gmres_tol = 1e-8
gmres_maxiter = 500

# Simulate an optimization trajectory by stepping theta
n_steps = 20
step_size = 15.0  # impedance change per step

theta_trajectory = Vector{Vector{Float64}}()
theta_current = fill(200.0, Nt)
push!(theta_trajectory, copy(theta_current))
for step in 1:n_steps
    global theta_current
    delta = randn(Nt) .* step_size
    theta_current = theta_current .+ delta
    push!(theta_trajectory, copy(theta_current))
end

println("  Trajectory: $(n_steps+1) steps, step size ≈ $step_size Ω")

# Build initial Z and preconditioner
Z_init = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_trajectory[1]))
P_init = build_randomized_preconditioner(Z_init, precond_rank; seed=42)
mp_omega_cache = cache_MpOmega(Mp, P_init.Omega)

# Track results
results = DataFrame(
    step          = Int[],
    strategy      = String[],
    gmres_iters   = Int[],
    precond_time  = Float64[],
    solve_time    = Float64[],
    total_time    = Float64[],
    sol_rel_err   = Float64[],
)

# State for recycled strategies
P_recycled_inc = P_init   # incremental recycling
P_recycled_stale = P_init # never updated (stale)

println("\n── Running trajectory ──\n")

for step in 1:(n_steps+1)
    global P_recycled_inc
    theta = theta_trajectory[step]
    Z = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta))
    I_direct = Z \ v  # ground truth

    delta_theta = step > 1 ? theta .- theta_trajectory[step-1] : zeros(Nt)

    # --- Strategy 1: Fresh preconditioner (full rebuild) ---
    t_fresh_precond = @elapsed P_fresh = build_randomized_preconditioner(Z, precond_rank; seed=42)
    t_fresh_solve = @elapsed I_fresh, stats_fresh = solve_gmres(Z, v;
        preconditioner=P_fresh, tol=gmres_tol, maxiter=gmres_maxiter)
    err_fresh = norm(I_fresh - I_direct) / max(norm(I_direct), 1e-30)

    push!(results, (step=step, strategy="fresh",
                     gmres_iters=stats_fresh.niter,
                     precond_time=t_fresh_precond,
                     solve_time=t_fresh_solve,
                     total_time=t_fresh_precond + t_fresh_solve,
                     sol_rel_err=err_fresh))

    # --- Strategy 2: Recycled preconditioner (incremental update) ---
    t_recycle_precond = 0.0
    if step > 1
        t_recycle_precond = @elapsed P_recycled_inc = update_randomized_preconditioner(
            P_recycled_inc, Z, delta_theta, Mp; MpOmega=mp_omega_cache)
    end
    t_recycle_solve = @elapsed I_recycled, stats_recycled = solve_gmres(Z, v;
        preconditioner=P_recycled_inc, tol=gmres_tol, maxiter=gmres_maxiter)
    err_recycled = norm(I_recycled - I_direct) / max(norm(I_direct), 1e-30)

    push!(results, (step=step, strategy="recycled",
                     gmres_iters=stats_recycled.niter,
                     precond_time=t_recycle_precond,
                     solve_time=t_recycle_solve,
                     total_time=t_recycle_precond + t_recycle_solve,
                     sol_rel_err=err_recycled))

    # --- Strategy 3: Stale preconditioner (no update) ---
    t_stale_solve = @elapsed I_stale, stats_stale = solve_gmres(Z, v;
        preconditioner=P_recycled_stale, tol=gmres_tol, maxiter=gmres_maxiter)
    err_stale = norm(I_stale - I_direct) / max(norm(I_direct), 1e-30)

    push!(results, (step=step, strategy="stale",
                     gmres_iters=stats_stale.niter,
                     precond_time=0.0,
                     solve_time=t_stale_solve,
                     total_time=t_stale_solve,
                     sol_rel_err=err_stale))

    if step % 5 == 1 || step == n_steps + 1
        println("  Step $step:")
        println("    Fresh:    iters=$(stats_fresh.niter)  err=$(round(err_fresh, sigdigits=3))  " *
                "precond=$(round(t_fresh_precond*1000, sigdigits=3))ms  solve=$(round(t_fresh_solve*1000, sigdigits=3))ms")
        println("    Recycled: iters=$(stats_recycled.niter)  err=$(round(err_recycled, sigdigits=3))  " *
                "precond=$(round(t_recycle_precond*1000, sigdigits=3))ms  solve=$(round(t_recycle_solve*1000, sigdigits=3))ms")
        println("    Stale:    iters=$(stats_stale.niter)  err=$(round(err_stale, sigdigits=3))  " *
                "solve=$(round(t_stale_solve*1000, sigdigits=3))ms")
    end
end

# Save
CSV.write(joinpath(DATADIR, "precond_recycling_analysis.csv"), results)

# Summary statistics
println("\n── Summary ──")
for strat in ["fresh", "recycled", "stale"]
    rows = filter(r -> r.strategy == strat, results)
    mean_iters = sum(rows.gmres_iters) / nrow(rows)
    mean_total = sum(rows.total_time) / nrow(rows)
    max_err = maximum(rows.sol_rel_err)
    println("  $(rpad(strat, 10))  mean_iters=$(round(mean_iters, digits=1))  " *
            "mean_time=$(round(mean_total*1000, sigdigits=3))ms  max_err=$(round(max_err, sigdigits=3))")
end

# Speedup analysis
fresh_total = sum(filter(r -> r.strategy == "fresh", results).total_time)
recycled_total = sum(filter(r -> r.strategy == "recycled", results).total_time)
println("\n  Total time (fresh): $(round(fresh_total, sigdigits=4))s")
println("  Total time (recycled): $(round(recycled_total, sigdigits=4))s")
println("  Recycling speedup: $(round(fresh_total / recycled_total, sigdigits=3))x")

println("\n" * "="^60)
println("RECYCLING ANALYSIS COMPLETE")
println("="^60)
println("Results saved to: $(joinpath(DATADIR, "precond_recycling_analysis.csv"))")
