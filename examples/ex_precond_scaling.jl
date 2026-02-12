# ex_precond_scaling.jl — Solver scaling benchmark: direct LU vs GMRES
#
# Sweeps N (via increasing mesh size) and compares:
#   - Direct LU solve time
#   - GMRES (no preconditioner) iterations and time
#   - GMRES + near-field sparse preconditioner (various cutoffs)
#   - GMRES + randomized preconditioner (for comparison)
#
# Tests both impedance-loaded EFIE (metasurface) and PEC EFIE (general MoM).
#
# Key findings:
#   1. Near-field preconditioner gives N-independent iteration counts
#   2. Randomized subspace preconditioner increases iterations (counterproductive)
#   3. GMRES crossover vs LU occurs at N ≈ 200-300
#
# Run: julia --project=. examples/ex_precond_scaling.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames
using Random

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

println("="^60)
println("Solver Scaling Benchmark: Direct LU vs GMRES")
println("="^60)

# Problem parameters
freq = 3e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668

# Mesh refinement levels (increasing N)
mesh_sizes = [3, 4, 6, 8, 10, 12, 16]

# Near-field cutoffs (in wavelengths)
nf_cutoffs_lam = [0.5, 1.0, 2.0]

# Plane wave excitation
k_vec = Vec3(0.0, 0.0, -k)
E0    = 1.0
pol_inc = Vec3(1.0, 0.0, 0.0)

# GMRES parameters
gmres_tol = 1e-8
gmres_maxiter = 500

results = DataFrame(
    Nx       = Int[],
    N_rwg    = Int[],
    problem  = String[],
    method   = String[],
    time_s   = Float64[],
    iters    = Int[],
    rel_err  = Float64[],
    residual = Float64[],
)

# ─────────────────────────────────────────────────
# Part 1: Impedance-loaded EFIE
# ─────────────────────────────────────────────────
println("\n── Part 1: Impedance-Loaded EFIE ──\n")

for (idx, Nx) in enumerate(mesh_sizes)
    Ny = Nx
    Lx = 4 * lambda0
    Ly = Lx

    mesh = make_rect_plate(Lx, Ly, Nx, Ny)
    rwg  = build_rwg(mesh)
    N    = rwg.nedges
    Nt   = ntriangles(mesh)

    theta_test = fill(200.0, Nt)
    partition = PatchPartition(collect(1:Nt), Nt)
    Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

    Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
    Z_full = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_test))
    v = Vector{ComplexF64}(assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3))

    println("  Nx=$Nx  N=$N  Nt=$Nt")

    # JIT warmup
    if idx == 1
        global _ = Z_full \ v
        global _, _ = solve_gmres(Z_full, v; tol=gmres_tol, maxiter=gmres_maxiter)
    end

    # Direct LU
    t_direct = @elapsed I_direct = Z_full \ v
    res_direct = norm(Z_full * I_direct - v) / norm(v)
    push!(results, (Nx=Nx, N_rwg=N, problem="impedance",
                     method="direct_LU", time_s=t_direct, iters=0,
                     rel_err=0.0, residual=res_direct))
    println("    Direct LU:   $(round(t_direct*1000, sigdigits=3))ms")

    # GMRES (no preconditioner)
    t_nop = @elapsed begin
        I_nop, stats_nop = solve_gmres(Z_full, v; tol=gmres_tol, maxiter=gmres_maxiter)
    end
    rel_nop = norm(I_nop - I_direct) / max(norm(I_direct), 1e-30)
    res_nop = norm(Z_full * I_nop - v) / norm(v)
    push!(results, (Nx=Nx, N_rwg=N, problem="impedance",
                     method="gmres_noprecond", time_s=t_nop, iters=stats_nop.niter,
                     rel_err=rel_nop, residual=res_nop))
    println("    GMRES (no P): $(round(t_nop*1000, sigdigits=3))ms  iters=$(stats_nop.niter)")

    # Near-field preconditioner
    for nf_lam in nf_cutoffs_lam
        cutoff = nf_lam * lambda0
        t_nf = @elapsed begin
            P_nf = build_nearfield_preconditioner(Z_full, mesh, rwg, cutoff)
            I_nf, stats_nf = solve_gmres(Z_full, v;
                                           preconditioner=P_nf, tol=gmres_tol, maxiter=gmres_maxiter)
        end
        rel_nf = norm(I_nf - I_direct) / max(norm(I_direct), 1e-30)
        res_nf = norm(Z_full * I_nf - v) / norm(v)
        push!(results, (Nx=Nx, N_rwg=N, problem="impedance",
                         method="gmres_nf_$(nf_lam)lam", time_s=t_nf, iters=stats_nf.niter,
                         rel_err=rel_nf, residual=res_nf))
        println("    NF $(nf_lam)λ:    $(round(t_nf*1000, sigdigits=3))ms  iters=$(stats_nf.niter)  nnz=$(round(P_nf.nnz_ratio*100, digits=1))%")
    end

    # Randomized (k=20, left) for comparison
    k_eff = min(20, N)
    t_rp = @elapsed begin
        P_rp = build_randomized_preconditioner(Z_full, k_eff; seed=42)
        I_rp, stats_rp = solve_gmres(Z_full, v; preconditioner=P_rp, tol=gmres_tol, maxiter=gmres_maxiter)
    end
    rel_rp = norm(I_rp - I_direct) / max(norm(I_direct), 1e-30)
    push!(results, (Nx=Nx, N_rwg=N, problem="impedance",
                     method="gmres_rand_k$(k_eff)", time_s=t_rp, iters=stats_rp.niter,
                     rel_err=rel_rp, residual=norm(Z_full * I_rp - v) / norm(v)))
    println("    Rand k=$k_eff:  $(round(t_rp*1000, sigdigits=3))ms  iters=$(stats_rp.niter)")

    println()
end

# ─────────────────────────────────────────────────
# Part 2: PEC EFIE (no impedance loading)
# ─────────────────────────────────────────────────
println("\n── Part 2: PEC EFIE (no impedance) ──\n")

for (idx, Nx) in enumerate(mesh_sizes)
    Ny = Nx
    Lx = 4 * lambda0
    Ly = Lx

    mesh = make_rect_plate(Lx, Ly, Nx, Ny)
    rwg  = build_rwg(mesh)
    N    = rwg.nedges

    Z_pec = Matrix{ComplexF64}(assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0))
    v = Vector{ComplexF64}(assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3))

    println("  Nx=$Nx  N=$N")

    # Direct LU
    t_direct = @elapsed I_direct = Z_pec \ v
    res_direct = norm(Z_pec * I_direct - v) / norm(v)
    push!(results, (Nx=Nx, N_rwg=N, problem="pec",
                     method="direct_LU", time_s=t_direct, iters=0,
                     rel_err=0.0, residual=res_direct))
    println("    Direct LU:   $(round(t_direct*1000, sigdigits=3))ms")

    # GMRES (no preconditioner)
    t_nop = @elapsed begin
        I_nop, stats_nop = solve_gmres(Z_pec, v; tol=gmres_tol, maxiter=gmres_maxiter)
    end
    rel_nop = norm(I_nop - I_direct) / max(norm(I_direct), 1e-30)
    push!(results, (Nx=Nx, N_rwg=N, problem="pec",
                     method="gmres_noprecond", time_s=t_nop, iters=stats_nop.niter,
                     rel_err=rel_nop, residual=norm(Z_pec * I_nop - v) / norm(v)))
    println("    GMRES (no P): $(round(t_nop*1000, sigdigits=3))ms  iters=$(stats_nop.niter)")

    # Near-field preconditioner
    for nf_lam in nf_cutoffs_lam
        cutoff = nf_lam * lambda0
        t_nf = @elapsed begin
            P_nf = build_nearfield_preconditioner(Z_pec, mesh, rwg, cutoff)
            I_nf, stats_nf = solve_gmres(Z_pec, v;
                                           preconditioner=P_nf, tol=gmres_tol, maxiter=gmres_maxiter)
        end
        rel_nf = norm(I_nf - I_direct) / max(norm(I_direct), 1e-30)
        push!(results, (Nx=Nx, N_rwg=N, problem="pec",
                         method="gmres_nf_$(nf_lam)lam", time_s=t_nf, iters=stats_nf.niter,
                         rel_err=rel_nf, residual=norm(Z_pec * I_nf - v) / norm(v)))
        println("    NF $(nf_lam)λ:    $(round(t_nf*1000, sigdigits=3))ms  iters=$(stats_nf.niter)")
    end

    # Randomized (k=20) for comparison
    k_eff = min(20, N)
    t_rp = @elapsed begin
        P_rp = build_randomized_preconditioner(Z_pec, k_eff; seed=42)
        I_rp, stats_rp = solve_gmres(Z_pec, v; preconditioner=P_rp, tol=gmres_tol, maxiter=gmres_maxiter)
    end
    push!(results, (Nx=Nx, N_rwg=N, problem="pec",
                     method="gmres_rand_k$(k_eff)", time_s=t_rp, iters=stats_rp.niter,
                     rel_err=norm(I_rp - I_direct) / max(norm(I_direct), 1e-30),
                     residual=norm(Z_pec * I_rp - v) / norm(v)))
    println("    Rand k=$k_eff:  $(round(t_rp*1000, sigdigits=3))ms  iters=$(stats_rp.niter)")

    println()
end

# Save results
CSV.write(joinpath(DATADIR, "precond_scaling_benchmark.csv"), results)

println("="^60)
println("SCALING BENCHMARK COMPLETE")
println("="^60)
println("\nResults saved to: $(joinpath(DATADIR, "precond_scaling_benchmark.csv"))")

# Summary: iteration count comparison
for prob in ["impedance", "pec"]
    prob_name = prob == "impedance" ? "Impedance-Loaded EFIE" : "PEC EFIE"
    println("\n── $prob_name: Iteration Count ──")
    println(rpad("N", 6), rpad("no_P", 7), rpad("nf0.5λ", 7), rpad("nf1.0λ", 7), rpad("nf2.0λ", 7), rpad("rand", 7))
    for Nx in mesh_sizes
        r_nop = filter(r -> r.Nx == Nx && r.problem == prob && r.method == "gmres_noprecond", results)
        n_rwg = nrow(r_nop) > 0 ? r_nop[1, :N_rwg] : 0
        i_nop = nrow(r_nop) > 0 ? r_nop[1, :iters] : -1

        iters = Int[i_nop]
        for c in nf_cutoffs_lam
            r = filter(r -> r.Nx == Nx && r.problem == prob && r.method == "gmres_nf_$(c)lam", results)
            push!(iters, nrow(r) > 0 ? r[1, :iters] : -1)
        end
        r_rp = filter(r -> r.Nx == Nx && r.problem == prob && contains(r.method, "rand"), results)
        push!(iters, nrow(r_rp) > 0 ? r_rp[1, :iters] : -1)

        println(rpad(n_rwg, 6), [rpad(i, 7) for i in iters]...)
    end
end
