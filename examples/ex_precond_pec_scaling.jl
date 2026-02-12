# ex_precond_pec_scaling.jl — PEC EFIE solver scaling: direct LU vs GMRES
#
# Tests solvers on PURE PEC scattering problems (no impedance loading).
# PEC EFIE matrices are poorly conditioned (cond(Z) grows with N),
# making this the key test case for preconditioning.
#
# Compares:
#   - Direct LU
#   - GMRES (no preconditioner)
#   - GMRES + near-field sparse preconditioner (various cutoffs)
#
# Run: julia --project=. examples/ex_precond_pec_scaling.jl

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
println("PEC EFIE Solver Scaling: Direct LU vs GMRES")
println("="^60)

# Problem parameters
freq = 3e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668

# Mesh refinement levels
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
    method   = String[],
    time_s   = Float64[],
    iters    = Int[],
    rel_err  = Float64[],
    residual = Float64[],
    cond_est = Float64[],
)

println("\n── Running PEC EFIE scaling sweep ──\n")

for (idx, Nx) in enumerate(mesh_sizes)
    Ny = Nx
    Lx = 4 * lambda0
    Ly = Lx

    mesh = make_rect_plate(Lx, Ly, Nx, Ny)
    rwg  = build_rwg(mesh)
    N    = rwg.nedges
    Nt   = ntriangles(mesh)

    # PEC EFIE only — NO impedance loading
    Z_pec = Matrix{ComplexF64}(assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0))
    v = Vector{ComplexF64}(assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3))

    # Condition number
    cond_est = N <= 500 ? cond(Z_pec) : -1.0
    cond_str = cond_est > 0 ? "$(round(cond_est, sigdigits=3))" : "skipped"

    println("  Nx=$Nx  N=$N  Nt=$Nt  cond(Z)≈$cond_str")

    # JIT warmup
    if idx == 1
        global _ = Z_pec \ v
        global _, _ = solve_gmres(Z_pec, v; tol=gmres_tol, maxiter=gmres_maxiter)
    end

    # --- Direct LU ---
    t_direct = @elapsed I_direct = Z_pec \ v
    res_direct = norm(Z_pec * I_direct - v) / norm(v)
    push!(results, (Nx=Nx, N_rwg=N, method="direct_LU",
                     time_s=t_direct, iters=0, rel_err=0.0,
                     residual=res_direct, cond_est=cond_est))
    println("    Direct LU:     $(round(t_direct*1000, sigdigits=3))ms  residual=$(round(res_direct, sigdigits=3))")

    # --- GMRES (no preconditioner) ---
    t_gmres_nop = @elapsed begin
        I_gmres_nop, stats_nop = solve_gmres(Z_pec, v;
                                               tol=gmres_tol, maxiter=gmres_maxiter)
    end
    rel_nop = norm(I_gmres_nop - I_direct) / max(norm(I_direct), 1e-30)
    res_nop = norm(Z_pec * I_gmres_nop - v) / norm(v)
    push!(results, (Nx=Nx, N_rwg=N, method="gmres_noprecond",
                     time_s=t_gmres_nop, iters=stats_nop.niter,
                     rel_err=rel_nop, residual=res_nop, cond_est=cond_est))
    println("    GMRES (no P):  $(round(t_gmres_nop*1000, sigdigits=3))ms  iters=$(stats_nop.niter)  rel_err=$(round(rel_nop, sigdigits=3))")

    # --- GMRES + near-field preconditioner at various cutoffs ---
    for nf_lam in nf_cutoffs_lam
        cutoff = nf_lam * lambda0
        t_nf = @elapsed begin
            P_nf = build_nearfield_preconditioner(Z_pec, mesh, rwg, cutoff)
            I_nf, stats_nf = solve_gmres(Z_pec, v;
                                           preconditioner=P_nf,
                                           tol=gmres_tol, maxiter=gmres_maxiter)
        end
        rel_nf = norm(I_nf - I_direct) / max(norm(I_direct), 1e-30)
        res_nf = norm(Z_pec * I_nf - v) / norm(v)
        push!(results, (Nx=Nx, N_rwg=N, method="gmres_nf_$(nf_lam)lam",
                         time_s=t_nf, iters=stats_nf.niter,
                         rel_err=rel_nf, residual=res_nf, cond_est=cond_est))
        println("    NF $(nf_lam)λ:    $(round(t_nf*1000, sigdigits=3))ms  iters=$(stats_nf.niter)  rel_err=$(round(rel_nf, sigdigits=3))  nnz=$(round(P_nf.nnz_ratio*100, digits=1))%")
    end

    println()
end

# Save results
CSV.write(joinpath(DATADIR, "precond_pec_scaling.csv"), results)

println("="^60)
println("PEC EFIE SCALING BENCHMARK COMPLETE")
println("="^60)
println("\nResults saved to: $(joinpath(DATADIR, "precond_pec_scaling.csv"))")

# Summary comparison
println("\n── Iteration Count Comparison ──")
println(rpad("N", 6), rpad("no_P", 8),
        [rpad("nf_$(c)λ", 8) for c in nf_cutoffs_lam]...)
for Nx in mesh_sizes
    nop = filter(r -> r.Nx == Nx && r.method == "gmres_noprecond", results)
    nop_n = nrow(nop) > 0 ? nop[1, :N_rwg] : 0
    nop_i = nrow(nop) > 0 ? nop[1, :iters] : -1

    nf_iters = Int[]
    for c in nf_cutoffs_lam
        nf = filter(r -> r.Nx == Nx && r.method == "gmres_nf_$(c)lam", results)
        push!(nf_iters, nrow(nf) > 0 ? nf[1, :iters] : -1)
    end

    println(rpad(nop_n, 6), rpad(nop_i, 8),
            [rpad(i, 8) for i in nf_iters]...)
end

# Condition number growth
cond_rows = filter(r -> r.method == "direct_LU" && r.cond_est > 0, results)
if nrow(cond_rows) >= 2
    println("\n── Condition Number Growth (PEC EFIE) ──")
    for row in eachrow(cond_rows)
        println("  N=$(row.N_rwg)  cond(Z)=$(round(row.cond_est, sigdigits=4))")
    end
end
