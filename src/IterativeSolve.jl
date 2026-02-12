# IterativeSolve.jl — GMRES iterative solver via Krylov.jl
#
# Provides iterative solve alternatives to the dense direct factorization
# in Solve.jl, with support for near-field sparse preconditioning.

using Krylov

export solve_gmres, solve_gmres_adjoint

"""
    solve_gmres(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8, maxiter=200, verbose=false)

Solve Z x = rhs using GMRES from Krylov.jl.

If `preconditioner` is a `NearFieldPreconditionerData`, it is applied via:
- `precond_side=:left` (default): left preconditioner M in Krylov.gmres
- `precond_side=:right`: right preconditioner N in Krylov.gmres

Returns `(x, stats)` where `stats` is the Krylov.jl convergence info.
"""
function solve_gmres(Z::Matrix{ComplexF64}, rhs::Vector{ComplexF64};
                     preconditioner::Union{Nothing, NearFieldPreconditionerData}=nothing,
                     precond_side::Symbol=:left,
                     tol::Float64=1e-8,
                     maxiter::Int=200,
                     verbose::Bool=false)
    if preconditioner === nothing
        x, stats = Krylov.gmres(Z, rhs;
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :right
        N_op = NearFieldOperator(preconditioner)
        x, stats = Krylov.gmres(Z, rhs;
                                 N=N_op,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    else
        M = NearFieldOperator(preconditioner)
        x, stats = Krylov.gmres(Z, rhs;
                                 M=M,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    return x, stats
end

"""
    solve_gmres_adjoint(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8, maxiter=200, verbose=false)

Solve Z† x = rhs using GMRES, with the adjoint preconditioner Z_nf⁻ᴴ.

This is used for the adjoint linear system in sensitivity analysis:
  Z†(θ) λ = ∂Φ/∂I*

Returns `(x, stats)`.
"""
function solve_gmres_adjoint(Z::Matrix{ComplexF64}, rhs::Vector{ComplexF64};
                              preconditioner::Union{Nothing, NearFieldPreconditionerData}=nothing,
                              precond_side::Symbol=:left,
                              tol::Float64=1e-8,
                              maxiter::Int=200,
                              verbose::Bool=false)
    if preconditioner === nothing
        x, stats = Krylov.gmres(Z', rhs;
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    else
        M_adj = NearFieldAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(Z', rhs;
                                 M=M_adj,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    return x, stats
end
