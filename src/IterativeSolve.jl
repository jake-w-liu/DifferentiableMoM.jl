# IterativeSolve.jl — GMRES iterative solver via Krylov.jl
#
# Provides iterative solve alternatives to the dense direct factorization
# in Solve.jl, with support for randomized and near-field preconditioning.

using Krylov

export solve_gmres, solve_gmres_adjoint

"""
    solve_gmres(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8, maxiter=200, verbose=false)

Solve Z x = rhs using GMRES from Krylov.jl.

If `preconditioner` is a `RandomizedPreconditionerData`, it is applied via:
- `precond_side=:left` (default): left preconditioner M in Krylov.gmres
- `precond_side=:right`: right preconditioner N in Krylov.gmres

Right preconditioning preserves the true residual norm in GMRES and can
perform better for non-normal matrices like the EFIE operator.

Returns `(x, stats)` where `stats` is the Krylov.jl convergence info.
"""
function solve_gmres(Z::Matrix{ComplexF64}, rhs::Vector{ComplexF64};
                     preconditioner::Union{Nothing, RandomizedPreconditionerData, NearFieldPreconditionerData}=nothing,
                     precond_side::Symbol=:left,
                     tol::Float64=1e-8,
                     maxiter::Int=200,
                     verbose::Bool=false)
    if preconditioner === nothing
        x, stats = Krylov.gmres(Z, rhs;
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif preconditioner isa NearFieldPreconditionerData
        M_nf = NearFieldOperator(preconditioner)
        if precond_side == :right
            x, stats = Krylov.gmres(Z, rhs;
                                     N=M_nf,
                                     rtol=tol, atol=0.0,
                                     itmax=maxiter,
                                     verbose=(verbose ? 1 : 0))
        else
            x, stats = Krylov.gmres(Z, rhs;
                                     M=M_nf,
                                     rtol=tol, atol=0.0,
                                     itmax=maxiter,
                                     verbose=(verbose ? 1 : 0))
        end
    elseif precond_side == :right
        N_op = PreconditionerOperator(preconditioner)
        x, stats = Krylov.gmres(Z, rhs;
                                 N=N_op,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    else
        M = PreconditionerOperator(preconditioner)
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

Solve Z† x = rhs using GMRES, with the adjoint preconditioner P⁻ᴴ.

This is used for the adjoint linear system in sensitivity analysis:
  Z†(θ) λ = ∂Φ/∂I*

For right preconditioning of the forward system (ZP⁻¹)(Py) = b, the adjoint
system becomes (P⁻ᴴZ†)(P^ᴴy) = rhs, so P⁻ᴴ is applied as left preconditioner
of the adjoint.

Returns `(x, stats)`.
"""
function solve_gmres_adjoint(Z::Matrix{ComplexF64}, rhs::Vector{ComplexF64};
                              preconditioner::Union{Nothing, RandomizedPreconditionerData, NearFieldPreconditionerData}=nothing,
                              precond_side::Symbol=:left,
                              tol::Float64=1e-8,
                              maxiter::Int=200,
                              verbose::Bool=false)
    if preconditioner === nothing
        x, stats = Krylov.gmres(Z', rhs;
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif preconditioner isa NearFieldPreconditionerData
        # Near-field adjoint: Z_nf⁻ᴴ
        M_nf_adj = NearFieldAdjointOperator(preconditioner)
        if precond_side == :right
            # Forward right-preconditioned: Z Z_nf⁻¹ y = b
            # Adjoint: Z_nf⁻ᴴ Z† y = rhs → left-precondition Z† with Z_nf⁻ᴴ
            x, stats = Krylov.gmres(Z', rhs;
                                     M=M_nf_adj,
                                     rtol=tol, atol=0.0,
                                     itmax=maxiter,
                                     verbose=(verbose ? 1 : 0))
        else
            x, stats = Krylov.gmres(Z', rhs;
                                     M=M_nf_adj,
                                     rtol=tol, atol=0.0,
                                     itmax=maxiter,
                                     verbose=(verbose ? 1 : 0))
        end
    elseif precond_side == :right
        # Forward: Z P⁻¹ y = b
        # Adjoint: (Z P⁻¹)† y = rhs → P⁻ᴴ Z† y = rhs
        # So P⁻ᴴ acts as LEFT preconditioner for Z†
        M_adj = PreconditionerAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(Z', rhs;
                                 M=M_adj,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    else
        M_adj = PreconditionerAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(Z', rhs;
                                 M=M_adj,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    return x, stats
end
