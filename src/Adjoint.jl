# Adjoint.jl — Adjoint solve and gradient evaluation
#
# Adjoint eq:  Z† λ = ∂Φ/∂I* = Q I
# Gradient:    ∂J/∂θ_p = -2 Re{ λ† (∂Z/∂θ_p) I }
#            = -2 Re{ λ† (-M_p) I }
#            = +2 Re{ λ† M_p I }

export solve_adjoint, gradient_impedance, compute_objective

"""
    compute_objective(I, Q)

Compute the quadratic objective J = Re(I† Q I).
"""
function compute_objective(I::Vector{<:Number}, Q::Matrix{<:Number})
    return real(dot(I, Q * I))
end

"""
    solve_adjoint(Z, Q, I; solver=:direct, preconditioner=nothing, gmres_tol=1e-8, gmres_maxiter=200)

Solve the adjoint system: Z† λ = Q I
Returns λ ∈ C^N.

When `solver=:gmres`, uses GMRES with the adjoint preconditioner P⁻ᴴ.
"""
function solve_adjoint(Z::Matrix{<:Number}, Q::Matrix{<:Number},
                       I::Vector{<:Number};
                       solver::Symbol=:direct,
                       preconditioner=nothing,
                       gmres_tol::Float64=1e-8,
                       gmres_maxiter::Int=200)
    rhs = Q * I
    if solver == :direct
        return Z' \ rhs
    elseif solver == :gmres
        x, stats = solve_gmres_adjoint(Matrix{ComplexF64}(Z), Vector{ComplexF64}(rhs);
                                        preconditioner=preconditioner,
                                        tol=gmres_tol, maxiter=gmres_maxiter)
        return x
    else
        error("Unknown solver: $solver (expected :direct or :gmres)")
    end
end

"""
    gradient_impedance(Mp, I, lambda; reactive=false)

Compute the adjoint gradient for impedance parameters:
  g[p] = -2 Re{ λ† (∂Z/∂θ_p) I }

For resistive impedance (Z_s = θ_p, default):
  ∂Z/∂θ_p = -M_p  →  g[p] = +2 Re{ λ† M_p I }

For reactive impedance (Z_s = iθ_p, reactive=true):
  ∂Z/∂θ_p = -iM_p  →  g[p] = +2 Re{ i λ† M_p I } = -2 Im{ λ† M_p I }

Returns g ∈ R^P.
"""
function gradient_impedance(Mp::Vector{<:AbstractMatrix},
                            I::Vector{<:Number},
                            lambda::Vector{<:Number};
                            reactive::Bool=false)
    P = length(Mp)
    g = zeros(Float64, P)
    for p in 1:P
        lMI = dot(lambda, Mp[p] * I)
        if reactive
            # ∂Z/∂θ_p = -iM_p
            # g[p] = -2 Re{ λ† (-iM_p) I } = 2 Re{ i λ† M_p I } = -2 Im{ λ† M_p I }
            g[p] = -2 * imag(lMI)
        else
            # ∂Z/∂θ_p = -M_p
            # g[p] = -2 Re{ λ† (-M_p) I } = 2 Re{ λ† M_p I }
            g[p] = 2 * real(lMI)
        end
    end
    return g
end
