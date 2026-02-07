# Solve.jl — Forward and adjoint linear system solves

export solve_forward, solve_system, assemble_full_Z

"""
    solve_forward(Z, v)

Solve Z I = v using direct factorization (for small/medium N).
Returns I ∈ C^N.
"""
function solve_forward(Z::Matrix{<:Number}, v::Vector{<:Number})
    return Z \ v
end

"""
    solve_system(Z, rhs)

General linear solve Z x = rhs.
"""
function solve_system(Z::Matrix{<:Number}, rhs::Vector{<:Number})
    return Z \ rhs
end

"""
    assemble_full_Z(Z_efie, Mp, theta; reactive=false)

Assemble the full MoM matrix: Z(θ) = Z_efie + Z_imp(θ)

For resistive impedance (default):  Z_imp = -Σ_p θ_p M_p
For reactive impedance:             Z_imp = -Σ_p (iθ_p) M_p
"""
function assemble_full_Z(Z_efie::Matrix{<:Number},
                         Mp::Vector{<:AbstractMatrix},
                         theta::AbstractVector;
                         reactive::Bool=false)
    Z = copy(Z_efie)
    for p in eachindex(theta)
        coeff = reactive ? (1im * theta[p]) : theta[p]
        Z .-= coeff .* Mp[p]
    end
    return Z
end
