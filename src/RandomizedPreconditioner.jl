# RandomizedPreconditioner.jl — Randomized subspace preconditioning for EFIE systems
#
# Two-level preconditioner:
#   Level 1 (subspace): Exact solve via randomized sketch Q and reduced system Z_r
#   Level 2 (complement): Diagonal (Jacobi) preconditioning on the complement of Q
#
# Algorithm:
#   1. Draw random sketch Ω ∈ C^{N×(k+p)} with p oversampling columns
#   2. Compute Y = Z Ω  (k+p matvecs)
#   3. SVD: Y ≈ U Σ Vᴴ, take Q = U[:, 1:k] (best rank-k subspace)
#   4. Reduced system: Z_r = Qᴴ Z Q ∈ C^{k×k}
#   5. Factorize Z_r
#   6. Preconditioner action:
#        P⁻¹ v = Q Z_r⁻¹ Qᴴ v + (I - QQᴴ) D⁻¹ v
#      where D = diag(Z) is the Jacobi complement preconditioner.
#
# For adjoint: P⁻ᴴ v = Q Z_r⁻ᴴ Qᴴ v + D⁻ᴴ (I - QQᴴ) v
#
# Legacy scalar-μ mode is retained for backward compatibility:
#   P⁻¹ v = Q Z_r⁻¹ Qᴴ v + μ (v - Q Qᴴ v)

export RandomizedPreconditionerData,
       build_randomized_preconditioner,
       update_randomized_preconditioner,
       apply_preconditioner,
       apply_preconditioner_adjoint,
       cache_MpOmega,
       PreconditionerOperator,
       PreconditionerAdjointOperator

"""
    RandomizedPreconditionerData

Stores the data for a randomized subspace preconditioner.

Fields:
- `Q`: N×k orthonormal basis spanning the sketch subspace
- `Z_r_fac`: LU factorization of k×k reduced system Qᴴ Z Q
- `mu`: shift parameter for the complement subspace (legacy scalar mode)
- `Omega`: N×k random sketch matrix (stored for recycling)
- `Y`: N×k sketch product Y = Z Ω (stored for incremental updates)
- `MpOmega`: cached products M_p Ω for recycling (or nothing)
- `D_inv`: diagonal complement preconditioner D⁻¹ = 1./diag(Z) (or nothing for scalar-μ mode)
"""
struct RandomizedPreconditionerData
    Q::Matrix{ComplexF64}
    Z_r_fac::LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}
    mu::ComplexF64
    Omega::Matrix{ComplexF64}
    Y::Matrix{ComplexF64}
    MpOmega::Union{Nothing, Vector{Matrix{ComplexF64}}}
    D_inv::Union{Nothing, Vector{ComplexF64}}
end

"""
    build_randomized_preconditioner(Z, k; seed=nothing, mu_mode=:auto, oversampling=10)

Build a randomized subspace preconditioner of rank k for the matrix Z.

With the default `mu_mode=:auto`, the preconditioner uses a two-level scheme:
  P⁻¹ v = Q Z_r⁻¹ Qᴴ v + (I - QQᴴ) D⁻¹ v

where Q ∈ C^{N×k} spans the dominant subspace of Z, Z_r = Qᴴ Z Q is the
reduced system, and D = diag(Z) provides Jacobi preconditioning on the complement.

With oversampling, the sketch uses k+p columns and SVD is used to extract
the optimal rank-k subspace (more robust than QR alone).

# Arguments
- `Z::Matrix{ComplexF64}`: the N×N system matrix
- `k::Int`: rank of the randomized approximation (sketch size)
- `seed`: optional RNG seed for reproducibility
- `mu_mode`: complement preconditioning strategy
  - `:auto` — two-level: diagonal of Z for complement (default)
  - `:diag` — scalar μ = 1 / mean(|diag(Z)|) (legacy)
  - `:trace` — scalar μ = N / tr(Z) (legacy)
  - a Number — use directly as scalar μ (legacy)
- `oversampling::Int`: number of extra sketch columns for better subspace (default 10)
"""
function build_randomized_preconditioner(Z::Matrix{ComplexF64}, k::Int;
                                          seed=nothing,
                                          mu_mode=:auto,
                                          oversampling::Int=10)
    N = size(Z, 1)
    k = min(k, N)
    p = min(oversampling, N - k)
    k_total = k + p

    # 1. Draw random sketch with oversampling
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    Omega_full = randn(rng, ComplexF64, N, k_total)

    # 2. Sketch: Y_full = Z Ω (N × k_total)
    Y_full = Z * Omega_full

    # 3. Extract subspace Q via SVD (better than QR with oversampling)
    if p > 0 && k < N
        U_Y, _, _ = svd(Y_full)
        Q = U_Y[:, 1:k]
    else
        F_qr = qr(Y_full)
        Q = Matrix(F_qr.Q)[:, 1:min(k, size(Y_full, 2))]
    end

    # Keep only k columns of Omega and Y for recycling
    Omega = Omega_full[:, 1:k]
    Y = Y_full[:, 1:k]

    # 4. Reduced system
    Z_r = Q' * Z * Q

    # 5. Factorize reduced system
    Z_r_fac = lu(Z_r)

    # 6. Complement treatment
    if mu_mode === :auto
        # Two-level: diagonal (Jacobi) preconditioning on complement
        d = diag(Z)
        D_inv = [abs(d[i]) > 1e-30 ? 1.0 / d[i] : 1.0 + 0.0im for i in 1:N]
        mu = 0.0 + 0.0im  # unused in :auto mode
    else
        # Legacy scalar-μ mode
        D_inv = nothing
        mu = _compute_mu(Z, mu_mode)
    end

    return RandomizedPreconditionerData(Q, Z_r_fac, mu, Omega, Y, nothing, D_inv)
end

"""
    _compute_mu(Z, mu_mode)

Compute the complement-subspace shift parameter μ (legacy scalar modes).
"""
function _compute_mu(Z::Matrix{ComplexF64}, mu_mode)
    if mu_mode === :diag
        d = diag(Z)
        mean_abs = sum(abs.(d)) / length(d)
        return mean_abs > 0 ? 1.0 / mean_abs : 1.0 + 0.0im
    elseif mu_mode === :trace
        t = tr(Z)
        N = size(Z, 1)
        return abs(t) > 0 ? ComplexF64(N) / t : 1.0 + 0.0im
    elseif mu_mode isa Number
        return ComplexF64(mu_mode)
    else
        error("Unknown mu_mode: $mu_mode")
    end
end

"""
    apply_preconditioner(P::RandomizedPreconditionerData, v::Vector{ComplexF64})

Apply the preconditioner.

Two-level mode (D_inv ≠ nothing):
  P⁻¹ v = Q Z_r⁻¹ Qᴴ v + (I - QQᴴ) D⁻¹ v

Legacy scalar mode (D_inv === nothing):
  P⁻¹ v = Q Z_r⁻¹ Qᴴ v + μ (v - Q Qᴴ v)
"""
function apply_preconditioner(P::RandomizedPreconditionerData, v::Vector{ComplexF64})
    # Project onto subspace
    Qhv = P.Q' * v                    # k-vector
    z_r_inv_Qhv = P.Z_r_fac \ Qhv    # k-vector: Z_r⁻¹ Qᴴ v

    # Subspace contribution: Q Z_r⁻¹ Qᴴ v
    subspace_part = P.Q * z_r_inv_Qhv

    if P.D_inv !== nothing
        # Two-level: (I - QQᴴ) D⁻¹ v
        D_inv_v = P.D_inv .* v                        # D⁻¹ v
        complement_part = D_inv_v .- P.Q * (P.Q' * D_inv_v)  # (I - QQᴴ) D⁻¹ v
    else
        # Legacy scalar: μ (v - Q Qᴴ v)
        complement_part = P.mu .* (v .- P.Q * Qhv)
    end

    return subspace_part .+ complement_part
end

"""
    apply_preconditioner_adjoint(P::RandomizedPreconditionerData, v::Vector{ComplexF64})

Apply the adjoint preconditioner.

Two-level mode:
  P⁻ᴴ v = Q Z_r⁻ᴴ Qᴴ v + D⁻ᴴ (I - QQᴴ) v

Legacy scalar mode:
  P⁻ᴴ v = Q Z_r⁻ᴴ Qᴴ v + μ̄ (v - Q Qᴴ v)
"""
function apply_preconditioner_adjoint(P::RandomizedPreconditionerData, v::Vector{ComplexF64})
    Qhv = P.Q' * v
    z_r_invH_Qhv = P.Z_r_fac' \ Qhv  # Z_r⁻ᴴ Qᴴ v

    subspace_part = P.Q * z_r_invH_Qhv

    if P.D_inv !== nothing
        # Two-level adjoint: D⁻ᴴ (I - QQᴴ) v
        complement_v = v .- P.Q * Qhv                        # (I - QQᴴ) v
        complement_part = conj.(P.D_inv) .* complement_v     # D⁻ᴴ (I - QQᴴ) v
    else
        # Legacy scalar adjoint: μ̄ (v - Q Qᴴ v)
        complement_part = conj(P.mu) .* (v .- P.Q * Qhv)
    end

    return subspace_part .+ complement_part
end

"""
    cache_MpOmega(Mp::Vector{<:AbstractMatrix}, Omega::Matrix{ComplexF64})

Precompute M_p Ω products for all patches. These are cached once and
reused for cheap preconditioner updates across optimization iterations.

Returns Vector{Matrix{ComplexF64}} of length P, each of size N×k.
"""
function cache_MpOmega(Mp::Vector{<:AbstractMatrix}, Omega::Matrix{ComplexF64})
    return [Matrix{ComplexF64}(Mp[p]) * Omega for p in eachindex(Mp)]
end

"""
    update_randomized_preconditioner(P_old, Z_new, delta_theta, Mp;
                                      mu_mode=:auto, MpOmega=nothing)

Cheaply update the randomized preconditioner after an impedance parameter change.

Exploits: Z(θ_new) = Z(θ_old) + ΔZ, where ΔZ = -Σ_p δθ_p M_p.

If `MpOmega` is provided (cached M_p Ω products), the sketch update
  Y_new = Y_old + ΔZ Ω = Y_old - Σ_p δθ_p (M_p Ω)
costs O(N·k·P) instead of O(N²·k) for a full rebuild.

Without `MpOmega`, falls back to Y_new = Z_new Ω (O(N²·k)).
"""
function update_randomized_preconditioner(P_old::RandomizedPreconditionerData,
                                           Z_new::Matrix{ComplexF64},
                                           delta_theta::AbstractVector,
                                           Mp::Vector{<:AbstractMatrix};
                                           mu_mode=:auto,
                                           MpOmega::Union{Nothing, Vector{Matrix{ComplexF64}}}=nothing,
                                           reactive::Bool=false)
    k = size(P_old.Q, 2)

    # Update sketch Y_new
    if MpOmega !== nothing
        # Incremental update: Y_new = Y_old - Σ_p δθ_p (M_p Ω)
        Y_new = copy(P_old.Y)
        for p in eachindex(delta_theta)
            if abs(delta_theta[p]) > 0
                coeff = reactive ? (1im * delta_theta[p]) : delta_theta[p]
                Y_new .-= coeff .* MpOmega[p]
            end
        end
    else
        # Full rebuild of sketch
        Y_new = Z_new * P_old.Omega
    end

    # Re-orthonormalize
    F_qr = qr(Y_new)
    Q_new = Matrix(F_qr.Q)[:, 1:k]

    # New reduced system
    Z_r_new = Q_new' * Z_new * Q_new
    Z_r_fac_new = lu(Z_r_new)

    # Complement treatment
    if mu_mode === :auto
        # Two-level: recompute diagonal of new Z (cheap: O(N))
        d_new = diag(Z_new)
        N = size(Z_new, 1)
        D_inv_new = [abs(d_new[i]) > 1e-30 ? 1.0 / d_new[i] : 1.0 + 0.0im for i in 1:N]
        mu_new = 0.0 + 0.0im
    else
        D_inv_new = P_old.D_inv  # preserve from old if it was set
        mu_new = _compute_mu(Z_new, mu_mode === :auto ? :diag : mu_mode)
    end

    return RandomizedPreconditionerData(Q_new, Z_r_fac_new, mu_new,
                                         P_old.Omega, Y_new, P_old.MpOmega, D_inv_new)
end

"""
    PreconditionerOperator

Callable wrapper for use with Krylov.jl as a left preconditioner.
Krylov.jl calls `mul!(y, M, x)` or `M * x` for preconditioning.
"""
struct PreconditionerOperator
    P::RandomizedPreconditionerData
end

Base.size(op::PreconditionerOperator) = (size(op.P.Q, 1), size(op.P.Q, 1))
Base.eltype(::PreconditionerOperator) = ComplexF64

function Base.:*(op::PreconditionerOperator, v::AbstractVector)
    return apply_preconditioner(op.P, Vector{ComplexF64}(v))
end

function LinearAlgebra.mul!(y::AbstractVector, op::PreconditionerOperator, x::AbstractVector)
    y .= apply_preconditioner(op.P, Vector{ComplexF64}(x))
    return y
end

"""
    PreconditionerAdjointOperator

Callable wrapper for the adjoint preconditioner P⁻ᴴ, for use in adjoint solves.
"""
struct PreconditionerAdjointOperator
    P::RandomizedPreconditionerData
end

Base.size(op::PreconditionerAdjointOperator) = (size(op.P.Q, 1), size(op.P.Q, 1))
Base.eltype(::PreconditionerAdjointOperator) = ComplexF64

function Base.:*(op::PreconditionerAdjointOperator, v::AbstractVector)
    return apply_preconditioner_adjoint(op.P, Vector{ComplexF64}(v))
end

function LinearAlgebra.mul!(y::AbstractVector, op::PreconditionerAdjointOperator, x::AbstractVector)
    y .= apply_preconditioner_adjoint(op.P, Vector{ComplexF64}(x))
    return y
end
