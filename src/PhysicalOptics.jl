# PhysicalOptics.jl — Physical Optics (PO) high-frequency RCS solver
#
# Computes scattered far-field and bistatic RCS using the PO approximation:
#   Illuminated face: J_s = 2(n̂ × H_inc)
#   Shadow face:      J_s = 0
#
# Works directly on triangle meshes — no RWG basis functions needed.
# Provides a fast, mesh-resolution-independent reference for MoM validation.

export POResult, solve_po

"""
    POResult

Result from the PO solver containing far-field, surface currents,
illumination mask, and problem metadata.
"""
struct POResult
    E_ff::Matrix{ComplexF64}     # (3, NΩ) scattered far-field
    J_s::Vector{CVec3}           # (Nt,) PO surface current per triangle centroid
    illuminated::BitVector       # (Nt,) which triangles are illuminated
    grid::SphGrid
    freq_hz::Float64
    k::Float64
end

"""
    solve_po(mesh, freq_hz, excitation; grid, quad_order=3, c0=299792458.0, eta0=376.730313668)

Compute the Physical Optics scattered far-field for a PEC body.

# Arguments
- `mesh::TriMesh`: triangle mesh of the scatterer
- `freq_hz`: frequency in Hz
- `excitation::PlaneWaveExcitation`: incident plane wave
- `grid::SphGrid`: spherical observation grid (default: 36×72)
- `quad_order::Int`: triangle quadrature order (1=centroid, 3=3-point, etc.)
- `c0, eta0`: physical constants

# Returns
`POResult` with far-field `E_ff`, surface currents `J_s`, illumination mask, etc.

# Physics
For a plane wave E_inc = E₀ pol exp(-jk·r), the PO surface current on
illuminated faces is J_s = 2(n̂ × H_inc), where H_inc = (k̂ × E_inc)/η₀.
The scattered far-field is integrated over all illuminated triangles using
quadrature for the phase variation exp(jk r̂·r').
"""
function solve_po(mesh::TriMesh, freq_hz::Real, excitation::PlaneWaveExcitation;
                  grid::SphGrid=make_sph_grid(36, 72),
                  quad_order::Int=3,
                  c0::Float64=299792458.0,
                  eta0::Float64=376.730313668)
    Nt = ntriangles(mesh)
    NΩ = length(grid.w)

    k = 2π * freq_hz / c0
    k_vec = excitation.k_vec
    E0 = excitation.E0
    pol = excitation.pol

    # Incident direction unit vector
    k_hat = Vec3(k_vec / norm(k_vec))

    # H_inc polarization direction: (k̂ × pol) / η₀
    h_pol = cross(k_hat, Vec3(pol))  # (k̂ × pol), unit-ish

    # Quadrature rule for triangle integration
    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # ─── Phase 1: Determine illumination and PO surface currents ───
    illuminated = falses(Nt)
    J_s = Vector{CVec3}(undef, Nt)
    # Precomputed per-triangle vector factor V_t = n̂_t × (k̂ × pol)
    # (constant over each triangle — only phase varies)
    V_t = Vector{Vec3}(undef, Nt)

    for t in 1:Nt
        n_hat = triangle_normal(mesh, t)

        # Illumination: wave hits the outward-facing side
        # k_hat points in propagation direction; face is illuminated when
        # the wave arrives from the normal side: k̂ · n̂ < 0
        if dot(k_hat, n_hat) < 0.0
            illuminated[t] = true
            V_t[t] = cross(n_hat, h_pol)  # n̂ × (k̂ × pol)

            # Store J_s at centroid for diagnostics
            rc = triangle_center(mesh, t)
            phase = exp(-1im * dot(k_vec, rc))
            J_s[t] = CVec3(complex.(2.0 * E0 / eta0 * V_t[t] * phase))
        else
            illuminated[t] = false
            V_t[t] = Vec3(0.0, 0.0, 0.0)
            J_s[t] = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        end
    end

    # ─── Phase 2: Far-field integration ───
    # E_scat(r̂) = (-jkη₀/4π) * (2E₀/η₀) * Σ_t [r̂ × (r̂ × V_t)] * I_t
    # where I_t = ∫_t exp(jk(r̂ - k̂)·r') dS'  (phase integral over triangle)
    #
    # The prefactor simplifies:
    # (-jkη₀/4π) * (2E₀/η₀) = -jk E₀ / (2π)

    prefactor = -1im * k * E0 / (2π)

    E_ff = zeros(ComplexF64, 3, NΩ)

    for q in 1:NΩ
        r_hat = Vec3(grid.rhat[:, q])
        # Phase shift direction: r̂ - k̂
        delta_k = r_hat - k_hat

        E_q = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)

        for t in 1:Nt
            !illuminated[t] && continue

            # Phase integral over triangle with quadrature
            A_t = triangle_area(mesh, t)
            pts = tri_quad_points(mesh, t, xi)
            phase_sum = zero(ComplexF64)
            @inbounds for qp in 1:Nq
                phase_sum += wq[qp] * exp(1im * k * dot(delta_k, pts[qp]))
            end
            I_t = 2.0 * A_t * phase_sum  # ∫ exp(jk δk·r') dS'

            # r̂ × (r̂ × V_t) = (r̂·V_t)r̂ - V_t
            Vt = V_t[t]
            proj = r_hat * dot(r_hat, Vt) - Vt

            E_q += CVec3(complex.(prefactor * proj * I_t))
        end

        E_ff[1, q] = E_q[1]
        E_ff[2, q] = E_q[2]
        E_ff[3, q] = E_q[3]
    end

    return POResult(E_ff, J_s, illuminated, grid, Float64(freq_hz), k)
end
