# PeriodicEFIE.jl — Periodic EFIE assembly via image correction
#
# Assembles Z_per = Z_free + Z_correction, where:
#   Z_free      = standard free-space EFIE (existing code, handles singularity)
#   Z_correction = contribution from periodic images (m,n) ≠ (0,0)
#
# The correction uses greens_periodic_correction() which is smooth (no 1/R
# singularity), so standard product quadrature suffices for all entries.

export assemble_Z_efie_periodic

"""
    assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order=3, eta0=376.730313668)

Assemble the dense periodic EFIE matrix `Z_per ∈ C^{N×N}` for a unit cell
with 2D periodicity defined by `lattice::PeriodicLattice`.

Strategy:
  Z_per = Z_free + Z_correction

- Z_free: standard free-space EFIE (with singularity extraction for self-cells)
- Z_correction: image sum using ΔG = G_per - G_0 (smooth, no singularity)

Both use the mixed-potential form:
  Z_mn = -iωμ₀ [ ∫∫ f_m·f_n G dS dS' - (1/k²) ∫∫ (∇·f_m)(∇'·f_n) G dS dS' ]
"""
function assemble_Z_efie_periodic(mesh::TriMesh, rwg::RWGData, k,
                                  lattice::PeriodicLattice;
                                  quad_order::Int=3,
                                  eta0::Float64=376.730313668)
    # Step 1: Free-space EFIE (handles self-cell singularity)
    Z_free = assemble_Z_efie(mesh, rwg, k;
                             quad_order=quad_order, eta0=eta0,
                             mesh_precheck=false)

    # Step 2: Periodic image correction (smooth, no singularity)
    Z_corr = _assemble_periodic_correction(mesh, rwg, k, lattice;
                                            quad_order=quad_order, eta0=eta0)

    return Z_free + Z_corr
end

"""
Assemble the periodic correction matrix using ΔG = G_per - G_0.
Since ΔG is smooth everywhere, standard product quadrature is used for all entries.
"""
function _assemble_periodic_correction(mesh::TriMesh, rwg::RWGData, k,
                                       lattice::PeriodicLattice;
                                       quad_order::Int=3,
                                       eta0::Float64=376.730313668)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    omega_mu0 = k * eta0

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # Precompute quadrature points and areas
    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas = [triangle_area(mesh, t) for t in 1:Nt]

    # Precompute RWG values and divergences
    tri_ids = zeros(Int, 2, N)
    div_vals = zeros(Float64, 2, N)
    rwg_vals = Vector{NTuple{2,Vector{Vec3}}}(undef, N)

    for n in 1:N
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        tri_ids[1, n] = tp
        tri_ids[2, n] = tm
        div_vals[1, n] = div_rwg(rwg, n, tp)
        div_vals[2, n] = div_rwg(rwg, n, tm)
        vals_p = [eval_rwg(rwg, n, quad_pts[tp][q], tp) for q in 1:Nq]
        vals_m = [eval_rwg(rwg, n, quad_pts[tm][q], tm) for q in 1:Nq]
        rwg_vals[n] = (vals_p, vals_m)
    end

    CT = ComplexF64
    Z_corr = zeros(CT, N, N)

    @inbounds for m_idx in 1:N
        for n_idx in 1:N
            val = zero(CT)

            for itm in 1:2
                tm = tri_ids[itm, m_idx]
                Am = areas[tm]
                dvm = div_vals[itm, m_idx]
                fm_vals = itm == 1 ? rwg_vals[m_idx][1] : rwg_vals[m_idx][2]

                for itn in 1:2
                    tn = tri_ids[itn, n_idx]
                    An = areas[tn]
                    dvn = div_vals[itn, n_idx]
                    fn_vals = itn == 1 ? rwg_vals[n_idx][1] : rwg_vals[n_idx][2]

                    # Product quadrature with ΔG (smooth for all triangle pairs)
                    for qm in 1:Nq
                        rm = quad_pts[tm][qm]
                        fm = fm_vals[qm]
                        for qn in 1:Nq
                            rn = quad_pts[tn][qn]
                            fn = fn_vals[qn]

                            dG = greens_periodic_correction(rm, rn, k, lattice)
                            vec_part = dot(fm, fn) * dG
                            scl_part = dvm * dvn * dG / (k^2)
                            weight = wq[qm] * wq[qn] * (2 * Am) * (2 * An)
                            val += (vec_part - scl_part) * weight
                        end
                    end
                end
            end

            Z_corr[m_idx, n_idx] = -1im * omega_mu0 * val
        end
    end

    return Z_corr
end
