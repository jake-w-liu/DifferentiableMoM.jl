# EFIE.jl — Dense assembly of the EFIE MoM impedance matrix
#
# Z_mn = <f_m, T[f_n]>  (EFIE operator only, without impedance sheet)
#
# Uses mixed-potential form:
#   Z_mn = -iωμ₀ [ ∫∫ f_m(r)·f_n(r') G(r,r') dS dS'
#                   - (1/k²) ∫∫ (∇·f_m)(∇'·f_n) G(r,r') dS dS' ]
#
# Note: the EFIE operator T[J] = P_t E^sca, with E^sca = -iωμ₀ ∫[I + ∇∇/k²]G·J dS'
# so <f_m, T[f_n]> = -iωμ₀ [ ∫∫ f_m·f_n G - (1/k²) ∫∫ (∇·f_m)(∇·f_n) G ] dS dS'

export assemble_Z_efie

"""
    assemble_Z_efie(mesh, rwg, k; quad_order=3, mesh_precheck=true, allow_boundary=true, require_closed=false)

Assemble the dense EFIE matrix Z_efie ∈ C^{N×N}.
`k` is the wavenumber (can be complex for complex-step).
"""
function assemble_Z_efie(mesh::TriMesh, rwg::RWGData, k;
                         quad_order::Int=3, eta0=376.730313668,
                         mesh_precheck::Bool=true,
                         allow_boundary::Bool=true,
                         require_closed::Bool=false,
                         area_tol_rel::Float64=1e-12)
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    N = rwg.nedges
    omega_mu0 = k * eta0   # ωμ₀ = k η₀

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    Nt = ntriangles(mesh)
    CT = complex(typeof(real(k)))
    Z = zeros(CT, N, N)

    # Precompute quadrature points and areas for all triangles
    quad_pts = Vector{Vector{Vec3}}(undef, Nt)
    areas    = Vector{Float64}(undef, Nt)
    for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t]    = triangle_area(mesh, t)
    end

    # Build lookup: for each basis function n, its two triangles
    # and precomputed RWG values at quad points
    rwg_vals = Vector{Vector{Vector{Vec3}}}(undef, N)   # rwg_vals[n][t_local][q]
    div_vals = Vector{Vector{Float64}}(undef, N)         # div_vals[n][t_local]
    tri_ids  = Vector{Vector{Int}}(undef, N)

    for n in 1:N
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        tri_ids[n] = [tp, tm]
        div_vals[n] = [div_rwg(rwg, n, tp), div_rwg(rwg, n, tm)]
        rwg_vals[n] = Vector{Vector{Vec3}}(undef, 2)
        for (it, t) in enumerate([tp, tm])
            rwg_vals[n][it] = [eval_rwg(rwg, n, quad_pts[t][q], t) for q in 1:Nq]
        end
    end

    # Double loop over basis functions (dense assembly)
    for m in 1:N
        for n in 1:N
            val = zero(CT)

            # Sum over triangles of m and n
            for (itm, tm) in enumerate(tri_ids[m])
                Am = areas[tm]
                for (itn, tn) in enumerate(tri_ids[n])
                    An = areas[tn]

                    if tm == tn
                        # ── Self-cell: singularity extraction ──
                        val += self_cell_contribution(
                            mesh, rwg, n, tm,
                            quad_pts[tm],
                            rwg_vals[m][itm],
                            rwg_vals[n][itn],
                            div_vals[m][itm],
                            div_vals[n][itn],
                            Am, wq, k)
                    else
                        # ── Non-self: standard product quadrature ──
                        for qm in 1:Nq
                            rm = quad_pts[tm][qm]
                            fm = rwg_vals[m][itm][qm]
                            dvm = div_vals[m][itm]

                            for qn in 1:Nq
                                rn = quad_pts[tn][qn]
                                fn = rwg_vals[n][itn][qn]
                                dvn = div_vals[n][itn]

                                G = greens(SVector{3}(rm), SVector{3}(rn), k)

                                vec_part = dot(fm, fn) * G
                                scl_part = dvm * dvn * G / (k^2)
                                weight = wq[qm] * wq[qn] * (2 * Am) * (2 * An)

                                val += (vec_part - scl_part) * weight
                            end
                        end
                    end
                end
            end

            Z[m, n] = -1im * omega_mu0 * val
        end
    end

    return Z
end
