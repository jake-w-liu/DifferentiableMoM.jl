# Excitation.jl — Incident field and excitation vector assembly
#
# v_m = -⟨f_m, E^inc_t⟩_Γ

export assemble_v_plane_wave

"""
    plane_wave_field(r, k_vec, E0, pol)

Evaluate a plane wave E^inc(r) = pol * E0 * exp(-i k_vec · r)
at point r. Convention: exp(+iωt).
"""
function plane_wave_field(r::Vec3, k_vec::Vec3, E0, pol::Vec3)
    return pol * E0 * exp(-1im * dot(k_vec, r))
end

"""
    assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)

Assemble the excitation vector v_m = -⟨f_m, E^inc_t⟩ for a plane wave.
"""
function assemble_v_plane_wave(mesh::TriMesh, rwg::RWGData,
                                k_vec::Vec3, E0, pol::Vec3;
                                quad_order::Int=3)
    N = rwg.nedges
    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    CT = ComplexF64
    v = zeros(CT, N)

    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = triangle_area(mesh, t)
            pts = tri_quad_points(mesh, t, xi)

            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = plane_wave_field(rq, k_vec, E0, pol)
                v[n] += -wq[q] * dot(fn, Einc) * (2 * A)
            end
        end
    end

    return v
end
