# RWG.jl — Rao-Wilton-Glisson basis function construction and evaluation

export build_rwg, eval_rwg, div_rwg, basis_triangles

"""
    build_rwg(mesh::TriMesh; precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12)

Construct RWG basis functions from interior edges of the triangle mesh.
Each interior edge shared by two triangles defines one RWG basis function.

When `precheck=true` (default), run mesh-quality checks before basis
construction and error out on invalid/degenerate/non-manifold or
orientation-inconsistent meshes.
"""
function build_rwg(mesh::TriMesh;
                   precheck::Bool=true,
                   allow_boundary::Bool=true,
                   require_closed::Bool=false,
                   area_tol_rel::Float64=1e-12)
    if precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    Nt = ntriangles(mesh)

    # Map sorted edge (v_lo, v_hi) -> list of (triangle, local_edge_idx)
    edge_tris = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for t in 1:Nt
        for le in 1:3
            v1 = mesh.tri[le, t]
            v2 = mesh.tri[mod1(le + 1, 3), t]
            key = v1 < v2 ? (v1, v2) : (v2, v1)
            if !haskey(edge_tris, key)
                edge_tris[key] = Tuple{Int,Int}[]
            end
            push!(edge_tris[key], (t, le))
        end
    end

    # Interior edges are shared by exactly 2 triangles
    tplus_vec   = Int[]
    tminus_vec  = Int[]
    evert_arr   = Vector{Tuple{Int,Int}}()
    vplus_opp_vec  = Int[]
    vminus_opp_vec = Int[]
    len_vec     = Float64[]
    area_p_vec  = Float64[]
    area_m_vec  = Float64[]

    for (edge_key, tlist) in edge_tris
        length(tlist) == 2 || continue  # skip boundary edges

        (t1, le1) = tlist[1]
        (t2, le2) = tlist[2]

        # Opposite vertex in each triangle
        # For local edge le, the opposite vertex is the one NOT on the edge
        function opposite_vertex(t, le)
            # edge le connects tri[le,t] and tri[mod1(le+1,3),t]
            # opposite is tri[mod1(le+2,3),t]
            return mesh.tri[mod1(le + 2, 3), t]
        end

        vopp1 = opposite_vertex(t1, le1)
        vopp2 = opposite_vertex(t2, le2)

        # T+ is the triangle whose opposite vertex is on the "positive" side
        # Convention: T+ has the opposite vertex such that the RWG diverges positively
        # We use t1 as T+ and t2 as T-
        push!(tplus_vec, t1)
        push!(tminus_vec, t2)
        push!(evert_arr, edge_key)
        push!(vplus_opp_vec, vopp1)
        push!(vminus_opp_vec, vopp2)

        r1 = _mesh_vertex(mesh, edge_key[1])
        r2 = _mesh_vertex(mesh, edge_key[2])
        push!(len_vec, norm(r2 - r1))

        push!(area_p_vec, triangle_area(mesh, t1))
        push!(area_m_vec, triangle_area(mesh, t2))
    end

    nedges = length(tplus_vec)
    evert = zeros(Int, 2, nedges)
    for n in 1:nedges
        evert[1, n] = evert_arr[n][1]
        evert[2, n] = evert_arr[n][2]
    end

    return RWGData(mesh, nedges, tplus_vec, tminus_vec, evert,
                   vplus_opp_vec, vminus_opp_vec, len_vec,
                   area_p_vec, area_m_vec)
end

"""
    eval_rwg(rwg, n, r, t)

Evaluate RWG basis function `n` at point `r` on triangle `t`.
Returns a Vec3. Returns zero if `t` is not part of basis function `n`.

For T+: f_n(r) = (l_n / 2A+) * (r - r_opp+)
For T-: f_n(r) = (l_n / 2A-) * (r_opp- - r)
"""
function eval_rwg(rwg::RWGData, n::Int, r::Vec3, t::Int)
    if t == rwg.tplus[n]
        r_opp = _mesh_vertex(rwg.mesh, rwg.vplus_opp[n])
        return (rwg.len[n] / (2.0 * rwg.area_plus[n])) * (r - r_opp)
    elseif t == rwg.tminus[n]
        r_opp = _mesh_vertex(rwg.mesh, rwg.vminus_opp[n])
        return (rwg.len[n] / (2.0 * rwg.area_minus[n])) * (r_opp - r)
    else
        return Vec3(0.0, 0.0, 0.0)
    end
end

"""
    div_rwg(rwg, n, t)

Compute the surface divergence of RWG basis function `n` on triangle `t`.
For T+: div f_n = l_n / A+
For T-: div f_n = -l_n / A-
"""
function div_rwg(rwg::RWGData, n::Int, t::Int)
    if t == rwg.tplus[n]
        return rwg.len[n] / rwg.area_plus[n]
    elseif t == rwg.tminus[n]
        return -rwg.len[n] / rwg.area_minus[n]
    else
        return 0.0
    end
end

"""
    basis_triangles(rwg, n)

Return the two triangle indices supporting basis function `n`.
"""
basis_triangles(rwg::RWGData, n::Int) = (rwg.tplus[n], rwg.tminus[n])
