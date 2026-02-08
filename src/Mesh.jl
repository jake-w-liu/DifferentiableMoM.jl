# Mesh.jl — Simple mesh generation and geometry utilities

export make_rect_plate, read_obj_mesh, triangle_area, triangle_center, triangle_normal
export mesh_quality_report, mesh_quality_ok, assert_mesh_quality
export write_obj_mesh, repair_mesh_for_simulation, repair_obj_mesh

"""
    make_rect_plate(Lx, Ly, Nx, Ny)

Generate a triangulated rectangular plate in the xy-plane, centered at the
origin. Returns a `TriMesh` with `(Nx+1)*(Ny+1)` vertices and `2*Nx*Ny`
triangles.
"""
function make_rect_plate(Lx::Real, Ly::Real, Nx::Int, Ny::Int)
    Nv = (Nx + 1) * (Ny + 1)
    Nt = 2 * Nx * Ny

    xyz = zeros(3, Nv)
    tri = zeros(Int, 3, Nt)

    # Vertex grid
    dx = Lx / Nx
    dy = Ly / Ny
    idx = 0
    for jy in 0:Ny
        for jx in 0:Nx
            idx += 1
            xyz[1, idx] = -Lx / 2 + jx * dx
            xyz[2, idx] = -Ly / 2 + jy * dy
            xyz[3, idx] = 0.0
        end
    end

    # Linear index helper: (ix, iy) -> vertex id (0-based ix, iy)
    vidx(ix, iy) = iy * (Nx + 1) + ix + 1

    # Triangulation: two triangles per grid cell
    tidx = 0
    for jy in 0:Ny-1
        for jx in 0:Nx-1
            v1 = vidx(jx,   jy)
            v2 = vidx(jx+1, jy)
            v3 = vidx(jx+1, jy+1)
            v4 = vidx(jx,   jy+1)

            tidx += 1
            tri[1, tidx] = v1
            tri[2, tidx] = v2
            tri[3, tidx] = v3

            tidx += 1
            tri[1, tidx] = v1
            tri[2, tidx] = v3
            tri[3, tidx] = v4
        end
    end

    return TriMesh(xyz, tri)
end

function _bbox_diagonal(mesh::TriMesh)
    mins = map(i -> minimum(@view mesh.xyz[i, :]), 1:3)
    maxs = map(i -> maximum(@view mesh.xyz[i, :]), 1:3)
    return norm(Vec3(maxs...) - Vec3(mins...))
end

"""
    mesh_quality_report(mesh; area_tol_rel=1e-12, check_orientation=true)

Compute mesh-quality diagnostics for a triangle surface mesh.
The report includes:
- invalid triangles (index out of bounds or repeated vertices),
- degenerate triangles (area below tolerance),
- boundary-edge count,
- non-manifold-edge count (>2 incident triangles),
- orientation-conflict count on interior edges.
"""
function mesh_quality_report(mesh::TriMesh; area_tol_rel::Float64=1e-12, check_orientation::Bool=true)
    Nv = nvertices(mesh)
    Nt = ntriangles(mesh)

    size(mesh.xyz, 1) == 3 || error("Mesh xyz must have size (3, Nv)")
    size(mesh.tri, 1) == 3 || error("Mesh tri must have size (3, Nt)")

    scale = max(_bbox_diagonal(mesh), 1.0)
    area_tol_abs = area_tol_rel * scale^2

    invalid_triangles = Int[]
    degenerate_triangles = Int[]

    # edge_map[(i,j)] = directions of edge traversal in each incident triangle
    # direction +1 means (i->j) where i<j, -1 means (j->i)
    edge_map = Dict{Tuple{Int,Int}, Vector{Int8}}()

    for t in 1:Nt
        i1 = mesh.tri[1, t]
        i2 = mesh.tri[2, t]
        i3 = mesh.tri[3, t]

        valid_idx = (1 <= i1 <= Nv) && (1 <= i2 <= Nv) && (1 <= i3 <= Nv)
        distinct = (i1 != i2) && (i2 != i3) && (i3 != i1)
        if !(valid_idx && distinct)
            push!(invalid_triangles, t)
            continue
        end

        if triangle_area(mesh, t) <= area_tol_abs
            push!(degenerate_triangles, t)
        end

        for (a, b) in ((i1, i2), (i2, i3), (i3, i1))
            key = a < b ? (a, b) : (b, a)
            dir = a < b ? Int8(1) : Int8(-1)
            push!(get!(edge_map, key, Int8[]), dir)
        end
    end

    n_boundary_edges = 0
    n_nonmanifold_edges = 0
    n_orientation_conflicts = 0

    for dirs in values(edge_map)
        nd = length(dirs)
        if nd == 1
            n_boundary_edges += 1
        elseif nd == 2
            if check_orientation && dirs[1] == dirs[2]
                n_orientation_conflicts += 1
            end
        elseif nd > 2
            n_nonmanifold_edges += 1
        end
    end

    n_edges_total = length(edge_map)
    n_interior_edges = n_edges_total - n_boundary_edges - n_nonmanifold_edges

    return (
        n_vertices = Nv,
        n_triangles = Nt,
        n_edges_total = n_edges_total,
        n_interior_edges = n_interior_edges,
        n_boundary_edges = n_boundary_edges,
        n_nonmanifold_edges = n_nonmanifold_edges,
        n_orientation_conflicts = n_orientation_conflicts,
        n_invalid_triangles = length(invalid_triangles),
        n_degenerate_triangles = length(degenerate_triangles),
        invalid_triangles = invalid_triangles,
        degenerate_triangles = degenerate_triangles,
        area_tol_abs = area_tol_abs,
    )
end

"""
    mesh_quality_ok(report; allow_boundary=true, require_closed=false)

Return `true` if a mesh-quality report passes hard checks:
- no invalid triangles,
- no degenerate triangles,
- no non-manifold edges,
- no orientation conflicts,
- boundary edges allowed unless `allow_boundary=false` or `require_closed=true`.
"""
function mesh_quality_ok(report; allow_boundary::Bool=true, require_closed::Bool=false)
    if report.n_invalid_triangles > 0
        return false
    end
    if report.n_degenerate_triangles > 0
        return false
    end
    if report.n_nonmanifold_edges > 0
        return false
    end
    if report.n_orientation_conflicts > 0
        return false
    end
    if require_closed && report.n_boundary_edges > 0
        return false
    end
    if !allow_boundary && report.n_boundary_edges > 0
        return false
    end
    return true
end

"""
    assert_mesh_quality(mesh; allow_boundary=true, require_closed=false, area_tol_rel=1e-12)

Run mesh-quality checks and throw a detailed error if the mesh is unsuitable.
Returns the computed quality report on success.
"""
function assert_mesh_quality(mesh::TriMesh;
                             allow_boundary::Bool=true,
                             require_closed::Bool=false,
                             area_tol_rel::Float64=1e-12)
    report = mesh_quality_report(mesh; area_tol_rel=area_tol_rel, check_orientation=true)
    problems = String[]

    if report.n_invalid_triangles > 0
        sample = join(report.invalid_triangles[1:min(end, 5)], ", ")
        push!(problems, "invalid triangles: $(report.n_invalid_triangles) (sample: $sample)")
    end
    if report.n_degenerate_triangles > 0
        sample = join(report.degenerate_triangles[1:min(end, 5)], ", ")
        push!(problems, "degenerate triangles: $(report.n_degenerate_triangles) (sample: $sample), area_tol_abs=$(report.area_tol_abs)")
    end
    if report.n_nonmanifold_edges > 0
        push!(problems, "non-manifold edges: $(report.n_nonmanifold_edges)")
    end
    if report.n_orientation_conflicts > 0
        push!(problems, "orientation conflicts on interior edges: $(report.n_orientation_conflicts)")
    end
    if require_closed && report.n_boundary_edges > 0
        push!(problems, "boundary edges present but closed surface required: $(report.n_boundary_edges)")
    elseif !allow_boundary && report.n_boundary_edges > 0
        push!(problems, "boundary edges not allowed: $(report.n_boundary_edges)")
    end

    if !isempty(problems)
        msg = "Mesh quality precheck failed:\n  - " * join(problems, "\n  - ")
        error(msg)
    end

    return report
end

"""
    read_obj_mesh(path)

Read a triangle mesh from a Wavefront OBJ file and return a `TriMesh`.

Supported records:
- `v x y z`
- `f i j k ...` (triangles or polygons; polygons are fan-triangulated)

Texture/normal indices (`f v/t/n`) are ignored. Positive and negative OBJ
vertex indices are supported.
"""
function read_obj_mesh(path::AbstractString)
    vertices = Vector{NTuple{3,Float64}}()
    faces = Vector{NTuple{3,Int}}()

    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, "#") && continue

            if startswith(line, "v ")
                fields = split(line)
                length(fields) < 4 && error("Invalid OBJ vertex line: $line")
                x = parse(Float64, fields[2])
                y = parse(Float64, fields[3])
                z = parse(Float64, fields[4])
                push!(vertices, (x, y, z))

            elseif startswith(line, "f ")
                fields = split(line)[2:end]
                length(fields) < 3 && error("Invalid OBJ face line: $line")

                face_idx = Int[]
                for token in fields
                    vtoken = split(token, "/")[1]
                    isempty(vtoken) && error("Invalid OBJ face token: $token")
                    idx_raw = parse(Int, vtoken)
                    idx = idx_raw > 0 ? idx_raw : (length(vertices) + idx_raw + 1)
                    (1 <= idx <= length(vertices)) || error("OBJ face index out of range in line: $line")
                    push!(face_idx, idx)
                end

                v1 = face_idx[1]
                for j in 2:(length(face_idx) - 1)
                    push!(faces, (v1, face_idx[j], face_idx[j + 1]))
                end
            end
        end
    end

    isempty(vertices) && error("OBJ mesh has no vertices: $path")
    isempty(faces) && error("OBJ mesh has no faces: $path")

    xyz = zeros(Float64, 3, length(vertices))
    for i in eachindex(vertices)
        x, y, z = vertices[i]
        xyz[1, i] = x
        xyz[2, i] = y
        xyz[3, i] = z
    end

    tri = zeros(Int, 3, length(faces))
    for t in eachindex(faces)
        i1, i2, i3 = faces[t]
        tri[1, t] = i1
        tri[2, t] = i2
        tri[3, t] = i3
    end

    return TriMesh(xyz, tri)
end

"""
    write_obj_mesh(path, mesh; header="...")

Write a `TriMesh` to a Wavefront OBJ file using triangle faces.
"""
function write_obj_mesh(path::AbstractString, mesh::TriMesh; header::AbstractString="Exported by DifferentiableMoM")
    open(path, "w") do io
        println(io, "# $header")
        for i in 1:nvertices(mesh)
            println(io, "v $(mesh.xyz[1, i]) $(mesh.xyz[2, i]) $(mesh.xyz[3, i])")
        end
        for t in 1:ntriangles(mesh)
            println(io, "f $(mesh.tri[1, t]) $(mesh.tri[2, t]) $(mesh.tri[3, t])")
        end
    end
    return path
end

function _clean_mesh_triangles(mesh::TriMesh;
                               drop_invalid::Bool=true,
                               drop_degenerate::Bool=true,
                               area_tol_rel::Float64=1e-12)
    nv = nvertices(mesh)
    nt = ntriangles(mesh)
    tri = mesh.tri
    xyz = mesh.xyz

    scale = max(_bbox_diagonal(mesh), 1.0)
    area_tol_abs = area_tol_rel * scale^2

    keep_triangle = trues(nt)
    removed_invalid = Int[]
    removed_degenerate = Int[]

    for t in 1:nt
        i1 = tri[1, t]
        i2 = tri[2, t]
        i3 = tri[3, t]

        valid_idx = (1 <= i1 <= nv) && (1 <= i2 <= nv) && (1 <= i3 <= nv)
        distinct = (i1 != i2) && (i2 != i3) && (i3 != i1)

        if !(valid_idx && distinct)
            if drop_invalid
                keep_triangle[t] = false
                push!(removed_invalid, t)
                continue
            else
                error("Triangle $t is invalid and `drop_invalid=false`.")
            end
        end

        v1 = Vec3(xyz[:, i1])
        v2 = Vec3(xyz[:, i2])
        v3 = Vec3(xyz[:, i3])
        area = 0.5 * norm(cross(v2 - v1, v3 - v1))

        if area <= area_tol_abs
            if drop_degenerate
                keep_triangle[t] = false
                push!(removed_degenerate, t)
            else
                error("Triangle $t is degenerate (area=$area <= $area_tol_abs) and `drop_degenerate=false`.")
            end
        end
    end

    tri_clean = copy(tri[:, keep_triangle])
    cleaned_mesh = TriMesh(copy(xyz), tri_clean)
    return cleaned_mesh, removed_invalid, removed_degenerate, area_tol_abs
end

function _edge_orientation_adjacency(mesh::TriMesh)
    nt = ntriangles(mesh)
    edge_map = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int8}}}()

    for t in 1:nt
        i1 = mesh.tri[1, t]
        i2 = mesh.tri[2, t]
        i3 = mesh.tri[3, t]
        for (a, b) in ((i1, i2), (i2, i3), (i3, i1))
            key = a < b ? (a, b) : (b, a)
            dir = a < b ? Int8(1) : Int8(-1)
            push!(get!(edge_map, key, Tuple{Int,Int8}[]), (t, dir))
        end
    end

    adjacency = [Tuple{Int,Int8}[] for _ in 1:nt]
    for refs in values(edge_map)
        if length(refs) == 2
            (t1, d1) = refs[1]
            (t2, d2) = refs[2]
            parity = d1 == d2 ? Int8(1) : Int8(0)
            push!(adjacency[t1], (t2, parity))
            push!(adjacency[t2], (t1, parity))
        end
    end

    return adjacency
end

function _compute_orientation_flips(mesh::TriMesh)
    nt = ntriangles(mesh)
    adjacency = _edge_orientation_adjacency(mesh)

    flip_flag = fill(Int8(-1), nt)
    queue = Int[]

    for start in 1:nt
        if flip_flag[start] != -1
            continue
        end

        flip_flag[start] = 0
        empty!(queue)
        push!(queue, start)
        queue_index = 1

        while queue_index <= length(queue)
            t = queue[queue_index]
            queue_index += 1

            for (nbr, parity) in adjacency[t]
                expected = Int8(mod(Int(flip_flag[t]) + Int(parity), 2))
                if flip_flag[nbr] == -1
                    flip_flag[nbr] = expected
                    push!(queue, nbr)
                elseif flip_flag[nbr] != expected
                    error("Orientation repair failed: inconsistent winding constraints in triangle graph.")
                end
            end
        end
    end

    return flip_flag
end

function _apply_orientation_flips(mesh::TriMesh, flip_flag::Vector{Int8})
    tri = copy(mesh.tri)
    flipped_triangles = Int[]
    for t in 1:ntriangles(mesh)
        if flip_flag[t] == 1
            tri[2, t], tri[3, t] = tri[3, t], tri[2, t]
            push!(flipped_triangles, t)
        end
    end
    return TriMesh(copy(mesh.xyz), tri), flipped_triangles
end

"""
    repair_mesh_for_simulation(mesh;
        allow_boundary=true, require_closed=false, area_tol_rel=1e-12,
        drop_invalid=true, drop_degenerate=true,
        fix_orientation=true, strict_nonmanifold=true)

Repair a triangle mesh so it can pass solver prechecks:
- optionally remove invalid/degenerate triangles,
- orient triangles consistently across manifold interior edges.

Returns a named tuple containing the repaired mesh and before/after reports.
"""
function repair_mesh_for_simulation(mesh::TriMesh;
                                    allow_boundary::Bool=true,
                                    require_closed::Bool=false,
                                    area_tol_rel::Float64=1e-12,
                                    drop_invalid::Bool=true,
                                    drop_degenerate::Bool=true,
                                    fix_orientation::Bool=true,
                                    strict_nonmanifold::Bool=true)
    report_before = mesh_quality_report(mesh; area_tol_rel=area_tol_rel, check_orientation=true)

    cleaned_mesh, removed_invalid, removed_degenerate, area_tol_abs = _clean_mesh_triangles(
        mesh;
        drop_invalid=drop_invalid,
        drop_degenerate=drop_degenerate,
        area_tol_rel=area_tol_rel,
    )
    report_cleaned = mesh_quality_report(cleaned_mesh; area_tol_rel=area_tol_rel, check_orientation=true)

    if strict_nonmanifold && report_cleaned.n_nonmanifold_edges > 0
        error("Mesh repair cannot continue with non-manifold edges ($(report_cleaned.n_nonmanifold_edges)).")
    end

    repaired_mesh = cleaned_mesh
    flipped_triangles = Int[]
    if fix_orientation && report_cleaned.n_orientation_conflicts > 0
        flip_flag = _compute_orientation_flips(cleaned_mesh)
        repaired_mesh, flipped_triangles = _apply_orientation_flips(cleaned_mesh, flip_flag)
    end

    report_after = mesh_quality_report(repaired_mesh; area_tol_rel=area_tol_rel, check_orientation=true)
    assert_mesh_quality(
        repaired_mesh;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
        area_tol_rel=area_tol_rel,
    )

    return (
        mesh = repaired_mesh,
        before = report_before,
        cleaned = report_cleaned,
        after = report_after,
        removed_invalid = removed_invalid,
        removed_degenerate = removed_degenerate,
        flipped_triangles = flipped_triangles,
        area_tol_abs = area_tol_abs,
    )
end

"""
    repair_obj_mesh(input_path, output_path; kwargs...)

Read an OBJ mesh, repair it for solver prechecks, and write a repaired OBJ.
Returns the same metadata as `repair_mesh_for_simulation`, plus `output_path`.
"""
function repair_obj_mesh(input_path::AbstractString, output_path::AbstractString; kwargs...)
    mesh = read_obj_mesh(input_path)
    result = repair_mesh_for_simulation(mesh; kwargs...)
    write_obj_mesh(output_path, result.mesh; header="Repaired from $input_path by DifferentiableMoM")
    return (; result..., output_path=output_path)
end

"""
    triangle_area(mesh, t)

Compute the area of triangle `t` in the mesh.
"""
function triangle_area(mesh::TriMesh, t::Int)
    v1 = Vec3(mesh.xyz[:, mesh.tri[1, t]])
    v2 = Vec3(mesh.xyz[:, mesh.tri[2, t]])
    v3 = Vec3(mesh.xyz[:, mesh.tri[3, t]])
    return 0.5 * norm(cross(v2 - v1, v3 - v1))
end

"""
    triangle_center(mesh, t)

Compute the centroid of triangle `t`.
"""
function triangle_center(mesh::TriMesh, t::Int)
    v1 = Vec3(mesh.xyz[:, mesh.tri[1, t]])
    v2 = Vec3(mesh.xyz[:, mesh.tri[2, t]])
    v3 = Vec3(mesh.xyz[:, mesh.tri[3, t]])
    return (v1 + v2 + v3) / 3
end

"""
    triangle_normal(mesh, t)

Compute the outward unit normal of triangle `t`.
"""
function triangle_normal(mesh::TriMesh, t::Int)
    v1 = Vec3(mesh.xyz[:, mesh.tri[1, t]])
    v2 = Vec3(mesh.xyz[:, mesh.tri[2, t]])
    v3 = Vec3(mesh.xyz[:, mesh.tri[3, t]])
    n = cross(v2 - v1, v3 - v1)
    return n / norm(n)
end
