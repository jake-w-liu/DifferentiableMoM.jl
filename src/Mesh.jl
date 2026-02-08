# Mesh.jl — Simple mesh generation and geometry utilities

export make_rect_plate, read_obj_mesh, triangle_area, triangle_center, triangle_normal

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
