# Mesh.jl — Simple mesh generation for flat plates and basic geometries

export make_rect_plate, triangle_area, triangle_center, triangle_normal

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
