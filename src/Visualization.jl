# Visualization.jl — package-level mesh plotting utilities

export plot_mesh_wireframe, plot_mesh_comparison, save_mesh_preview

using Plots

function _realistic_axis_limits(meshes::Vector{TriMesh}; pad_frac::Float64=0.04)
    mins = [Inf, Inf, Inf]
    maxs = [-Inf, -Inf, -Inf]
    for mesh in meshes
        for i in 1:3
            mins[i] = min(mins[i], minimum(@view mesh.xyz[i, :]))
            maxs[i] = max(maxs[i], maximum(@view mesh.xyz[i, :]))
        end
    end

    spans = maxs .- mins
    max_span = max(maximum(spans), 1e-9)
    centers = (mins .+ maxs) ./ 2
    half = 0.5 * max_span * (1 + pad_frac)

    xlims = (centers[1] - half, centers[1] + half)
    ylims = (centers[2] - half, centers[2] + half)
    zlims = (centers[3] - half, centers[3] + half)
    return xlims, ylims, zlims
end

"""
    plot_mesh_wireframe(mesh; kwargs...)

Create a 3D wireframe plot of a mesh using package-native mesh segments.
"""
function plot_mesh_wireframe(mesh::TriMesh;
                             color=:steelblue,
                             title::AbstractString="Mesh",
                             camera::Tuple{Real,Real}=(30, 30),
                             linewidth::Real=0.7,
                             xlims=nothing,
                             ylims=nothing,
                             zlims=nothing,
                             kwargs...)
    seg = mesh_wireframe_segments(mesh)
    p = plot(
        seg.x, seg.y, seg.z;
        seriestype=:path3d,
        color=color,
        linewidth=linewidth,
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="z (m)",
        title=title,
        camera=camera,
        legend=false,
        grid=true,
        framestyle=:box,
        kwargs...,
    )

    if xlims !== nothing
        plot!(p; xlims=xlims)
    end
    if ylims !== nothing
        plot!(p; ylims=ylims)
    end
    if zlims !== nothing
        plot!(p; zlims=zlims)
    end
    return p
end

"""
    plot_mesh_comparison(mesh_a, mesh_b; kwargs...)

Create side-by-side 3D wireframe plots with shared axis limits and equal aspect.
"""
function plot_mesh_comparison(mesh_a::TriMesh, mesh_b::TriMesh;
                              title_a::AbstractString="Mesh A",
                              title_b::AbstractString="Mesh B",
                              color_a=:steelblue,
                              color_b=:darkorange,
                              camera::Tuple{Real,Real}=(30, 30),
                              size::Tuple{Int,Int}=(1200, 520),
                              pad_frac::Float64=0.04,
                              kwargs...)
    xlims, ylims, zlims = _realistic_axis_limits([mesh_a, mesh_b]; pad_frac=pad_frac)

    p1 = plot_mesh_wireframe(
        mesh_a;
        color=color_a,
        title=title_a,
        camera=camera,
        xlims=xlims,
        ylims=ylims,
        zlims=zlims,
        aspect_ratio=:equal,
        kwargs...,
    )
    p2 = plot_mesh_wireframe(
        mesh_b;
        color=color_b,
        title=title_b,
        camera=camera,
        xlims=xlims,
        ylims=ylims,
        zlims=zlims,
        aspect_ratio=:equal,
        kwargs...,
    )

    return plot(p1, p2; layout=(1, 2), size=size)
end

"""
    save_mesh_preview(mesh_a, mesh_b, out_prefix; kwargs...)

Generate and save side-by-side mesh preview plots as PNG and PDF.
Returns a named tuple with plot object and file paths.
"""
function save_mesh_preview(mesh_a::TriMesh, mesh_b::TriMesh, out_prefix::AbstractString; kwargs...)
    mkpath(dirname(out_prefix))
    p = plot_mesh_comparison(mesh_a, mesh_b; kwargs...)
    png_path = out_prefix * ".png"
    pdf_path = out_prefix * ".pdf"
    savefig(p, png_path)
    savefig(p, pdf_path)
    return (plot=p, png_path=png_path, pdf_path=pdf_path)
end
