#!/usr/bin/env julia

using MAT

function main(args)
    length(args) == 2 || error("Usage: julia convert_aircraft_mat_to_obj.jl input.mat output.obj")
    input_mat = args[1]
    output_obj = args[2]

    data = matread(input_mat)
    haskey(data, "coord") || error("MAT file missing key 'coord'")
    haskey(data, "facet") || error("MAT file missing key 'facet'")

    coord = Matrix{Float64}(data["coord"])
    facet_raw = data["facet"]
    ndims(facet_raw) == 2 || error("facet must be a 2D matrix")
    size(coord, 2) == 3 || error("coord must have shape (Nv,3), got $(size(coord))")
    size(facet_raw, 2) >= 3 || error("facet must have at least 3 columns, got $(size(facet_raw))")

    tri = Int.(facet_raw[:, 1:3])
    if minimum(tri) == 0
        tri .+= 1
    end

    Nv = size(coord, 1)
    tmin, tmax = minimum(tri), maximum(tri)
    1 <= tmin || error("Triangle index min out of bounds: $tmin")
    tmax <= Nv || error("Triangle index max out of bounds: $tmax > Nv=$Nv")

    mkpath(dirname(output_obj))
    open(output_obj, "w") do io
        println(io, "# Converted from $(basename(input_mat))")
        for i in 1:Nv
            x, y, z = coord[i, 1], coord[i, 2], coord[i, 3]
            println(io, "v $x $y $z")
        end
        Nt = size(tri, 1)
        for t in 1:Nt
            i, j, k = tri[t, 1], tri[t, 2], tri[t, 3]
            println(io, "f $i $j $k")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
