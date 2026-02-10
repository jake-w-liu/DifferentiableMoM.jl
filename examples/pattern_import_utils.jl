module PatternImportUtils

using CSV
using DataFrames
using LinearAlgebra

export PatternLite,
       parse_complex_i,
       load_pattern_csv,
       maybe_make_radiationpatterns_patterns,
       phi_cut_power

"""
    PatternLite

Lightweight pattern container with `U`, `x`, and `y` fields.
It matches the interface required by `make_pattern_feed`.
"""
struct PatternLite{T1<:Number,T2<:Real}
    U::Matrix{T1}
    x::Vector{T2}
    y::Vector{T2}
end

"""
    parse_complex_i(x) -> ComplexF64

Parse values like `"1.0-2.0i"` or `"3.0+4.0im"` to `ComplexF64`.
"""
function parse_complex_i(x)::ComplexF64
    if x isa Complex
        return ComplexF64(x)
    elseif x isa Real
        return ComplexF64(Float64(x), 0.0)
    end
    s = strip(String(x))
    if isempty(s)
        error("Cannot parse empty complex field.")
    end
    if occursin('i', s) && !occursin("im", s)
        s = replace(s, "i" => "im")
    end
    return parse(ComplexF64, s)
end

"""
    load_pattern_csv(path; drop_phi_endpoint=true, endpoint_tol=1e-9, duplicate_col_tol=1e-6)

Load a CSV containing columns `theta`, `phi`, `Etheta`, `Ephi` into regular
`(Nθ, Nϕ)` complex matrices.

Angles are returned in degrees as stored in the file.
If `phi` contains both 0 and 360 (duplicate periodic endpoint), the last
column is removed when `drop_phi_endpoint=true`.
"""
function load_pattern_csv(path::AbstractString;
                          drop_phi_endpoint::Bool=true,
                          endpoint_tol::Float64=1e-9,
                          duplicate_col_tol::Float64=1e-6)
    isfile(path) || error("Pattern CSV not found: $path")

    df = CSV.read(path, DataFrame)

    normalize_colname(nm) = lowercase(replace(strip(String(nm)), '\ufeff' => ""))
    col_lookup = Dict(normalize_colname(nm) => nm for nm in names(df))
    for key in ("theta", "phi", "etheta", "ephi")
        haskey(col_lookup, key) || error("CSV must contain columns: theta, phi, Etheta, Ephi")
    end
    θ_col = df[!, col_lookup["theta"]]
    ϕ_col = df[!, col_lookup["phi"]]
    Eθ_col = df[!, col_lookup["etheta"]]
    Eϕ_col = df[!, col_lookup["ephi"]]

    θ_vals = sort(unique(Float64.(θ_col)))
    ϕ_vals = sort(unique(Float64.(ϕ_col)))
    Nθ = length(θ_vals)
    Nϕ = length(ϕ_vals)
    nrow(df) == Nθ * Nϕ || error("CSV grid is incomplete: rows=$(nrow(df)) but Nθ×Nϕ=$(Nθ*Nϕ).")

    θ_to_i = Dict(v => i for (i, v) in enumerate(θ_vals))
    ϕ_to_j = Dict(v => j for (j, v) in enumerate(ϕ_vals))

    Fθ = zeros(ComplexF64, Nθ, Nϕ)
    Fϕ = zeros(ComplexF64, Nθ, Nϕ)

    for idx in 1:nrow(df)
        i = θ_to_i[Float64(θ_col[idx])]
        j = ϕ_to_j[Float64(ϕ_col[idx])]
        Fθ[i, j] = parse_complex_i(Eθ_col[idx])
        Fϕ[i, j] = parse_complex_i(Eϕ_col[idx])
    end

    dropped_endpoint = false
    endpoint_mismatch = 0.0
    if drop_phi_endpoint && Nϕ ≥ 2 && abs((ϕ_vals[end] - ϕ_vals[1]) - 360.0) ≤ endpoint_tol
        endpoint_mismatch = max(maximum(abs.(Fθ[:, 1] .- Fθ[:, end])),
                                maximum(abs.(Fϕ[:, 1] .- Fϕ[:, end])))
        ref = max(maximum(abs.(Fθ)), maximum(abs.(Fϕ)), 1.0)
        if endpoint_mismatch / ref > duplicate_col_tol
            @warn "phi includes a 360° endpoint but first/last columns are not close (relative mismatch=$(endpoint_mismatch / ref)). Dropping last column anyway to enforce open period."
        end
        ϕ_vals = ϕ_vals[1:end-1]
        Fθ = Fθ[:, 1:end-1]
        Fϕ = Fϕ[:, 1:end-1]
        dropped_endpoint = true
    end

    return (
        theta_deg = θ_vals,
        phi_deg = ϕ_vals,
        Ftheta = Fθ,
        Fphi = Fϕ,
        dropped_phi_endpoint = dropped_endpoint,
        endpoint_mismatch = endpoint_mismatch,
    )
end

"""
    maybe_make_radiationpatterns_patterns(theta_deg, phi_deg, Ftheta, Fphi)

Try creating `RadiationPatterns.Pattern` objects. If `RadiationPatterns.jl`
cannot be loaded in the current environment, return `PatternLite` objects.
"""
function maybe_make_radiationpatterns_patterns(theta_deg::AbstractVector{<:Real},
                                               phi_deg::AbstractVector{<:Real},
                                               Ftheta::AbstractMatrix{<:Number},
                                               Fphi::AbstractMatrix{<:Number})
    θ = Float64.(collect(theta_deg))
    ϕ = Float64.(collect(phi_deg))
    Fθ = ComplexF64.(Matrix(Ftheta))
    Fϕ = ComplexF64.(Matrix(Fphi))

    try
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                @eval import RadiationPatterns
            end
        end
        Eθ_pattern = RadiationPatterns.Pattern(Fθ, θ, ϕ)
        Eϕ_pattern = RadiationPatterns.Pattern(Fϕ, θ, ϕ)
        return (
            Etheta_pattern = Eθ_pattern,
            Ephi_pattern = Eϕ_pattern,
            backend = :radiationpatterns,
            message = "Using RadiationPatterns.Pattern",
        )
    catch err
        msg = sprint(showerror, err)
        length(msg) > 300 && (msg = msg[1:300] * "...")
        @warn "RadiationPatterns.jl unavailable in this environment; using local PatternLite fallback." error=msg
        Eθ_pattern = PatternLite(Fθ, θ, ϕ)
        Eϕ_pattern = PatternLite(Fϕ, θ, ϕ)
        return (
            Etheta_pattern = Eθ_pattern,
            Ephi_pattern = Eϕ_pattern,
            backend = :patternlite,
            message = "RadiationPatterns load failed; fallback PatternLite",
        )
    end
end

"""
    phi_cut_power(theta_deg, phi_deg, Ftheta, Fphi; phi_target_deg=0.0)

Return normalized power cut at `phi_target_deg` from complex `Ftheta/Fphi`
coefficients.
"""
function phi_cut_power(theta_deg::AbstractVector{<:Real},
                       phi_deg::AbstractVector{<:Real},
                       Ftheta::AbstractMatrix{<:Number},
                       Fphi::AbstractMatrix{<:Number};
                       phi_target_deg::Float64=0.0)
    ϕ = Float64.(collect(phi_deg))
    j = argmin(abs.(ϕ .- phi_target_deg))
    P = abs2.(ComplexF64.(Ftheta[:, j])) .+ abs2.(ComplexF64.(Fphi[:, j]))
    P ./= max(maximum(P), 1e-30)
    return (
        theta_deg = Float64.(collect(theta_deg)),
        phi_used_deg = ϕ[j],
        power = P,
    )
end

end # module
