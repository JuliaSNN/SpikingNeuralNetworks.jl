@snn_kw struct IZParameter{FT=Float32}
    a::FT = 0.01
    b::FT = 0.2
    c::FT = -65
    d::FT = 2
end

@snn_kw mutable struct IZ{VFT=Vector{Float32},VBT=Vector{Bool}}
    param::IZParameter = IZParameter()
    N::Int32 = 100
    v::VFT = fill(-65.0, N)
    u::VFT = param.b * v
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end
vars(iz::IZ) = (:v=>iz.v, :u=>iz.u)

function integrate!(_du, _u, p::IZ, t)
    @unpack a, b, c, d = p.param
    I = p.I
    v, u = _u.x
    dv, du = _du.x
    dv .= 0.5 .* ((0.04f0 .* (v .^2)) .+ (5f0 .* v) .+ 140f0 .- u .+ I)
    dv .+= 0.5 .* ((0.04f0 .* ((dv .+ v) .^2)) .+ (5f0 .* (dv .+ v)) .+ 140f0 .- u .+ I)
    du .= a .* (b .* v - u)
end

function condition(iz::IZ, u)
    v, u = u.x
    any(_v->_v > 30f0, v)
end

function affect!(iz::IZ, u)
    v, u = u.x
    F = v .> 30f0
    v[F] .= iz.param.c
    u[F] .+= iz.param.d
    iz.fire .= F
end
