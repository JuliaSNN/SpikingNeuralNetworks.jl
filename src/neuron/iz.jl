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

function integrate!(dv, v, p, t)
    @unpack a, b, c, d, I = p
    dv.x[1] .= 0.04f0 .* v.x[1] .^2 .+ 5f0 .* v.x[1] .+ 140f0 .- v.x[2] .+ I
    dv.x[2] .= a .* (b .* v.x[1] - v.x[2])
end

function fire(v,t,integrator)
    any(30f0 .- v.x[1] .> 0)
end

function affect!(integrator)
    F = integrator.u.x[1] .- 30f0 .> 0
    integrator.u.x[1][F] .= integrator.p.c
    integrator.u.x[2][F] .+= integrator.p.d
end

