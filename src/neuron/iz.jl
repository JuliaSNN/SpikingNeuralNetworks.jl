@snn_kw struct IZParameter{FT = Float32}
    a::FT = 0.01
    b::FT = 0.2
    c::FT = -65
    d::FT = 2
end

@snn_kw mutable struct IZ{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractPopulation
    id::String = randstring(12)
    name::String = "IZ"
    param::IZParameter = IZParameter()
    N::Int32 = 100
    v::VFT = fill(-65.0, N)
    u::VFT = param.b * v
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
[Izhikevich Neuron](https://www.izhikevich.org/publications/spikes.htm)
"""
IZ

function integrate!(p::IZ, param::IZParameter, dt::Float32)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d = param
    @inbounds for i = 1:N
        v[i] += 0.5f0dt * (0.04f0v[i]^2 + 5.0f0v[i] + 140.0f0 - u[i] + I[i])
        v[i] += 0.5f0dt * (0.04f0v[i]^2 + 5.0f0v[i] + 140.0f0 - u[i] + I[i])
        u[i] += dt * (a * (b * v[i] - u[i]))
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > 30.0f0
        v[i] = ifelse(fire[i], c, v[i])
        u[i] += ifelse(fire[i], d, 0.0f0)
    end
end
